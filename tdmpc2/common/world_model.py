from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from common import layers, math, init
from tensordict.nn import TensorDictParams


class WorldModel(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    Can be used for both single-task and multi-task experiments.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.multitask:
            self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
            self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
            for i in range(len(cfg.tasks)):
                self._action_masks[i, :cfg.action_dims[i]] = 1.
        self._encoder = layers.enc(cfg)
        self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
        self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
        self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
        self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
        self.init()

    def init(self):
        # Create params
        self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
        self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

        with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
            self._detach_Qs = deepcopy(self._Qs)
            self._target_Qs = deepcopy(self._Qs)
        self._detach_Qs = deepcopy(self._Qs)
        self._target_Qs = deepcopy(self._Qs)

        # Assign params to modules
        self._detach_Qs.params = self._detach_Qs_params
        self._target_Qs.params = self._target_Qs_params


    def __repr__(self):
        repr = 'TD-MPC2 World Model\n'
        modules = ['Encoder', 'Dynamics', 'Reward', 'Policy prior', 'Q-functions']
        for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._pi, self._Qs]):
            repr += f"{modules[i]}: {m}\n"
        repr += "Learnable parameters: {:,}".format(self.total_params)
        return repr

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.init()
        return self

    def train(self, mode=True): 
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        # self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)
        # M: not update 'edge_index'
        new_tensordict = self._detach_Qs_params.exclude("edge_index")
        self._target_Qs_params.lerp_(new_tensordict, self.cfg.tau)

    def task_emb(self, x, task):
        """
        Continuous task embedding for multi-task experiments.
        Retrieves the task embedding for a given task ID `task`
        and concatenates it to the input `x`.
        """
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs, task):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        if self.cfg.multitask:
            obs = self.task_emb(obs, task)
        if self.cfg.obs == 'rgb' and obs.ndim == 5:
            return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
        return self._encoder[self.cfg.obs](obs)

    def next(self, z, a, task):
        """
        Predicts the next latent state given the current latent state and action.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._dynamics(z)

    def reward(self, z, a, task):
        """
        Predicts instantaneous (single-step) reward.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def pi(self, z, task):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        # Gaussian policy prior
        mu, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mu)

        if self.cfg.multitask: # Mask out unused action dimensions
            mu = mu * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else: # No masking
            action_dims = None

        log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
        pi = mu + eps * log_std.exp()
        mu, pi, log_pi = math.squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def Q(self, z, a, task, return_type='min', target=False, detach=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled Q-values.
            - `avg`: return the average of two randomly subsampled Q-values.
            - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {'min', 'avg', 'all'}

        if self.cfg.multitask:
            z = self.task_emb(z, task)

        z = torch.cat([z, a], dim=-1)
        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        out = qnet(z)

        if return_type == 'all':
            return out

        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q = math.two_hot_inv(out[qidx], self.cfg)
        if return_type == "min":
            return Q.min(0).values
        return Q.sum(0) / 2


class FacWorldModel(WorldModel):
    """
    Factored TD-MPC2 implicit world model architecture.
    Can be used for both single-task and multi-task experiments.
    """

    def __init__(self, cfg):

        nn.Module.__init__(self)

        # TODO: multi-task not supported yet
        # M: current seperate agents by the action
        self.num_nodes = cfg.action_dim # number of agents
        self.num_edges = self.num_nodes * (self.num_nodes - 1) // 2 # correlated reward graph
        self.action_dim_node = 1
        self.latent_dim_node = cfg.latent_dim // cfg.action_dim  
        cfg.latent_dim = cfg.latent_dim // cfg.action_dim * cfg.action_dim
        cfg.mlp_dim = cfg.latent_dim 
        cfg.simnorm_dim = 5 # to consider not divisible
        cfg.temperature = 1.0 # larger value, more random 

        # rewrite all init functions in WorldModel
        self.cfg = cfg

        # encoder
        self._encoder = layers.enc(cfg) # global encoder
        # M: independent encoder
        # self._encoder = layers.FullyConnectedGraph(self.num_nodes, cfg.obs_shape['state'][0], max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], self.latent_dim_node, act=layers.SimNorm(cfg)) 
        # self.encode = self.ind_encode

        # Define factored dynamics/rewards/Q
        self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
        # self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
        # self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
        # self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
        self._dynamics = layers.FullyConnectedGraph(self.num_nodes, self.latent_dim_node + self.action_dim_node, 2*[cfg.mlp_dim // self.num_nodes], self.latent_dim_node, act=layers.SimNorm(cfg)) 
        self._reward = layers.FullyConnectedGraph(self.num_nodes, self.latent_dim_node + self.action_dim_node, 2*[cfg.mlp_dim // self.num_nodes], max(cfg.num_bins, 1))
        self._Qs = layers.Ensemble([layers.FullyConnectedGraph(self.num_nodes, self.latent_dim_node + self.action_dim_node, 2*[cfg.mlp_dim // self.num_nodes], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])

        self.apply(init.weight_init)
        # M: initialize certain value-related weights
        # init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
        self.init()

        # M: edges for reward/value function TODO: fail on compiled graph
        # TODO: edge_probs/edge_logits, which is better?
        edge_type = "auto_edges" # ["full_edges", "none_edges", "topo_edges", "auto_edges"]
        self.temperature = 0.01
        # initialize
        if edge_type == "full_edges":  # fully connected graph
            edge_probs = torch.ones((self.num_edges))
        elif edge_type == "zero_edges": # nodes-only graph
            edge_probs = torch.zeros((self.num_edges))
        elif edge_type == "topo_edges": # connected by robots' topology
            edge_probs = torch.zeros((self.num_edges))
            for i in [0,2,5,12,14]: # special for walker robot
                edge_probs[i] = 1.0
        else: 
            edge_probs = torch.rand(self.num_edges) # in [0, 1]
        # trainable
        if "auto" in edge_type:
            self.edge_probs = nn.Parameter(edge_probs, requires_grad=True)  
        else:
            self.edge_probs = nn.Parameter(edge_probs, requires_grad=False)  

    def adjacency_matrix(self, hard=False, temperature=1.0):
        device = self.edge_probs.device
        edge_index = torch.combinations(torch.arange(self.num_nodes), r=2).T.to(device)
        src, dest = edge_index
        edge_probs = self.edge_probs.data
        adj_matrix = torch.eye(self.num_nodes).to(device) * 0.5 # for reference of medium color
        for i,(s,d) in enumerate(zip(src, dest)):
            if hard:
               adj_matrix[s][d] = 0.0 if edge_probs[i] < 0.5 else 1.0
               adj_matrix[d][s] = adj_matrix[s][d]
               adj_matrix[s][s] = 1.0
               adj_matrix[d][d] = 1.0
            else:
               adj_matrix[s][d] = edge_probs[i]
               adj_matrix[d][s] = edge_probs[i]
        return adj_matrix

    def _generate_node_features(self, z, a):
        latent_node = torch.reshape(z, (*z.shape[:-1], self.num_nodes, self.latent_dim_node))
        action_node = torch.reshape(a, (*a.shape[:-1], self.num_nodes, self.action_dim_node))
        node_features = torch.cat([latent_node, action_node], dim=-1) # [num_step*num_traj, num_nodes, node_dim]
        return node_features

    def ind_encode(self, obs, task):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        # no multi-task and rgb support now
        obs_dim = obs.shape
        node_obs = obs.unsqueeze(-2).repeat(*([1]*(len(obs_dim)-1)), self.num_nodes, 1) # [..., num_nodes, obs_dim]
        node_z, _ = self._encoder(node_obs)
        return node_z.view(*obs_dim[:-1], -1).contiguous()
        
    def rescale_edges(self, edges):
        """
            Args:
                - edges: [..., num_edges]
            Returns: 
                - edges: [..., num_edges]
        """
        if self.training and self.edge_probs.requires_grad : # keep gradients
            # TODO: now edge weights are same across the whole batch, try sample differently
            edge_weights = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(probs=self.edge_probs, temperature=self.temperature).rsample()
            # # M: STE for less training-testing mismatch
            # soft_weights = RelaxedBernoulli(
            #     logits=self.edge_logits, 
            #     temperature=self.temperature
            # ).rsample()
            # hard_weights = (soft_weights > 0.5).float()
            # edge_weights = hard_weights.detach() + soft_weights - soft_weights.detach()
        else: # remove gradients
            edge_weights = (self.edge_probs > 0.5).float()
        return (edges.transpose(-2, -1) * edge_weights).transpose(-2, -1)

    def next(self, z, a, task):
        """
        Predicts the next latent state given the current latent state and action.
        Args:
            - z: [*, hidden_dim] * might be 1 or 2 dims
            - a: [*, action_dim] 
        """
        node_features = self._generate_node_features(z, a)
        x, _ = self._dynamics(node_features) 
        x = torch.reshape(x, (*x.shape[:-2], -1)) # [*, hidden_dim]
        return x

    def reward(self, z, a, task):
        """
        Predicts instantaneous (single-step) reward.
        """
        node_features = self._generate_node_features(z, a)
        reward_nodes, reward_edges = self._reward(node_features)  # [*, num_nodes, num_bins], [*, num_edges, num_bins]
        reward_edges = self.rescale_edges(reward_edges)
        total_reward = torch.sum(reward_nodes, dim=-2) + torch.sum(reward_edges, dim=-2) # [*, num_bins]
        return total_reward

    def Q(self, z, a, task, return_type='min', target=False, detach=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled Q-values.
            - `avg`: return the average of two randomly subsampled Q-values.
            - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {'min', 'avg', 'all'}

        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs

        # M: generate qvalues
        node_features = self._generate_node_features(z, a)
        value_nodes, value_edges = qnet(node_features)  # [num_q, *, num_nodes, num_bins], [num_q, *, num_edges, num_bins]
        value_edges = self.rescale_edges(value_edges)
        out = torch.sum(value_nodes, dim=-2) + torch.sum(value_edges, dim=-2) # [num_q, *, num_bins]

        if return_type == 'all':
            return out

        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q = math.two_hot_inv(out[qidx], self.cfg)
        if return_type == "min":
            return Q.min(0).values
        return Q.sum(0) / 2

