from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tensordict.nn import TensorDictParams
# import monotonicnetworks as lmn

from common import layers, math, init
from common.utils import benchmark_torch_function
from common import tdmpc_utils # tdmpc-1 utils

# TDMPC2
class WorldModel(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    Can be used for both single-task and multi-task experiments.
    """
    def __init__(self, cfg, is_student=False):
        super().__init__()
        self.cfg = cfg
        self.is_student = is_student
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

        # Assign params to modules
        # We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
        delattr(self._detach_Qs, "params")
        self._detach_Qs.__dict__["params"] = self._detach_Qs_params
        delattr(self._target_Qs, "params")
        self._target_Qs.__dict__["params"] = self._target_Qs_params

    def __repr__(self):
        repr = 'TD-MPC2 World Model\n'
        module_names = ['Encoder', 'Dynamics', 'Reward', 'Q-functions'] 
        modules = [self._encoder, self._dynamics, self._reward, self._Qs]
        if not self.is_student:  
            module_names.append('Policy')
            modules.append(self._pi)
        for i, m in enumerate(modules):
            repr += f"{module_names[i]}: {m}\n"
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
        self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

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
            if emb.shape[0] == x.shape[0]:
                emb = emb.unsqueeze(1).repeat(1, x.shape[1], 1)
            else:
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

    def reward(self, z, a, task, return_individual=False):
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

    def Q(self, z, a, task, return_type='min', target=False, detach=False, return_individual=False):
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


# TDMPC1
class TOLD(WorldModel):
    """
    Factored TD-MPC1 implicit world model architecture.
    Can be used for both single-task and multi-task experiments.
    """
    def __init__(self, cfg, is_student=False):

        nn.Module.__init__(self)
        self.cfg = cfg
        self.is_student = is_student

        if is_student:
            self._encoder = tdmpc_utils.mlp(cfg.latent_dim, cfg.enc_dim, cfg.latent_dim)
            self.num_agents = 1
            self.action_dim_agent = cfg.action_dim
            self.latent_dim_agent = cfg.latent_dim
        else:
            self._encoder = tdmpc_utils.enc(cfg)
            self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)

        self._dynamics = tdmpc_utils.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
        self._reward = tdmpc_utils.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, 1)
        self._Qs = layers.Ensemble([tdmpc_utils.q(cfg) for _ in range(cfg.num_q)])

        self.apply(tdmpc_utils.orthogonal_init)
        init.zero_([self._reward[-1].weight, self._Qs.params["5", "weight"]])
        init.zero_([self._reward[-1].bias, self._Qs.params["5", "bias"]])

        if not is_student:
            self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
            self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
            self.init() # target Q

    def encode(self, obs, task=None):
        return self._encoder(obs)


# Fac-TDMPC
class FacTOLD(WorldModel):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""
    def __init__(self, cfg, is_student=False):

        nn.Module.__init__(self)

        # MC: action/latent dimensions
        self.num_agents = cfg.num_agents # number of agents
        self.action_dim_agent = cfg.action_dim // self.num_agents
        self.latent_dim_agent = cfg.latent_dim_agent # cfg.latent_dim // self.num_agents
        self.cfg = cfg
        self.is_student = is_student

        # centralized modules
        if is_student:
            self._encoder = tdmpc_utils.mlp(cfg.latent_dim, cfg.enc_dim, self.latent_dim_agent*self.num_agents)
            # self._encoder = tdmpc_utils.mlp(cfg.obs_shape['state'][0], cfg.enc_dim, self.latent_dim_agent*self.num_agents)
        else:
            self._encoder = tdmpc_utils.enc(cfg)
            self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)

        # factored modules
        mlp_dim_agent = max(cfg.mlp_dim // self.num_agents, 50)
        self._dynamics = tdmpc_utils.FacMLP(self.num_agents, self.latent_dim_agent+self.action_dim_agent, 2*[mlp_dim_agent], self.latent_dim_agent) 
        self._reward = tdmpc_utils.FacMLP(self.num_agents, self.latent_dim_agent+self.action_dim_agent, 2*[mlp_dim_agent], 1)
        self._Qs = layers.Ensemble([tdmpc_utils.FacMLP(self.num_agents, self.latent_dim_agent+self.action_dim_agent, 2*[cfg.mlp_dim//self.num_agents], 1, is_q=True) for _ in range(cfg.num_q)])

        # MC: mixing network
        # === LMN ===
        # self._reward_mixer = lmn.MonotonicLayer(self.num_agents, 1)
        # self._value_mixer = lmn.MonotonicLayer(self.num_agents, 1)
        # lip_nn = nn.Sequential(
        #     lmn.LipschitzLinear(self.num_agents, 32, kind="one-inf"),
        #     lmn.GroupSort(2),
        #     lmn.LipschitzLinear(32, 1, kind="inf"),
        # )
        # self._value_mixer = lmn.MonotonicWrapper(lip_nn) # 2 layer

        # === VDN ===
        # self._reward_mixer = nn.AvgPool1d(self.num_agents)
        # self._value_mixer = nn.AvgPool1d(self.num_agents)

        # === QMIX ===
        # self._reward_mixer = layers.QMIXNet(self.num_agents, self.cfg.latent_dim) 
        # self._value_mixer = layers.QMIXNet(self.num_agents, self.cfg.latent_dim) 

        # init values
        self.apply(tdmpc_utils.orthogonal_init)
        init.zero_([self._reward.node_update[-1].weight, self._Qs.params["node_update", "2", "weight"]])
        init.zero_([self._reward.node_update[-1].bias, self._Qs.params["node_update", "2", "bias"]])

        if not is_student:
            self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
            self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
            self.init() # target Q
 
    def _generate_node_features(self, z, a):
        """
        Concat state and action for nodes
        """
        latent_node = torch.reshape(z, (*z.shape[:-1], self.num_agents, self.latent_dim_agent))
        action_node = torch.reshape(a, (*a.shape[:-1], self.num_agents, self.action_dim_agent))
        node_features = torch.cat([latent_node, action_node], dim=-1) # [num_step*num_traj, num_agents, node_dim]
        return node_features

    def encode(self, obs, task=None):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        return self._encoder(obs)

    def next(self, z, a, task):
        """
        Predicts the next latent state given the current latent state and action.
        Args:
            - z: [*, hidden_dim] * might be 1 or 2 dims
            - a: [*, action_dim] 
        """
        node_features = self._generate_node_features(z, a)
        x = self._dynamics(node_features) # [*, num_agents, hidden_dim_agent]
        x = torch.reshape(x, (*x.shape[:-2], -1)) # [*, hidden_dim_agent * num_agents]
        return x

    def reward(self, z, a, task, return_individual=False):
        """
        Predicts instantaneous (single-step) reward.
        """
        node_features = self._generate_node_features(z, a)
        reward_nodes = self._reward(node_features)  # [*, num_agents, num_bins], [*, num_edges, num_bins]
        if return_individual:
            return reward_nodes
        else:
            total_reward = torch.mean(reward_nodes, dim=-2)
            # total_reward = self._reward_mixer(reward_nodes.squeeze(-1)).unsqueeze(-1)
            return total_reward

    def Q(self, z, a, task, return_type='min', target=False, detach=False, return_individual=False): 
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled Q-values.
            - `avg`: return the average of two randomly subsampled Q-values.
            - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {'min', 'avg', 'all'}
        qnet = self._Qs

        # M: generate qvalues
        node_features = self._generate_node_features(z, a)
        value_nodes = qnet(node_features)  # [num_q, *, num_agents, num_bins], [num_q, *, num_edges, num_bins]
        if return_individual:
            out = value_nodes
        else:
            out = torch.mean(value_nodes, dim=-2)
            # out = self._value_mixer(value_nodes.squeeze(-1)).unsqueeze(-1)

        if return_type == 'all':
            return out

        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q = out[qidx] # Q = math.two_hot_inv(out[qidx], self.cfg) # M: factor models only support num_bins=0
        if return_type == "min":
            return Q.min(0).values
        return Q.sum(0) / 2

    def __repr__(self):
        repr = 'FacTOLD\n'
        modules = ['Encoder', 'Dynamics', 'Reward', 'Q-functions']
        for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._Qs]):
            repr += f"{modules[i]}: {m}\n"
        repr += "Learnable parameters: {:,}".format(self.total_params)
        return repr


# class FacActTOLD(FacTOLD):
#     """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""
#     def __init__(self, cfg):

#         nn.Module.__init__(self)

#         self.num_agents = cfg.num_agents # number of agents
#         self.action_dim_agent = cfg.action_dim // self.num_agents
#         self.latent_dim_agent = 50 # cfg.latent_dim
#         self.cfg = cfg

#         # modules
#         # Baseline: encode from z or obs
#         self._encoder = tdmpc_utils.mlp(cfg.latent_dim, cfg.enc_dim, self.latent_dim_agent*self.num_agents)
#         self._action_encoder = tdmpc_utils.mlp(cfg.action_dim, cfg.enc_dim, self.action_dim_agent*self.num_agents)
#         # self._action_inn = tdmpc_utils.InvertibleNN(cfg.action_dim)

#         self._dynamics = tdmpc_utils.FacMLP(self.num_agents, self.latent_dim_agent+self.action_dim_agent, 2*[cfg.mlp_dim//self.num_agents], self.latent_dim_agent) 
#         self._reward = tdmpc_utils.FacMLP(self.num_agents, self.latent_dim_agent+self.action_dim_agent, 2*[cfg.mlp_dim//self.num_agents], 1)
#         self._Qs = layers.Ensemble([tdmpc_utils.FacMLP(self.num_agents, self.latent_dim_agent+self.action_dim_agent, 2*[cfg.mlp_dim//self.num_agents], 1, is_q=True) for _ in range(cfg.num_q)])

#         # init values
#         self.apply(tdmpc_utils.orthogonal_init)
#         init.zero_([self._reward.node_update[-1].weight, self._Qs.params["node_update", "2", "weight"]])
#         init.zero_([self._reward.node_update[-1].bias, self._Qs.params["node_update", "2", "bias"]])

#     # Currently still plan in the original action space, but first map to some latent actions...
#     def _generate_node_features(self, z, a):
#         """
#         Concat state and action for nodes
#         """
#         a = self._action_encoder(a) # [..., action_dim_agent*num_agents]
#         latent_node = torch.reshape(z, (*z.shape[:-1], self.num_agents, self.latent_dim_agent))
#         action_node = torch.reshape(a, (*a.shape[:-1], self.num_agents, self.action_dim_agent))
#         node_features = torch.cat([latent_node, action_node], dim=-1) # [num_step*num_traj, num_agents, node_dim]
#         return node_features


# class FacWorldModel(WorldModel):
#     """
#     Factored TD-MPC2 implicit world model architecture.
#     Can be used for both single-task and multi-task experiments.
#     """

#     def __init__(self, cfg):

#         nn.Module.__init__(self)

#         # M: action/latent dimensions
#         self.num_agents = cfg.num_agents
#         self.action_dim_agent = cfg.action_dim // self.num_agents
#         self.latent_dim_agent = max(cfg.latent_dim // self.num_agents, 50)
#         self.latent_dim_agent = self.latent_dim_agent // cfg.simnorm_dim * cfg.simnorm_dim
#         self.cfg = cfg

#         # central modules
#         self._encoder = layers.mlp(cfg.latent_dim, cfg.enc_dim, self.latent_dim_agent*self.num_agents, act=layers.SimNorm(cfg))

#         # factored modules
#         mlp_dim_agent = max(cfg.mlp_dim // self.num_agents, 50)
#         self._dynamics = layers.FacMLP(self.num_agents, self.latent_dim_agent + self.action_dim_agent, 2*[mlp_dim_agent], self.latent_dim_agent, act=layers.SimNorm(cfg)) 
#         self._reward = layers.FacMLP(self.num_agents, self.latent_dim_agent + self.action_dim_agent, 2*[mlp_dim_agent], 1)
#         self._Qs = layers.Ensemble([layers.FacMLP(self.num_agents, self.latent_dim_agent + self.action_dim_agent, 2*[mlp_dim_agent], 1, dropout=cfg.dropout) for _ in range(cfg.num_q)])

#         # init
#         self.apply(init.weight_init)
#         init.zero_([self._reward.node_update[-1].weight, self._Qs.params["node_update", "2", "weight"]])
#         init.zero_([self._reward.node_update[-1].bias, self._Qs.params["node_update", "2", "bias"]])

#     def encode(self, obs, task=None):
#         """
#         Encodes an observation into its latent representation.
#         This implementation assumes a single state-based observation.
#         """
#         return self._encoder(obs)

#     def _generate_node_features(self, z, a):
#         """
#         Concat state and action for nodes
#         """
#         latent_node = torch.reshape(z, (*z.shape[:-1], self.num_agents, self.latent_dim_agent))
#         action_node = torch.reshape(a, (*a.shape[:-1], self.num_agents, self.action_dim_agent))
#         node_features = torch.cat([latent_node, action_node], dim=-1) # [num_step*num_traj, num_agents, node_dim]
#         return node_features

#     def next(self, z, a, task, return_individual=False):
#         """
#         Predicts the next latent state given the current latent state and action.
#         Args:
#             - z: [*, hidden_dim] * might be 1 or 2 dims
#             - a: [*, action_dim] 
#         """
#         node_features = self._generate_node_features(z, a)
#         x = self._dynamics(node_features) # [*, num_agents, hidden_dim_node]
#         x = torch.reshape(x, (*x.shape[:-2], -1)) # [*, hidden_dim_node * num_agents]
#         return x

#     def reward(self, z, a, task, return_individual=False):
#         """
#         Predicts instantaneous (single-step) reward.
#         """
#         node_features = self._generate_node_features(z, a)
#         reward_nodes = self._reward(node_features)  # [*, num_agents, num_bins], [*, num_edges, num_bins]

#         if return_individual:
#             return reward_nodes
#         else:
#             total_reward = torch.mean(reward_nodes, dim=-2)
#             return total_reward

#     def Q(self, z, a, task, return_type='min', target=False, detach=False, return_individual=False):
#         """
#         Predict state-action value.
#         `return_type` can be one of [`min`, `avg`, `all`]:
#             - `min`: return the minimum of two randomly subsampled Q-values.
#             - `avg`: return the average of two randomly subsampled Q-values.
#             - `all`: return all Q-values.
#         `target` specifies whether to use the target Q-networks or not.
#         """
#         assert return_type in {'min', 'avg', 'all'}
#         qnet = self._Qs

#         # M: generate qvalues
#         node_features = self._generate_node_features(z, a)
#         value_nodes = qnet(node_features)  # [num_q, *, num_agents, num_bins], [num_q, *, num_edges, num_bins]
        
#         # VDN style 
#         if return_individual:
#             out = value_nodes
#         else:
#             out = torch.mean(value_nodes, dim=-2)

#         if return_type == 'all':
#             return out

#         qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
#         # Q = math.two_hot_inv(out[qidx], self.cfg)
#         Q = out[qidx] # M: factor models only support num_bins=0
#         if return_type == "min":
#             return Q.min(0).values
#         return Q.sum(0) / 2

#     def __repr__(self):
#         repr = 'FacWorldModel\n'
#         modules = ['Encoder', 'Dynamics', 'Reward', 'Q-functions']
#         for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._Qs]):
#             repr += f"{modules[i]}: {m}\n"
#         repr += "Learnable parameters: {:,}".format(self.total_params)
#         return repr

