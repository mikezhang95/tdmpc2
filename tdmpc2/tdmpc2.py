import torch
import torch.nn.functional as F
from tensordict import TensorDict
import numpy as np
import time

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.world_model import FacWorldModel
from common.world_model import TOLD, FacTOLD


class TDMPC2(torch.nn.Module):
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device= torch.device(cfg.device)
        if hasattr(cfg, 'model_type') and cfg.model_type == 'fac_world_model':
            self.model = WorldModel(cfg).to(self.device) # TDMPC2
        else:
            self.model = TOLD(cfg).to(self.device) # TDMPC
        self.optim = torch.optim.Adam([
            {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
            {'params': self.model._dynamics.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
            ], lr=self.cfg.lr, capturable=True)

        if self.cfg.fac_model:
            if hasattr(cfg, 'model_type') and cfg.model_type == 'fac_world_model':
                self.fac_model = FacWorldModel(cfg).to(self.device)
            else:
                self.fac_model = FacTOLD(cfg).to(self.device)
            self.fac_optim = torch.optim.Adam([
                {'params': self.fac_model._encoder.parameters()}, 
                {'params': self.fac_model._dynamics.parameters()},
                {'params': self.fac_model._reward.parameters()},
                {'params': self.fac_model._Qs.parameters()},
                {'params': self.fac_model._reward_mixer.parameters() if hasattr(self.fac_model, '_reward_mixer') else []},
                {'params': self.fac_model._value_mixer.parameters() if hasattr(self.fac_model, '_value_mixer') else []},
                ], lr=self.cfg.fac_lr, capturable=True)
            print(self.fac_model)

        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
        self.discount = torch.tensor(
            [self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device=self.device
        ) if self.cfg.multitask else self._get_discount(cfg.episode_length)
        self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.num_envs, self.cfg.horizon, self.cfg.action_dim, device=self.device))
        if cfg.compile:
            print('Compiling update function with torch.compile...')
            self._update = torch.compile(self._update, mode="reduce-overhead")

    @property
    def plan(self):
        _plan_val = getattr(self, "_plan_val", None)
        if _plan_val is not None:
            return _plan_val
        if self.cfg.compile:
            plan = torch.compile(self._plan, mode="reduce-overhead")
        else:
            plan = self._plan
        self._plan_val = plan
        return self._plan_val

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
            episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
            float: Discount factor for the task.
        """
        frac = episode_length/self.cfg.discount_denom
        return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

    def save(self, fp):
        """
        Save state dict of the agent to filepath.

        Args:
            fp (str): Filepath to save state dict to.
        """
        state_dict = {"model": self.model.state_dict()}
        if self.cfg.fac_model:
            state_dict["fac_model"] = self.fac_model.state_dict()
        torch.save(state_dict, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.

        Args:
            fp (str or dict): Filepath or state dict to load.
        """
        state_dict = fp if isinstance(fp, dict) else torch.load(fp, map_location=self.device)
        self.model.load_state_dict(state_dict["model"])
        if self.cfg.fac_model and "fac_model" in state_dict:
            self.fac_model.load_state_dict(state_dict["fac_model"])

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        Select an action by planning in the latent space of the world model.

        Args:
            obs (torch.Tensor): Observation from the environment.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (int): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: Action to take in the environment.
        """
        obs = obs.to(self.device, non_blocking=True)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        if self.cfg.mpc:
            a = self.plan(obs, t0=t0, eval_mode=eval_mode, task=task)
        else:
            z = self.model.encode(obs, task)
            a = self.model.pi(z, task)[int(not eval_mode)][0]
        return a.cpu()

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(self.cfg.horizon - 1):
            reward = math.two_hot_inv(self.model.reward(z, actions[:, t], task), self.cfg)
            z = self.model.next(z, actions[:, t], task)
            G = G + discount * reward
            discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
            discount = discount * discount_update
        return G + discount * self.model.Q(z, actions[:, -1], task, return_type='avg')
        # return G + discount * self.model.Q(z, self.model.pi(z, task)[1], task, return_type='avg')

    @torch.no_grad()
    def _estimate_individual_value(self, z, actions, task, return_individual=True):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(self.cfg.horizon - 1): 
            # reward = math.two_hot_inv(self.fac_model.reward(z, actions[:, t], task, return_individual=return_individual), self.cfg)
            reward = self.fac_model.reward(z, actions[:, t], task, return_individual=return_individual) # M: only support num_bins=0
            z = self.fac_model.next(z, actions[:, t], task, return_individual=True)
            G = G + discount * reward
            discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
            discount = discount * discount_update
        return G + discount * self.fac_model.Q(z, actions[:, -1], task, return_type='avg', return_individual=return_individual)
        # return G + discount * self.fac_model.Q(z, self.model.pi(z, task)[1], task, return_type='avg', return_individual=True)


    # @benchmark_torch_function
    @torch.no_grad()
    def _plan(self, obs, t0=False, eval_mode=False, task=None):
        """
        Plan a sequence of actions using the learned world model.

        Args:
            task (Torch.Tensor): Task index (only used for multi-task experiments).
z (torch.Tensor): Latent state from which to plan.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
                
        Returns:
            torch.Tensor: Action to take in the environment.
        """
        # Sample policy trajectories
        z = self.model.encode(obs, task)
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.num_envs, self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon-1):
                pi_actions[:,t] = self.model.pi(_z, task)[1]
                _z = self.model.next(_z, pi_actions[:,t], task)
            pi_actions[:,-1] = self.model.pi(_z, task)[1]

        # Initialize state and parameters
        z = z.unsqueeze(1).repeat(1, self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.num_envs, self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std*torch.ones(self.cfg.num_envs, self.cfg.horizon, self.cfg.action_dim, device=self.device)
        if not t0:
            mean[:, :-1] = self._prev_mean[:, 1:]
        actions = torch.empty(self.cfg.num_envs, self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :, :self.cfg.num_pi_trajs] = pi_actions
    
        # Iterate MPPI
        for iter in range(self.cfg.iterations):

            # Sample actions
            r = torch.randn(self.cfg.num_envs, self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
            actions_sample = mean.unsqueeze(2) + std.unsqueeze(2) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, :, self.cfg.num_pi_trajs:] = actions_sample
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # TODO: this 2 branches can be combined
            # M: factorized version
            if self.cfg.fac_model: # decentralized version

                # Compute elite actions
                fac_z = self.fac_model.encode(z, task)
                value = self._estimate_individual_value(fac_z, actions, task).nan_to_num(0).squeeze(-1) # [E, N, AN]
                elite_idxs = torch.topk(value, self.cfg.num_elites, dim=1).indices # [E, EL, NA]
                elite_value = torch.gather(value, 1, elite_idxs) # [E, EL, NA]
                reshaped_actions = actions.view(self.cfg.num_envs, self.cfg.horizon, self.cfg.num_samples, self.cfg.num_agents, self.fac_model.action_dim_agent) # [E, H, N, NA, AN] 
                elite_actions = torch.gather(reshaped_actions, 2, elite_idxs.unsqueeze(1).unsqueeze(4).expand(-1, self.cfg.horizon, -1, -1, self.fac_model.action_dim_agent)) # [E, H, EL, NA, AN]

                # Update parameters
                max_value = elite_value.max(1).values # [E, NA]
                score = torch.exp(self.cfg.temperature*(elite_value - max_value.unsqueeze(1))) # [E, EL, NA]
                score = (score / score.sum(1, keepdim=True)) # [E, EL, NA]
                extend_score = score.unsqueeze(1).unsqueeze(4) # [E, 1, EL, NA, 1]
                mean = (extend_score * elite_actions).sum(2) / (extend_score.sum(2) + 1e-9) # [E, H, NA, AN]
                std = ((extend_score * (elite_actions - mean.unsqueeze(2)) ** 2).sum(2) / (extend_score.sum(2) + 1e-9)).sqrt() # [E, H, NA, AN]
                mean = mean.view(self.cfg.num_envs, self.cfg.horizon, -1) # [E, H, A]
                std = std.view(self.cfg.num_envs, self.cfg.horizon, -1) # [E, H, A]
                std = std.clamp(self.cfg.min_std, self.cfg.max_std)

            # M: centralized version
            else:  
                # Compute elite actions
                value = self._estimate_value(z, actions, task).nan_to_num(0) # [E, N, 1] 
                elite_idxs = torch.topk(value.squeeze(2), self.cfg.num_elites, dim=1).indices # [E, EL]
                elite_value = torch.gather(value, 1, elite_idxs.unsqueeze(2)) # [E, EL, 1]
                elite_actions = torch.gather(actions, 2, elite_idxs.unsqueeze(1).unsqueeze(3).expand(-1, self.cfg.horizon, -1, self.cfg.action_dim)) # [E, H, EL, A]
                # Update parameters
                max_value = elite_value.max(1).values # [E, 1]
                score = torch.exp(self.cfg.temperature*(elite_value - max_value.unsqueeze(1))) # [E, EL, 1]
                score = (score / score.sum(1, keepdim=True)) # [E, EL, 1]
                mean = (score.unsqueeze(1) * elite_actions).sum(2) / (score.sum(1, keepdim=True) + 1e-9) # [E, H, A]
                std = ((score.unsqueeze(1) * (elite_actions - mean.unsqueeze(2)) ** 2).sum(2) / (score.sum(1, keepdim=True) + 1e-9)).sqrt() # [E, H, A]
                std = std.clamp(self.cfg.min_std, self.cfg.max_std)

            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # Select action
        # factorized version
        if self.cfg.fac_model: 
            rand_idx = torch.stack([math.gumbel_softmax_sample(score[..., i], dim=1) for i in range(self.cfg.num_agents)], dim=-1) # [E, NA] gumbel_softmax_sample is compatible with cuda graphs
            actions = torch.gather(elite_actions, 2, rand_idx.unsqueeze(1).unsqueeze(2).unsqueeze(4).expand(-1, self.cfg.horizon, -1, -1, self.fac_model.action_dim_agent)).squeeze(2) # [E, H, NA, AN]
            actions = actions.view(self.cfg.num_envs, self.cfg.horizon, -1) # [E, H, A]
        # centralized version
        else: 
            rand_idx = math.gumbel_softmax_sample(score.squeeze(2), dim=1)  # [E,] gumbel_softmax_sample is compatible with cuda graphs
            actions = elite_actions[torch.arange(self.cfg.num_envs), :, rand_idx] # [E, H, A]
        action, std = actions[:, 0], std[:, 0] # MPC run first step
        if not eval_mode:
            action = action + std * torch.randn(self.cfg.action_dim, device=std.device)
        self._prev_mean.copy_(mean)
        return action.clamp(-1, 1)
            
            
    def update_pi(self, zs, task):
        """
        Update policy using a sequence of latent states.

        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """
        _, pis, log_pis, _ = self.model.pi(zs, task)
        qs = self.model.Q(zs, pis, task, return_type='avg', detach=True)
        self.scale.update(qs[0]) # normalize qs, speedup training \pi
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        if not self.cfg.fac_model:
            self.pi_optim.step()
        self.pi_optim.zero_grad(set_to_none=True)

        return pi_loss.detach(), pi_grad_norm

    @torch.no_grad()
    def _td_target(self, next_z, reward, task):
        """
        Compute the TD-target from a reward and the observation at the following time step.

        Args:
            next_z (torch.Tensor): Latent state at the following time step.
            reward (torch.Tensor): Reward at the current time step.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: TD-target.
        """
        pi = self.model.pi(next_z, task)[1]
        discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        return reward + discount * self.model.Q(next_z, pi, task, return_type='min', target=True)

    # @benchmark_torch_function
    def _update(self, obs, action, reward, task=None):
        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, task)

        # Prepare for update
        self.model.train()

        # Latent rollout
        zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
            z = self.model.next(z, _action, task)
            consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
            zs[t+1] = z

        # Predictions
        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type='all')
        reward_preds = self.model.reward(_zs, action, task)
        
        # Compute losses
        reward_loss, value_loss = 0, 0
        for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
            reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
            for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
                value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

        consistency_loss = consistency_loss / self.cfg.horizon
        reward_loss = reward_loss / self.cfg.horizon
        value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
        total_loss = (
            self.cfg.consistency_coef * consistency_loss +
            self.cfg.reward_coef * reward_loss +
            self.cfg.value_coef * value_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        if not self.cfg.fac_model:
            self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        # Update policy
        pi_loss, pi_grad_norm = self.update_pi(zs.detach(), task)

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        return_dict =  TensorDict({
            "consistency_loss": consistency_loss,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
            "pi_loss": pi_loss,
            "total_loss": total_loss,
            "grad_norm": grad_norm,
            "pi_grad_norm": pi_grad_norm,
            "pi_scale": self.scale.value,
        }).detach().mean()

        # M: Training factored models
        if self.cfg.fac_model:
            
            # Prepare for update
            self.fac_model.train()

            # sampled parameters
            num_noises, std_noises = self.cfg.num_noises, self.cfg.std_noises

            # sampled actions
            r = torch.randn(self.cfg.horizon, self.cfg.batch_size, num_noises, self.cfg.action_dim, device=action.device)
            action_sample = action.unsqueeze(2) + std_noises * r # [H, BS, NS, A]
            fac_z = self.fac_model.encode(zs[0].clone().detach()).unsqueeze(1).repeat(1, num_noises, 1) # [BS, NS, FS]
            # fac_z = self.fac_model.encode(obs[0].clone().detach()).unsqueeze(1).repeat(1, num_noises, 1) # [BS, NS, FS] # encode from obs
            fac_zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, num_noises, self.fac_model.latent_dim_agent * self.fac_model.num_agents, device=self.device) # [H, BS, NS, FS]
            fac_zs[0] = fac_z
            fac_zs_sp = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, num_noises, self.fac_model.latent_dim_agent * self.fac_model.num_agents, device=self.device) # [H, BS, NS, FS]
            fac_zs_sp[0] = fac_z.detach() # self_predictive
            # sampled zs
            global_z = zs[0].clone().detach().unsqueeze(1).repeat(1, num_noises, 1) # [BS, NS, S]
            global_zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, num_noises, self.cfg.latent_dim, device=self.device) # [H, BS, NS, S]
            global_zs[0] = global_z 
            # rollout in latent space
            fac_consistency_loss = 0.0
            for t, _action in enumerate(action_sample.unbind(0)):
                fac_z = self.fac_model.next(fac_z, _action, task, return_individual=True)
                fac_zs[t+1] = fac_z
                global_z = self.model.next(global_z, _action, task).detach()
                global_zs[t+1] = global_z
                fac_z_sp = self.fac_model.encode(global_z).detach()
                fac_zs_sp[t+1] = fac_z_sp
                fac_consistency_loss = fac_consistency_loss + F.mse_loss(fac_z, fac_z_sp) * self.cfg.fac_rho**t

            _fac_zs_sample = fac_zs[:-1]
            _fac_zs_sample = _fac_zs_sample.view(self.cfg.horizon, -1, self.fac_model.latent_dim_agent * self.fac_model.num_agents) # [H, BS*NS, FS]
            _zs_sample = global_zs[:-1]
            _zs_sample = _zs_sample.view(self.cfg.horizon, -1, self.cfg.latent_dim) # [H, BS*NS, S]
            action_sample = action_sample.view(self.cfg.horizon, -1, self.cfg.action_dim) # [H, BS*NS, S]


            # MC: monotonic on which functions
            # ====== c1: monotonic R/Q ====== # 
            fac_rewards = self.fac_model.reward(_fac_zs_sample, action_sample, task) # [H, BS*NS, 1]
            fac_qs = self.fac_model.Q(_fac_zs_sample, action_sample, task, return_type='all') # [Q, HS, BS*NS, 1]
            target_rewards = self.model.reward(_zs_sample, action_sample, task).detach() # [H, BS*NS, 1]
            target_qs = self.model.Q(_zs_sample, action_sample, task, return_type='avg', detach=True).detach() # [H, BS*NS, 1]
            
            # Compute losses
            fac_reward_loss, fac_value_loss = 0, 0
            for t, (pred_rew_unbind, rew_unbind, pred_qs_unbind, qs_unbind) in enumerate(zip(fac_rewards.unbind(0), target_rewards.unbind(0), fac_qs.unbind(1), target_qs.unbind(0))):
                # Baseline: mse loss or kl loss
                # fac_reward_loss = fac_reward_loss + math.soft_ce(pred_rew_unbind, rew_unbind, self.cfg).mean() * self.cfg.fac_rho**t
                pred_rew_unbind = pred_rew_unbind.view(self.cfg.batch_size, num_noises)
                rew_unbind = math.two_hot_inv(rew_unbind, self.cfg)
                rew_unbind = rew_unbind.view(self.cfg.batch_size, num_noises)
                fac_reward_loss = fac_reward_loss + math.softmax_distillation_loss(pred_rew_unbind, rew_unbind, self.cfg.temperature).mean() * self.cfg.fac_rho**t
                for _, pred_qs_unbind_unbind in enumerate(pred_qs_unbind.unbind(0)):
                    # fac_value_loss = fac_value_loss + math.soft_ce(pred_qs_unbind_unbind, qs_unbind, self.cfg).mean() * self.cfg.fac_rho**t
                    pred_qs_unbind_unbind = pred_qs_unbind_unbind.view(self.cfg.batch_size, num_noises)
                    qs_unbind = qs_unbind.view(self.cfg.batch_size, num_noises)
                    fac_value_loss = fac_value_loss + math.softmax_distillation_loss(pred_qs_unbind_unbind, qs_unbind, self.cfg.temperature).mean() * self.cfg.fac_rho**t
            # ============ # 

            # # ==== c2: monotonic Return ==== # 
            # fac_rewards = self.fac_model.reward(_fac_zs_sample, action_sample, task, return_individual=True)
            # fac_qs = self.fac_model.Q(_fac_zs_sample, action_sample, task, return_type='all', return_individual=True)
            # target_rewards = self.model.reward(_zs_sample, action_sample, task).detach()
            # target_qs = self.model.Q(_zs_sample, action_sample, task, return_type='avg', detach=True).detach() 
            
            # fac_reward_loss, fac_value_loss = 0, 0
            # fac_return, global_return = 0.0, 0.0
            # discount = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
            # for t, (pred_rew_unbind, rew_unbind, pred_qs_unbind, qs_unbind) in enumerate(zip(fac_rewards.unbind(0), target_rewards.unbind(0), fac_qs.unbind(1), target_qs.unbind(0))):
            #     tmp_global_return = global_return + qs_unbind * discount**t
            #     for _, pred_qs_unbind_unbind in enumerate(pred_qs_unbind.unbind(0)):
            #         if t < self.cfg.horizon - 1: break # new2
            #         tmp_fac_return = fac_return + pred_qs_unbind_unbind * discount**t
            #         tmp_fac_return_mix = self.fac_model._value_mixer(tmp_fac_return.squeeze(-1) ) #, _zs_sample[0])
            #         fac_value_loss = fac_value_loss + math.soft_ce(tmp_fac_return_mix, tmp_global_return, self.cfg).mean()  * self.cfg.fac_rho**t
            #     pred_rew_unbind_mix = self.fac_model._reward_mixer(pred_rew_unbind.squeeze(-1) ) #, _zs_sample[0])
            #     fac_reward_loss = fac_reward_loss + math.soft_ce(pred_rew_unbind_mix, rew_unbind, self.cfg).mean() * self.cfg.fac_rho**t
            #     fac_return += pred_rew_unbind * discount**t
            #     global_return += rew_unbind * discount**t
            # # ======== # 

            fac_consistency_loss = fac_consistency_loss / self.cfg.horizon
            fac_reward_loss = fac_reward_loss / self.cfg.horizon
            fac_value_loss = fac_value_loss / (self.cfg.horizon * self.cfg.num_q)

            fac_total_loss = (
                self.cfg.fac_consistency_coef * fac_consistency_loss +
                self.cfg.fac_reward_coef * fac_reward_loss +
                self.cfg.fac_value_coef * fac_value_loss
            )

            fac_total_loss.backward()
            fac_grad_norm = torch.nn.utils.clip_grad_norm_(self.fac_model.parameters(), self.cfg.grad_clip_norm)
            self.fac_optim.step()
            self.fac_optim.zero_grad(set_to_none=True)

            return_dict["fac_consistency_loss"] = fac_consistency_loss.detach().mean()
            return_dict["fac_reward_loss"] = fac_reward_loss.detach().mean()
            return_dict["fac_value_loss"] = fac_value_loss.detach().mean()
            return_dict["fac_grad_norm"] = fac_grad_norm.detach().mean()
            self.fac_model.eval()

        return return_dict

    def update(self, buffer):
        """
        Main update function. Corresponds to one iteration of model learning.

        Args:
            buffer (common.buffer.Buffer): Replay buffer.

        Returns:
            dict: Dictionary of training statistics.
        """
         
        obs, action, reward, task = buffer.sample()
        kwargs = {}
        if task is not None:
            kwargs["task"] = task
        torch.compiler.cudagraph_mark_step_begin()
        return self._update(obs, action, reward, **kwargs)
