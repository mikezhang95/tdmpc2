import torch
import torch.nn.functional as F
from tensordict import TensorDict
import numpy as np
import time

from common import math
from common.scale import RunningScale
from common.layers import api_model_conversion
from common.world_model import WorldModel
from common.world_model import TOLD, FacTOLD
from common.world_model import TAPTOLD, LaLQR
# from common.world_model import FacWorldModel


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
        if cfg.model_name == 'tdmpc':
            self.model = TOLD(cfg).to(self.device) # TDMPC
        elif cfg.model_name == 'fac_tdmpc':
            self.model = FacTOLD(cfg).to(self.device) # Fac-TDMPC
        elif cfg.model_name == 'tap_tdmpc':
            self.model = TAPTOLD(cfg).to(self.device) # Fac-TDMPC
        elif cfg.model_name == 'lalqr':
            self.model = LaLQR(cfg).to(self.device) # Fac-TDMPC
        else:
            self.model = WorldModel(cfg).to(self.device) # TDMPC2
        self.optim = torch.optim.Adam([
            {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
            {'params': self.model._dynamics.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._reward_mixer.parameters() if hasattr(self.model, '_reward_mixer') else []},
            {'params': self.model._value_mixer.parameters() if hasattr(self.model, '_value_mixer') else []},
            {'params': self.model._action_encoder.parameters() if hasattr(self.model, '_action_encoder') else []},
            {'params': self.model._action_decoder.parameters() if hasattr(self.model, '_action_decoder') else []},
            {'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
            ], lr=self.cfg.lr, capturable=True)

        if hasattr(cfg, 'student_cfg'):
            self.student_cfg = cfg.student_cfg
            cfg.student_cfg.action_dim = cfg.action_dim
            cfg.student_cfg.action_dims = cfg.action_dims
            cfg.student_cfg.num_envs = cfg.num_envs
            if cfg.student_cfg.model_name == 'tdmpc':
                self.student_model = TOLD(cfg.student_cfg, is_student=True).to(self.device) # TDMPC
            elif cfg.student_cfg.model_name == 'fac_tdmpc':
                self.student_model = FacTOLD(cfg.student_cfg, is_student=True).to(self.device) # Fac-TDMPC
            elif cfg.student_cfg.model_name == 'tap_tdmpc':
                self.student_model = TAPTOLD(cfg.student_cfg, is_student=True).to(self.device) # TAP-TDMPC
            elif cfg.student_cfg.model_name == 'lalqr':
                self.student_model = LaLQR(cfg.student_cfg, is_student=True).to(self.device) # TAP-TDMPC
            else:
                raise NotImplementedError
            self.student_optim = torch.optim.Adam([
                {'params': self.student_model._encoder.parameters()}, 
                {'params': self.student_model._dynamics.parameters()},
                {'params': self.student_model._reward.parameters()},
                {'params': self.student_model._Qs.parameters()},
                {'params': self.student_model._reward_mixer.parameters() if hasattr(self.student_model, '_reward_mixer') else []},
                {'params': self.student_model._value_mixer.parameters() if hasattr(self.student_model, '_value_mixer') else []},
                {'params': self.student_model._action_encoder.parameters() if hasattr(self.student_model, '_action_encoder') else []},
                {'params': self.student_model._action_decoder.parameters() if hasattr(self.student_model, '_action_decoder') else []},
                {'params': self.student_model._fc_mu.parameters() if hasattr(self.student_model, '_fc_mu') else []},
                {'params': self.student_model._fc_logvar.parameters() if hasattr(self.student_model, '_fc_logvar') else []},
                ], lr=self.student_cfg.lr, capturable=True)
        else: 
            self.student_model = None

        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
        self.discount = torch.tensor(
            [self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device=self.device
        ) if self.cfg.multitask else self._get_discount(cfg.episode_length)
        if cfg.compile:
            print('Compiling update function with torch.compile...')
            self._update = torch.compile(self._update, mode="reduce-overhead")
        if hasattr(self.cfg, 'student_cfg'):
            cfg = self.cfg.student_cfg
        else: 
            cfg = self.cfg
        if 'tap' in cfg.model_name:
            self._prev_mean = torch.nn.Buffer(torch.zeros(cfg.num_envs, cfg.horizon, cfg.latent_action_dim, device=self.device))
        else:
            self._prev_mean = torch.nn.Buffer(torch.zeros(cfg.num_envs, cfg.horizon, cfg.action_dim, device=self.device))

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
        if self.student_model:
            state_dict["student_model"] = self.student_model.state_dict()
        torch.save(state_dict, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.

        Args:
            fp (str or dict): Filepath or state dict to load.
        """
        state_dict = fp if isinstance(fp, dict) else torch.load(fp, map_location=self.device)
        if "dmcontrol" in self.cfg.checkpoint: # compatible with official checkpoints
            model_state_dict = state_dict["model"] if "model" in state_dict else state_dict
            model_state_dict = api_model_conversion(self.model.state_dict(), model_state_dict)
            self.model.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(state_dict["model"], strict=False)
        if self.student_model:
            if "student_model" in state_dict:
                self.student_model.load_state_dict(state_dict["student_model"])
            elif "fac_model" in state_dict: # compatible with previous checkpoints
                self.student_model.load_state_dict(state_dict["fac_model"], strict=False)

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
    def _estimate_value(self, z, actions, task, model, cfg, return_individual=False):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(cfg.horizon - 1): 
            reward = math.two_hot_inv(model.reward(z, actions[:, t], task, return_individual=return_individual), cfg)
            # reward = self.student_model.reward(z, actions[:, t], task, return_individual=return_individual) # M: only support num_bins=0
            z = model.next(z, actions[:, t], task)
            G = G + discount * reward
            discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
            discount = discount * discount_update
        return G + discount * model.Q(z, actions[:, -1], task, return_type='avg', return_individual=return_individual)
        # return G + discount * self.student_model.Q(z, self.model.pi(z, task)[1], task, return_type='avg', return_individual=True)


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
        z = self.model.encode(obs, task)
        raw_z = z.clone()
        # Define preliminaries
        if self.student_model and eval_mode: 
            z = self.student_model.encode(z, task)
            # z = self.student_model.encode(obs, task)
            cfg, model = self.student_cfg, self.student_model
        else:
            cfg, model = self.cfg, self.model
        
        # LQR
        if 'lqr'in cfg.model_name:
            action = model.act(z, refresh=t0)
            return action.clamp(-1, 1)

        # Sample policy trajectories
        if cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.num_envs, cfg.horizon, cfg.num_pi_trajs, cfg.action_dim, device=self.device)
            _z = raw_z.unsqueeze(1).repeat(1, cfg.num_pi_trajs, 1)
            for t in range(cfg.horizon-1):
                pi_actions[:,t] = self.model.pi(_z, task)[1]
                _z = self.model.next(_z, pi_actions[:,t], task)
            pi_actions[:,-1] = self.model.pi(_z, task)[1]

        # Latent actions
        if 'tap' in cfg.model_name:
            pi_actions = pi_actions.transpose(1, 2).view(-1, cfg.horizon, cfg.action_dim) # [B*NPI, H, A]
            _, _, _, pi_actions = model.forward(z.repeat(cfg.num_pi_trajs, 1), pi_actions) # [B*NPI, H, LA]
            pi_actions = pi_actions.view(self.cfg.num_envs, cfg.num_pi_trajs, cfg.horizon, cfg.latent_action_dim).transpose(1, 2) # [B, H, NPI, LA]
            action_dim = cfg.latent_action_dim
        else:
            action_dim = cfg.action_dim
        num_agents = cfg.num_agents if hasattr(cfg, 'num_agents') else 1
        action_dim_agent = action_dim // num_agents

        # Initialize state and parameters
        z = z.unsqueeze(1).repeat(1, cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.num_envs, cfg.horizon, action_dim, device=self.device)
        std = cfg.max_std*torch.ones(self.cfg.num_envs, cfg.horizon, action_dim, device=self.device)
        if not t0:
            mean[:, :-1] = self._prev_mean[:, 1:]
        actions = torch.empty(self.cfg.num_envs, cfg.horizon, cfg.num_samples, action_dim, device=self.device)
        if cfg.num_pi_trajs > 0:
            actions[:, :, :cfg.num_pi_trajs] = pi_actions

        # Iterate MPPI
        for iter in range(cfg.iterations):

            # Sample actions
            r = torch.randn(self.cfg.num_envs, cfg.horizon, cfg.num_samples-cfg.num_pi_trajs, action_dim, device=std.device)
            actions_sample = mean.unsqueeze(2) + std.unsqueeze(2) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, :, cfg.num_pi_trajs:] = actions_sample
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # Compute actions' values
            return_individual = True if 'fac' in cfg.model_name else False
            value = self._estimate_value(z, actions, task, model, cfg, return_individual=return_individual).nan_to_num(0) # [E, N, NA]
            value = value.squeeze(-1) if len(value.shape)==4 else value

            # Compute elite actions
            elite_idxs = torch.topk(value, cfg.num_elites, dim=1).indices # [E, EL, NA]
            elite_value = torch.gather(value, 1, elite_idxs) # [E, EL, NA]
            reshaped_actions = actions.view(self.cfg.num_envs, cfg.horizon, cfg.num_samples, num_agents, action_dim_agent) # [E, H, N, NA, AN] 
            elite_actions = torch.gather(reshaped_actions, 2, elite_idxs.unsqueeze(1).unsqueeze(4).expand(-1, cfg.horizon, -1, -1, action_dim_agent)) # [E, H, EL, NA, AN]

            # Update parameters
            max_value = elite_value.max(1).values # [E, NA]
            score = torch.exp(cfg.temperature*(elite_value - max_value.unsqueeze(1))) # [E, EL, NA]
            score = (score / score.sum(1, keepdim=True)) # [E, EL, NA]
            extend_score = score.unsqueeze(1).unsqueeze(4) # [E, 1, EL, NA, 1]
            mean = (extend_score * elite_actions).sum(2) / (extend_score.sum(2) + 1e-9) # [E, H, NA, AN]
            std = ((extend_score * (elite_actions - mean.unsqueeze(2)) ** 2).sum(2) / (extend_score.sum(2) + 1e-9)).sqrt() # [E, H, NA, AN]
            mean = mean.view(self.cfg.num_envs, cfg.horizon, -1) # [E, H, A]
            std = std.view(self.cfg.num_envs, cfg.horizon, -1) # [E, H, A]
            std = std.clamp(cfg.min_std, cfg.max_std) # [E, H, A]

            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        self._prev_mean.copy_(mean)

        # Select action
        rand_idx = torch.stack([math.gumbel_softmax_sample(score[..., i], dim=1) for i in range(num_agents)], dim=-1) # [E, NA] gumbel_softmax_sample is compatible with cuda graphs
        actions = torch.gather(elite_actions, 2, rand_idx.unsqueeze(1).unsqueeze(2).unsqueeze(4).expand(-1, cfg.horizon, -1, -1, action_dim_agent)).squeeze(2) # [E, H, NA, AN]
        actions = actions.view(self.cfg.num_envs, cfg.horizon, -1) # [E, H, A]
        action, std = actions[:, 0], std[:, 0] # MPC run first step
        if not eval_mode:
            action = action + std * torch.randn(cfg.action_dim, device=std.device)
        if 'tap' in cfg.model_name:
            actions = model.decode_action(z[:, 0, :], actions) 
            action = actions[:, 0] # TODO: do not support explorations yet
        return action.clamp(-1, 1)
            
            
    def update_pi(self, zs, actions, task):
        """
        Update policy using a sequence of latent states.

        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """
        mus, pis, log_pis, log_stds = self.model.pi(zs, task)
        qs = self.model.Q(zs, pis, task, return_type='avg', detach=True)
        self.scale.update(qs[0]) # normalize qs, speedup training \pi
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()

        # M: BC loss
        if hasattr(self.cfg, 'bc_coef') and self.cfg.bc_coef > 0.0:
            mus = mus[:-1].view(-1, actions.shape[-1])
            log_stds = log_stds[:-1].view(-1, actions.shape[-1])
            actions = actions.view(-1, actions.shape[-1])
            # bc_loss = math.bc_loss(mus, log_stds, actions) # -log \pi(a'|s)
            bc_loss = ((mus - actions.detach())**2).sum(-1).mean() #  (mu(s) - a')^2
            pi_loss += bc_loss.mean() * self.cfg.bc_coef

        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
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
            if hasattr(self.cfg, 'loss_type') and self.cfg.loss_type == 'kl':
                rew_pred_unbind = rew_pred_unbind.view(-1, self.cfg.kl_batch)
                rew_unbind = rew_unbind.view(-1, self.cfg.kl_batch)
                reward_loss = reward_loss + math.softmax_distillation_loss(rew_pred_unbind, rew_unbind, self.cfg.kl_temp).mean() * self.cfg.rho**t
            else:
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
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        # Update policy
        pi_loss, pi_grad_norm = self.update_pi(zs.detach(), action, task)

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
        if self.student_model:
            
            # Prepare for update
            self.student_model.train()
            zs = zs.detach()
            action = action.detach()

            # sampled parameters
            num_noises, std_noises = self.student_cfg.num_noises, self.student_cfg.std_noises

            # sampled actions
            r = torch.randn(self.cfg.horizon, self.cfg.batch_size, num_noises, self.student_cfg.action_dim, device=action.device)
            action_sample = action.unsqueeze(2) + std_noises * r # [H, BS, NS, A]
            student_action_sample = action_sample.clone()
            # sampled zs
            student_z = self.student_model.encode(zs[0].clone().detach(), task).unsqueeze(1).repeat(1, num_noises, 1) # [BS, NS, FS]
            # student_z = self.student_model.encode(obs[0].clone().detach(), task).unsqueeze(1).repeat(1, num_noises, 1) # [BS, NS, FS] # encode from obs
            student_zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, num_noises, self.student_model.latent_dim_agent * self.student_model.num_agents, device=self.device) # [H, BS, NS, FS]
            student_zs[0] = student_z
            student_zs_sp = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, num_noises, self.student_model.latent_dim_agent * self.student_model.num_agents, device=self.device) # [H, BS, NS, FS]
            student_zs_sp[0] = student_z.detach() # self_predictive
            global_z = zs[0].clone().detach().unsqueeze(1).repeat(1, num_noises, 1) # [BS, NS, S]
            global_zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, num_noises, self.student_cfg.latent_dim, device=self.device) # [H, BS, NS, S]
            global_zs[0] = global_z 

            # vae loss for latent actions
            if 'tap' in self.student_cfg.model_name:
                raw_action_clone = student_action_sample.clone().view(self.cfg.horizon, -1, self.student_cfg.action_dim).transpose(0, 1) # [BS*NS, H, A]
                recon_action_sample, mu, logvar, latent_action_sample = self.student_model.forward(student_z.view(self.cfg.batch_size*num_noises, -1), raw_action_clone)
                vae_loss = math.vae_loss(recon_action_sample, raw_action_clone, mu, logvar)
                student_action_sample = latent_action_sample.transpose(0, 1).view(self.cfg.horizon, -1, num_noises, self.student_model.latent_action_dim) # [H, BS, NS, LA]
            else:
                vae_loss = None

            # rollout in latent space
            student_consistency_loss = 0.0
            for t, _action in enumerate(action_sample.unbind(0)):
                student_z = self.student_model.next(student_z, student_action_sample[t], task)
                student_zs[t+1] = student_z
                global_z = self.model.next(global_z, _action, task).detach()
                global_zs[t+1] = global_z
                student_z_sp = self.student_model.encode(global_z, task).detach()
                student_zs_sp[t+1] = student_z_sp
                student_consistency_loss = student_consistency_loss + F.mse_loss(student_z, student_z_sp) * self.student_cfg.rho**t

            _student_zs_sample = student_zs[:-1]
            _student_zs_sample = _student_zs_sample.view(self.student_cfg.horizon, -1, self.student_model.latent_dim_agent * self.student_model.num_agents) # [H, BS*NS, FS]
            _zs_sample = global_zs[:-1]
            _zs_sample = _zs_sample.view(self.student_cfg.horizon, -1, self.student_cfg.latent_dim) # [H, BS*NS, S]
            action_sample = action_sample.view(self.student_cfg.horizon, -1, self.student_cfg.action_dim) # [H, BS*NS, A]
            student_action_sample = student_action_sample.view(self.student_cfg.horizon, self.cfg.batch_size*num_noises, -1) # [H, BS*NS, A]
            if task is not None:
                task = task.unsqueeze(-1).repeat(1, num_noises).view(-1) # [BS*NS]

            # MC: monotonic on which functions
            # ====== c1: monotonic R/Q ====== # 
            student_rewards = self.student_model.reward(_student_zs_sample, student_action_sample, task) # [H, BS*NS, 1]
            student_qs = self.student_model.Q(_student_zs_sample, student_action_sample, task, return_type='all') # [Q, H, BS*NS, 1]
            target_rewards = self.model.reward(_zs_sample, action_sample, task).detach() # [H, BS*NS, num_bins]
            target_qs = self.model.Q(_zs_sample, action_sample, task, return_type='avg', detach=True).detach() # [H, BS*NS, 1]
            
            # Compute losses
            student_reward_loss, student_value_loss = 0, 0
            for t, (pred_rew_unbind, rew_unbind, pred_qs_unbind, qs_unbind) in enumerate(zip(student_rewards.unbind(0), target_rewards.unbind(0), student_qs.unbind(1), target_qs.unbind(0))):
                rew_unbind = math.two_hot_inv(rew_unbind, self.cfg) # expert models
                # MC: mse loss or kl loss
                if self.student_cfg.loss_type == 'mse':
                    student_reward_loss = student_reward_loss + math.soft_ce(pred_rew_unbind, rew_unbind, self.student_cfg).mean() * self.student_cfg.rho**t
                else:
                    pred_rew_unbind = pred_rew_unbind.view(self.cfg.batch_size, num_noises)
                    rew_unbind = rew_unbind.view(self.cfg.batch_size, num_noises)
                    student_reward_loss = student_reward_loss + math.softmax_distillation_loss(pred_rew_unbind, rew_unbind, self.student_cfg.temperature_noises).mean() * self.student_cfg.rho**t
                for _, pred_qs_unbind_unbind in enumerate(pred_qs_unbind.unbind(0)):
                    if self.student_cfg.loss_type == 'mse':
                        student_value_loss = student_value_loss + math.soft_ce(pred_qs_unbind_unbind, qs_unbind, self.student_cfg).mean() * self.student_cfg.rho**t
                    else:
                        pred_qs_unbind_unbind = pred_qs_unbind_unbind.view(self.cfg.batch_size, num_noises)
                        qs_unbind = qs_unbind.view(self.cfg.batch_size, num_noises)
                        student_value_loss = student_value_loss + math.softmax_distillation_loss(pred_qs_unbind_unbind, qs_unbind, self.student_cfg.temperature_noises).mean() * self.student_cfg.rho**t
            # ============ # 

            # # ==== c2: monotonic Return ==== # 
            # student_rewards = self.student_model.reward(_student_zs_sample, action_sample, task, return_individual=True)
            # student_qs = self.student_model.Q(_student_zs_sample, action_sample, task, return_type='all', return_individual=True)
            # target_rewards = self.model.reward(_zs_sample, action_sample, task).detach()
            # target_qs = self.model.Q(_zs_sample, action_sample, task, return_type='avg', detach=True).detach() 
            
            # student_reward_loss, student_value_loss = 0, 0
            # student_return, global_return = 0.0, 0.0
            # discount = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
            # for t, (pred_rew_unbind, rew_unbind, pred_qs_unbind, qs_unbind) in enumerate(zip(student_rewards.unbind(0), target_rewards.unbind(0), student_qs.unbind(1), target_qs.unbind(0))):
            #     tmp_global_return = global_return + qs_unbind * discount**t
            #     for _, pred_qs_unbind_unbind in enumerate(pred_qs_unbind.unbind(0)):
            #         if t < self.cfg.horizon - 1: break # new2
            #         tmp_student_return = student_return + pred_qs_unbind_unbind * discount**t
            #         tmp_student_return_mix = self.student_model._value_mixer(tmp_student_return.squeeze(-1) ) #, _zs_sample[0])
            #         student_value_loss = student_value_loss + math.soft_ce(tmp_student_return_mix, tmp_global_return, self.cfg).mean()  * self.cfg.student_rho**t
            #     pred_rew_unbind_mix = self.student_model._reward_mixer(pred_rew_unbind.squeeze(-1) ) #, _zs_sample[0])
            #     student_reward_loss = student_reward_loss + math.soft_ce(pred_rew_unbind_mix, rew_unbind, self.cfg).mean() * self.cfg.student_rho**t
            #     student_return += pred_rew_unbind * discount**t
            #     global_return += rew_unbind * discount**t
            # # ======== # 

            student_consistency_loss = student_consistency_loss / self.student_cfg.horizon
            student_reward_loss = student_reward_loss / self.student_cfg.horizon
            student_value_loss = student_value_loss / (self.student_cfg.horizon * self.student_cfg.num_q)

            student_total_loss = (
                self.student_cfg.consistency_coef * student_consistency_loss +
                self.student_cfg.reward_coef * student_reward_loss +
                self.student_cfg.value_coef * student_value_loss
            )

            if vae_loss is not None:
                student_total_loss += self.student_cfg.vae_coef * vae_loss

            student_total_loss.backward()
            student_grad_norm = torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.student_cfg.grad_clip_norm)
            self.student_optim.step()
            self.student_optim.zero_grad(set_to_none=True)

            return_dict["student_consistency_loss"] = student_consistency_loss.detach().mean()
            return_dict["student_reward_loss"] = student_reward_loss.detach().mean()
            return_dict["student_value_loss"] = student_value_loss.detach().mean()
            return_dict["student_grad_norm"] = student_grad_norm.detach().mean()
            if vae_loss is not None:
                return_dict["action_vae_loss"] = vae_loss.detach().mean()

            self.student_model.eval()

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
