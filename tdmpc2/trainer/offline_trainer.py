import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from common.buffer import Buffer
from trainer.base import Trainer


class OfflineTrainer(Trainer):
	"""Trainer class for multi-task offline TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._start_time = time()
		self._step = 0

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			iteration=self._step,
			step=self._step,
			total_time=time() - self._start_time,
		)
	
	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		results = dict()
		for task_idx in tqdm(range(len(self.cfg.tasks)), desc='Evaluating'):
			ep_rewards, ep_successes = [], []
			for _ in range(self.cfg.eval_episodes // self.cfg.num_envs):
				obs, done, ep_reward, t = self.env.reset(task_idx), torch.tensor(False), 0, 0
				while not done.any():
					torch.compiler.cudagraph_mark_step_begin()
					action = self.agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
					obs, reward, done, info = self.env.step(action)
					ep_reward += reward
					t += 1
				ep_rewards.append(ep_reward)
				ep_successes.append(info['success'])
			results.update({
				f'episode_reward': torch.cat(ep_rewards).mean(), # for logger save working
				f'episode_reward+{self.cfg.tasks[task_idx]}': torch.cat(ep_rewards).mean(),
				f'episode_success+{self.cfg.tasks[task_idx]}': torch.cat(ep_successes).mean(),})
		return results

	def train(self):
		"""Train a TD-MPC2 agent."""
		# assert self.cfg.multitask and self.cfg.task in {'mt30', 'mt80'}, \
		# 	'Offline training only supports multitask training with mt30 or mt80 task sets.'

		# Load data
		self.buffer = Buffer(self.cfg)
		self.buffer.load(self.cfg.data_dir)
		# fp = Path(os.path.join(self.cfg.data_dir, '*.pt'))
		# fps = sorted(glob(str(fp)))
		# assert len(fps) > 0, f'No data found at {fp}'
		# print(f'Found {len(fps)} files in {fp}')
		# assert len(fps) == (20 if self.cfg.task == 'mt80' else 4), \
		# 	f'Expected 20 files for mt80 task set, 4 files for mt30 task set, found {len(fps)} files.'
	
		# # Create buffer for sampling
		# _cfg = deepcopy(self.cfg)
		# _cfg.episode_length = 101 if self.cfg.task == 'mt80' else 501
		# _cfg.buffer_size = 550_450_000 if self.cfg.task == 'mt80' else 345_690_000
		# _cfg.steps = _cfg.buffer_size
		# self.buffer = Buffer(_cfg)
		# for fp in tqdm(fps, desc='Loading data'):
		# 	td = torch.load(fp)
		# 	assert td.shape[1] == _cfg.episode_length, \
		# 		f'Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, ' \
		# 		f'please double-check your config.'
		# 	for i in range(len(td)):
		# 		self.buffer.add(td[i])
		# expected_episodes = _cfg.buffer_size // _cfg.episode_length
		# assert self.buffer.num_eps == expected_episodes, \
		# 	f'Buffer has {self.buffer.num_eps} episodes, expected {expected_episodes} episodes.'
		
		best_episode_reward = 0.0
		print(f'Training agent for {self.cfg.steps} iterations...')
		for i in range(self.cfg.steps+1):
			self._step = i

			# Update agent
			train_metrics = self.agent.update(self.buffer)

			# Evaluate agent periodically
			if i % self.cfg.eval_freq == 0 or i % 1000 == 0:
				train_metrics.update(self.common_metrics())
				self.logger.log(train_metrics, 'pretrain')
				if i % self.cfg.eval_freq == 0:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					# metrics.update(self.eval())
					# self.logger.pprint_multitask(metrics, self.cfg)
					# save best model
					if eval_metrics['episode_reward'] > best_episode_reward:
						best_episode_reward = eval_metrics['episode_reward'] 
						model_path = f"{self.cfg.work_dir}/models/best.pt"
						self.agent.save(model_path)
			
		self.logger.finish(self.agent)
