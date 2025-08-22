import os
import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage, TensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler
from tqdm import tqdm


class Buffer():
    """
    Replay buffer for TD-MPC2 training. Based on torchrl.
    Uses CUDA memory if available, and CPU memory otherwise.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._device = torch.device('cuda:0')
        self._capacity = min(cfg.buffer_size, cfg.steps)
        self._sampler = SliceSampler(
            num_slices=self.cfg.batch_size,
            end_key=None,
            traj_key='episode',
            truncated_key=None,
            strict_length=True,
        )
        self._batch_size = cfg.batch_size * (cfg.horizon+1)
        self._num_eps = 0

    @property
    def capacity(self):
        """Return the capacity of the buffer."""
        return self._capacity

    @property
    def num_eps(self):
        """Return the number of episodes in the buffer."""
        return self._num_eps

    def _reserve_buffer(self, storage):
        """
        Reserve a buffer with the given storage.
        """
        return ReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=False,
            prefetch=0,
            batch_size=self._batch_size,
        )

    def _init(self, tds):
        """Initialize the replay buffer. Use the first episode to estimate storage requirements."""
        print(f'Buffer capacity: {self._capacity:,}')
        mem_free, _ = torch.cuda.mem_get_info()
        bytes_per_step = sum([
                (v.numel()*v.element_size() if not isinstance(v, TensorDict) \
                else sum([x.numel()*x.element_size() for x in v.values()])) \
            for v in tds.values()
        ]) / len(tds)
        total_bytes = bytes_per_step*self._capacity
        print(f'Storage required: {total_bytes/1e9:.2f} GB')
        # Heuristic: decide whether to use CUDA or CPU memory
        storage_device = 'cuda:0' if 2.5*total_bytes < mem_free else 'cpu'
        print(f'Using {storage_device.upper()} memory for storage.')
        self._storage_device = torch.device(storage_device)
        return self._reserve_buffer(
            LazyTensorStorage(self._capacity, device=self._storage_device)
        )

    def _prepare_batch(self, td):
        """
        Prepare a sampled batch for training (post-processing).
        Expects `td` to be a TensorDict with batch size TxB.
        """
        td = td.select("obs", "action", "reward", "task", strict=False).to(self._device, non_blocking=True)
        obs = td.get('obs').contiguous()
        action = td.get('action')[1:].contiguous()
        reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
        task = td.get('task', None)
        if task is not None:
            task = task[0].contiguous()
        return obs, action, reward, task

    def add(self, td):
        """Add an episode to the buffer."""
        td['episode'] = torch.ones_like(td['reward'], dtype=torch.int64) * torch.arange(self._num_eps, self._num_eps+self.cfg.num_envs)
        td = td.permute(1, 0)
        if self._num_eps == 0:
            self._buffer = self._init(td[0])
        for i in range(self.cfg.num_envs):
            self._buffer.extend(td[i])
        self._num_eps += self.cfg.num_envs
        return self._num_eps

    def sample(self):
        """Sample a batch of subsequences from the buffer."""
        td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
        return self._prepare_batch(td)

    def save(self, path):
        """Save the replay buffer state."""
        if not hasattr(self, '_buffer'):
            print("Buffer not initialized yet.")
            return
        save_data = {
            'num_eps': self._num_eps,
            # 'cfg': self.cfg,  # only if it's serializable
            'storage_device': str(self._storage_device),
            'buffer_state': self._buffer.state_dict()
        }
        torch.save(save_data, path)
        print(f"Buffer saved to {path}")

    def load(self, path):
        """Load the replay buffer from a file."""
        load_data = torch.load(path, map_location='cpu')
        self._num_eps = load_data['num_eps']
        self._storage_device = torch.device(load_data['storage_device'])

        # Recreate buffer and load its state dict
        storage = LazyTensorStorage(self._capacity, device=self._storage_device)
        self._buffer = self._reserve_buffer(storage)
        self._buffer.load_state_dict(load_data['buffer_state'])
        print(f"Load {self._num_eps} episodes from {path}")

    def load_multitask(self, td):
        num_new_eps = len(td)
        episode_idx = torch.arange(self._num_eps, self._num_eps+num_new_eps, dtype=torch.int64)
        td['episode'] = episode_idx.unsqueeze(-1).expand(-1, td['reward'].shape[1])
        if self._num_eps == 0:
            self._buffer = self._init(td[0])
        td = td.reshape(td.shape[0]*td.shape[1])
        self._buffer.extend(td)
        self._num_eps += num_new_eps
        return self._num_eps

    def materialize_buffer(self):
        """Convert LazyTensorStorage to TensorStorage for faster sampling.

        This function:
        1. Builds a prototype TensorDict whose batch dimension is [capacity]
            (so TensorStorage can infer the correct storage shape).
        2. Initializes a TensorStorage with that prototype.
        3. Copies all existing samples from the old buffer into the new storage.
        """
        if not hasattr(self, "_buffer"):
            raise RuntimeError("Buffer not initialized. Load or add episodes first.")

        old_storage = self._buffer._storage
        # Use exact type check to avoid inheritance confusion
        if type(old_storage) is TensorStorage:
            print("Buffer is already materialized (TensorStorage).")
            return

        print("Materializing buffer: copying from LazyTensorStorage to TensorStorage...")
        # Grab one sample to infer field shapes and dtypes
        example_td = self._buffer[0]
        if not isinstance(example_td, TensorDict):
            raise RuntimeError("Unexpected sample type when materializing buffer.")

        # Filter nan rewards/actions, the first transition in episode
        n_items = len(self._buffer)  # number of stored elements
        real_n_items = n_items - self._num_eps
        
        # Build prototype fields with batch dim = capacity
        prototype_fields = {}
        for k, v in example_td.items():
            # v is a tensor like shape [12] (action) or [] (scalar)
            # target prototype shape: (capacity, *v.shape)
            target_shape = (real_n_items,) + tuple(v.shape)
            # create zeros on the target device with same dtype
            prototype_fields[k] = torch.zeros(
                target_shape, dtype=v.dtype, device=self._storage_device
            )

        # Prototype TensorDict has batch_size = [capacity]
        prototype_td = TensorDict(prototype_fields, batch_size=[real_n_items])

        # Create TensorStorage with prototype and max_size = capacity
        storage = TensorStorage(prototype_td, device=self._storage_device, max_size=real_n_items)

        # Copy all samples from old buffer into new storage
        cursor = 0
        for i in tqdm(range(n_items)):
            sample_td = self._buffer[i]  # this is a TensorDict with batch_size=[]
            # Skip invalid transitions (first step of each episode)
            if torch.isnan(sample_td["reward"]).any() or torch.isnan(sample_td["action"]).any():
                continue
            # build a batched one-element TensorDict for assignment
            batched = {}
            for k, v in sample_td.items():
                # ensure shape is [1, *field_shape]
                batched[k] = v.unsqueeze(0).to(self._storage_device)
            storage[cursor] = TensorDict(batched, batch_size=[1])
            cursor += 1

        # create a new ReplayBuffer that wraps the storage
        self._buffer = self._reserve_buffer(storage)
        print(f"Materialization complete: copied {real_n_items}/{n_items} entries into TensorStorage.")