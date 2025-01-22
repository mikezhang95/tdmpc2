import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import from_modules
from copy import deepcopy
from common import layers, math, init

class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
	"""
	def __init__(self, modules, **kwargs):
		super().__init__()
		# combine_state_for_ensemble causes graph breaks
		self.params = from_modules(*modules, as_module=True)
		self.module = deepcopy(modules[0])
		with self.params[0].data.to("meta").to_module(modules[0]):
			self.module = deepcopy(modules[0])
		self._repr = str(modules)

	def _call(self, params, *args, **kwargs):
		with params.to_module(self.module):
			return self.module(*args, **kwargs)

	def forward(self, *args, **kwargs):
		return torch.vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs)

	def __repr__(self):
		return 'Vectorized ' + self._repr


class ShiftAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad
		self.padding = tuple([self.pad] * 4)

	def forward(self, x):
		x = x.float()
		n, _, h, w = x.size()
		assert h == w
		x = F.pad(x, self.padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
	"""
	Normalizes pixel observations to [-0.5, 0.5].
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.).sub(0.5)


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))

	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}{repr_dropout}, "\
			f"act={self.act.__class__.__name__})"


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


def conv(in_shape, num_channels, act=None):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
	layers = [
		ShiftAug(), PixelPreprocess(),
		nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]
	if act:
		layers.append(act)
	return nn.Sequential(*layers)


def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	for k in cfg.obs_shape.keys():
		if k == 'state':
			out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
		elif k == 'rgb':
			out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
		else:
			raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)


class FullyConnectedGraph(nn.Module):
	"""
	FullyConnectedGraph layer with LayerNorm, activation, and optionally dropout.
	M: currently doesn't need aggregate function in GNN, only use this structure to easily calculate node/edge features
	"""
	def __init__(self, num_nodes, node_in_dim, mlp_dims, node_out_dim, act=None, dropout=0., temperature=1e-2, edge_logits=None):
		super(FullyConnectedGraph, self).__init__()
		self.num_nodes = num_nodes
		self.num_edges = num_nodes * (num_nodes - 1) // 2

		# Define custom MLPs or other functions for node and edge updates
		# M: can be extended to non-shared networks, then use for-loop should run fast
		self.shared_parameters = True
		if self.shared_parameters:
			self.node_update = mlp(node_in_dim, mlp_dims, node_out_dim, act=act, dropout=dropout)
			self.edge_update = mlp(node_in_dim + node_in_dim, mlp_dims, node_out_dim, act=act, dropout=dropout)
		else:
			self.node_update = nn.ModuleList([mlp(node_in_dim, mlp_dims, node_out_dim, act=act, dropout=dropout) for i in range(self.num_nodes)])
			self.edge_update = nn.ModuleList([mlp(node_in_dim + node_in_dim, mlp_dims, node_out_dim, act=act, dropout=dropout) for i in range(self.num_edges*2)])

		# Create fully connected graph edges
		edge_index = torch.combinations(torch.arange(self.num_nodes), r=2).T
		edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Add reverse edges: [2, num_edges * 2] 
		self.edge_index = edge_index.to('cuda') # TODO: cpu support
		if edge_logits is None:
			self.edge_logits = nn.Parameter(torch.randn(self.num_edges), requires_grad=True)  
		else:
			self.edge_logits = edge_logits
		self.temperature = temperature


	def forward(self, node_features):

		# Compute edge features
		src, dest = self.edge_index
		edge_features = torch.cat([node_features[..., src, :].clone(), node_features[..., dest, :].clone()], dim=-1)
		if self.shared_parameters:
			edge_features = self.edge_update(edge_features)
			# average i->j and j->i
			edge_outputs  = torch.mean(torch.reshape(edge_features, (*edge_features.shape[:-2], self.num_edges, 2, -1)), dim=-2) 
		else:
			outputs = []
			for i in range(self.num_edges):
				# average i->j and j->i
				edge_average = (self.edge_update[i](edge_features[::, i, :])  + self.edge_update[i+self.num_edges](edge_features[::, i+self.num_edges, :]) ) / 2
				outputs.append(edge_average)
			edge_outputs = torch.stack(outputs, dim=-2)

		# Update node outputs
		if self.shared_parameters:
			node_outputs = self.node_update(node_features)
		else:
			outputs = []
			for i in range(self.num_nodes):
				outputs.append(self.node_update[i](node_features[::, i, :]))
			node_outputs = torch.stack(outputs, dim=-2)
		
		# Rescale edge representation
		edge_outputs = self.rescale_edges(edge_outputs.transpose(-2, -1)).transpose(-2, -1)

		return node_outputs, edge_outputs


	def rescale_edges(self, edges):
		"""
			Args:
				- edges: [*, num_edges]
			Returns: 
				- edges: [num_edges]
		"""
		if self.training: # keep gradients
			# TODO: now edge weights are same across the whole batch, try sample differently
			# edge_weights = torch.sigmoid(self.edge_logits / self.temperature)
			edge_weights = math.relaxed_bernoulli_reparameterization(self.edge_logits, temperature=self.temperature)
		else: # remove gradients
			edge_weights = (torch.sigmoid(self.edge_logits.detach()) > 0.5).float()
		return edges * edge_weights

	# def _generate_bernoulli_samples(self, size):
	# 	"""
	# 		Returns: 
	# 			- samples: [size, num_edges]
	# 	"""
	# 	num_samples = math.tuple_product(size)
	# 	# M: due to efficiency reasons, same samples in one batch
	# 	batch_edge_logits = self.edge_logits.expand(num_samples, -1).clone()
	# 	# batch_edge_logits = self.edge_logits
	# 	if self.training: # keep gradients
	# 		samples = math.relaxed_bernoulli_reparameterization(batch_edge_logits, temperature=self.cfg.temperature).float()
	# 	else: # remove gradients
	# 		samples = (torch.sigmoid(batch_edge_logits.detach()) > 0.5).float()
	# 	# samples = samples.expand(num_samples, -1).clone() # /M: clone is necessary for compile mode
	# 	samples = torch.reshape(samples, (*size, self.num_edges))
	# 	return samples
