import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

import math

__REDUCE__ = lambda b: 'mean' if b else 'none'


def l1(pred, target, reduce=False):
	"""Computes the L1-loss between predictions and targets."""
	return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
	"""Computes the MSE loss between predictions and targets."""
	return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


def _get_out_shape(in_shape, layers):
	"""Utility function. Returns the output shape of a network for a given input shape."""
	x = torch.randn(*in_shape).unsqueeze(0)
	return (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x).squeeze(0).shape


def orthogonal_init(m):
	"""Orthogonal layer initialization."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if m.bias is not None:
			nn.init.zeros_(m.bias)
	elif isinstance(m, nn.Conv2d):
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data, gain)
		if m.bias is not None:
			nn.init.zeros_(m.bias)


def ema(m, m_target, tau):
	"""Update slow-moving average of online network (target network) at rate tau."""
	with torch.no_grad():
		for p, p_target in zip(m.parameters(), m_target.parameters()):
			p_target.data.lerp_(p.data, tau)


def set_requires_grad(net, value):
	"""Enable/disable gradients for a given (sub)network."""
	for param in net.parameters():
		param.requires_grad_(value)


class TruncatedNormal(pyd.Normal):
	"""Utility class implementing the truncated normal distribution."""
	def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
		super().__init__(loc, scale, validate_args=False)
		self.low = low
		self.high = high
		self.eps = eps

	def _clamp(self, x):
		clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
		x = x - x.detach() + clamped_x.detach()
		return x

	def sample(self, clip=None, sample_shape=torch.Size()):
		shape = self._extended_shape(sample_shape)
		eps = _standard_normal(shape,
							   dtype=self.loc.dtype,
							   device=self.loc.device)
		eps *= self.scale
		if clip is not None:
			eps = torch.clamp(eps, -clip, clip)
		x = self.loc + eps
		return self._clamp(x)


class NormalizeImg(nn.Module):
	"""Normalizes pixel observations to [0,1) range."""
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.)


class Flatten(nn.Module):
	"""Flattens its input to a (batched) vector."""
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x.view(x.size(0), -1)


def enc(cfg):
	"""Returns a TOLD encoder."""
	layers = [nn.Linear(cfg.obs_shape['state'][0], cfg.enc_dim), nn.ELU(), nn.Linear(cfg.enc_dim, cfg.latent_dim)]
	return nn.Sequential(*layers)


def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
	"""Returns an MLP."""
	if isinstance(mlp_dim, int):
		mlp_dim = [mlp_dim, mlp_dim]
	return nn.Sequential(
		nn.Linear(in_dim, mlp_dim[0]), act_fn,
		nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
		nn.Linear(mlp_dim[1], out_dim))

def q(in_dim, mlp_dim, out_dim):
	"""Returns a Q-function that uses Layer Normalization."""
	return nn.Sequential(nn.Linear(in_dim, mlp_dim), nn.LayerNorm(mlp_dim), nn.Tanh(),
						 nn.Linear(mlp_dim, mlp_dim), nn.ELU(),
						 nn.Linear(mlp_dim, out_dim))


class RandomShiftsAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, cfg):
		super().__init__()
		self.pad = None 

	def forward(self, x):
		if not self.pad:
			return x
		n, c, h, w = x.size()
		assert h == w
		padding = tuple([self.pad] * 4)
		x = F.pad(x, padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class FacMLP(nn.Module):
    """
    FullyConnectedGraph layer with LayerNorm, activation, and optionally dropout.
    M: currently doesn't need aggregate function in GNN, only use this structure to easily calculate node/edge features
    """
    def __init__(self, num_nodes, node_in_dim, mlp_dims, node_out_dim, act=None, dropout=0., is_q=False, shared_parameters=False):
        super(FacMLP, self).__init__()
        self.num_nodes = num_nodes

        # Define custom MLPs or other functions for node and edge updates
        # M: can be extended to non-shared networks, then use for-loop should run fast
        self.shared_parameters = shared_parameters
        if self.shared_parameters:
            if is_q:
                self.node_update = q(node_in_dim, mlp_dims[0], node_out_dim)
            else:
                self.node_update = mlp(node_in_dim, mlp_dims, node_out_dim)
        else:
            # M: naive implementation
            # self.node_update = nn.ModuleList([mlp(node_in_dim, mlp_dims, node_out_dim, act=act, dropout=dropout) for i in range(self.num_nodes)])
            # M: vectorize implementation
            node_dims = [node_in_dim] + mlp_dims + [node_out_dim]
            self.node_update = []
            for i in range(len(node_dims)-1):
                if i == len(node_dims) - 2: layer_act = act # Follow TDMPC2, last layer use customized act
                else: layer_act = nn.ELU(inplace=False)
                if i == 0: layer_dropout = dropout
                else: layer_dropout = 0.
                use_layer_norm=False
                if is_q and i==0 :
                    use_layer_norm = True
                    layer_act = nn.Tanh()
                self.node_update.append(VectorizedLinearLayer(self.num_nodes, node_dims[i], node_dims[i+1], 
                                                              use_layer_norm=use_layer_norm, act=layer_act, dropout=layer_dropout))
            self.node_update = nn.Sequential(*self.node_update)

    def forward(self, node_features):
        """
        Args:
            - node_features: [..., num_nodes, feature_dim]
        """
        # Reshape inputs to 3D vectors
        raw_input_shape = node_features.shape # [..., num_nodes, input_dim]
        node_features = node_features.view(-1, raw_input_shape[-2], raw_input_shape[-1]).contiguous() # [B, num_nodes, input_dim]

        # Update node outputs
        node_outputs = self.node_update(node_features)

        # Reshape outputs back to 4D/3D vectors
        node_outputs = node_outputs.view(*raw_input_shape[:-2], self.num_nodes, -1).contiguous() # [..., num_nodes, node_dim]
        return node_outputs


class AttnMLP(nn.Module):
    """
    FullyConnectedGraph layer with LayerNorm, activation, and optionally dropout.
    M: currently doesn't need aggregate function in GNN, only use this structure to easily calculate node/edge features
    """
    def __init__(self, num_nodes, node_in_dim, mlp_dims, node_out_dim, act=None, dropout=0., is_q=False, shared_parameters=False):
        super(AttnMLP, self).__init__()
        self.num_nodes = num_nodes

        # TODO: hyperparameters for attention layer
        embed_dim = mlp_dims[0]
        self._attn_encoder = AttentionEncoder(
            input_dim=node_in_dim, 
            embed_dim=embed_dim,
            num_nodes=num_nodes,
            num_heads=1,
            num_layers=2,
            shared_parameters = True
        )

        # Define custom MLPs or other functions for node and edge updates
        # M: can be extended to non-shared networks, then use for-loop should run fast
        self.shared_parameters = shared_parameters
        if self.shared_parameters:
            if is_q:
                self.node_update = q(embed_dim, mlp_dims[0], node_out_dim)
            else:
                self.node_update = mlp(embed_dim, mlp_dims, node_out_dim)
        else:
            # M: naive implementation
            # self.node_update = nn.ModuleList([mlp(node_in_dim, mlp_dims, node_out_dim, act=act, dropout=dropout) for i in range(self.num_nodes)])
            # M: vectorize implementation
            node_dims = mlp_dims + [node_out_dim]
            self.node_update = []
            for i in range(len(node_dims)-1):
                if i == len(node_dims) - 2: layer_act = act # Follow TDMPC2, last layer use customized act
                else: layer_act = nn.ELU(inplace=False)
                if i == 0: layer_dropout = dropout
                else: layer_dropout = 0.
                use_layer_norm=False
                if is_q and i==0 :
                    use_layer_norm = True
                    layer_act = nn.Tanh()
                self.node_update.append(VectorizedLinearLayer(self.num_nodes, node_dims[i], node_dims[i+1], 
                                                              use_layer_norm=use_layer_norm, act=layer_act, dropout=layer_dropout))
            self.node_update = nn.Sequential(*self.node_update)


    def forward(self, node_features):
        """
        Args:
            - node_features: [..., num_nodes, feature_dim]
        """
        # Reshape inputs to 3D vectors
        raw_input_shape = node_features.shape # [..., num_nodes, input_dim]
        node_features = node_features.view(-1, raw_input_shape[-2], raw_input_shape[-1]).contiguous() # [B, num_nodes, input_dim]

        # Update node outputs
        node_features = self._attn_encoder(node_features) # [B, num_nodes, embe_dim]
        node_outputs = self.node_update(node_features) # [B, num_nodes, output_dim]

        # Reshape outputs back to 4D/3D vectors
        node_outputs = node_outputs.view(*raw_input_shape[:-2], self.num_nodes, -1).contiguous() # [..., num_nodes, node_dim]
        return node_outputs


class VectorizedLinearLayer(nn.Module):
    """Vectorized version of torch.nn.Linear."""

    def __init__(
        self,
        population_size: int,
        in_features: int,
        out_features: int,
        use_layer_norm: bool = False,
        dropout: float = 0., 
        act = None,
    ):
        super().__init__()
        self._population_size = population_size
        self._in_features = in_features
        self._out_features = out_features

        # linear weights
        self.weight = torch.nn.Parameter(
            torch.empty(self._population_size, self._in_features, self._out_features),
            requires_grad=True,
        )
        self.bias = torch.nn.Parameter(
            torch.empty(self._population_size, 1, self._out_features),
            requires_grad=True,
        )

        # M: init
        for member_id in range(population_size):
            torch.nn.init.orthogonal_(self.weight[member_id].data)
        torch.nn.init.zeros_(self.bias)
        # for member_id in range(population_size):
        #     torch.nn.init.kaiming_uniform_(self.weight[member_id], a=math.sqrt(5))
        # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        # torch.nn.init.uniform_(self.bias, -bound, bound)

        # layernorm
        self._layer_norm = (
            torch.nn.LayerNorm(self._out_features)
            if use_layer_norm
            else None
        )

        # M: activation and dropout
        self._act = act
        self._dropout = nn.Dropout(dropout, inplace=False) if dropout else None

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: [batch_size, population, in_features]
        Returns:
            - x: [batch_size, population, out_features]
        """
        x = x.transpose(0, 1)
        assert x.shape[0] == self._population_size
        x = x.matmul(self.weight) + self.bias
        if self._layer_norm is not None:
            x = self._layer_norm(x)
        if self._dropout:
            x = self._dropout(x)
        if self._act:
            x = self._act(x)
        x = x.transpose(0, 1)
        return x
    
    def __repr__(self):
        repr_dropout = f"dropout={self._dropout.p}" if self._dropout else "dropout=None"
        repr_ln = f"ln={self._layer_norm.__class__.__name__}" if self._layer_norm else "ln=None"
        repr_act = f"act={self._act.__class__.__name__})" if self._act else "act=None)"
        return f"VectorizedLinearLayer(population_size={self._population_size}, in_features={self._in_features}, "\
            f"out_features={self._out_features}, "\
            f"bias={self.bias is not None}, "\
            f"{repr_dropout}, "\
            f"{repr_ln}, "\
            f"{repr_act}"


class AttentionEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_nodes=6, num_heads=2, num_layers=2, shared_parameters=True):
        super().__init__()
        self.index_embedding = nn.Embedding(num_nodes, embed_dim)
        if shared_parameters:
            self.input_proj = nn.Linear(input_dim, embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_proj = nn.Linear(embed_dim, embed_dim)
        else:
            self.input_proj = VectorizedLinearLayer(num_nodes, input_dim, embed_dim)
            self.encoder = nn.Sequential(*[HeteAttentionLayer(num_nodes, embed_dim) for i in range(num_layers)])
            self.output_proj = VectorizedLinearLayer(num_nodes, embed_dim, embed_dim)
        
        self._input_dim = input_dim 
        self._embed_dim = embed_dim
        self._num_nodes = num_nodes
        self._num_heads = num_heads
        self._num_layers = num_layers

    def forward(self, x):
        """
        Args:
            - x: [..., num_nodes, feature_dim]
        Returns:
            - x: [..., num_nodes, embed_dim]
        """
        # reshape to 3-D tensors
        raw_input_shape = x.shape # [..., num_nodes, input_dim]
        x = x.view(-1, raw_input_shape[-2], raw_input_shape[-1]).contiguous() # [B, num_nodes, input_dim]

        # proj input 
        x = self.input_proj(x)
        idx = torch.arange(raw_input_shape[-2], device=x.device)
        index_embed = self.index_embedding(idx)  # [N, E]
        x = x + index_embed.unsqueeze(0)         # broadcast to [B, N, E]
        # encode with transformer
        x = self.encoder(x)
        # proj output 
        x = self.output_proj(x)

        # shape back to tensors
        x = x.view(*raw_input_shape[:-2], raw_input_shape[-2], -1).contiguous() # [..., num_nodes, node_dim]
        return x

    def __repr__(self):
        return f"SharedAttention(input_dim={self._input_dim}, embed_dim={self._embed_dim}, "\
                f"num_nodes={self._num_nodes}, num_heads={self._num_heads}, num_layers={self._num_layers})"


class HeteAttentionLayer(nn.Module):
    def __init__(self, num_agents, embed_dim):
        super(HeteAttentionLayer, self).__init__()
        self.emb_dim = embed_dim

        # Linear layers for query, key, and value transformations
        self.query_proj = VectorizedLinearLayer(num_agents, embed_dim, embed_dim)
        self.key_proj = VectorizedLinearLayer(num_agents, embed_dim, embed_dim)
        self.value_proj = VectorizedLinearLayer(num_agents, embed_dim, embed_dim)

    def forward(self, x):
        # Project q,k,v
        Q = self.query_proj(x)  # [B, num_actions, 64]
        K = self.key_proj(x)      # [B, num_actions, 64]
        V = self.value_proj(x)  # [B, num_actions, 64]

        # Compute scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.emb_dim ** 0.5)

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)   # [B, 21, 21]
        attn_output = torch.matmul(attn_weights, V)                 # [B, num_actions, 64]

        return attn_output
