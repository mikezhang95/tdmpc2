import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from tensordict import from_modules
from copy import deepcopy
import math


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
    def __init__(self, num_nodes, node_in_dim, mlp_dims, node_out_dim, act=None, dropout=0.):
        super(FullyConnectedGraph, self).__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_nodes * (num_nodes - 1) // 2

        # Define custom MLPs or other functions for node and edge updates
        # M: can be extended to non-shared networks, then use for-loop should run fast
        self.shared_parameters = False # True
        if self.shared_parameters:
            self.node_update = mlp(node_in_dim, mlp_dims, node_out_dim, act=act, dropout=dropout)
            self.edge_update = mlp(node_in_dim + node_in_dim, mlp_dims, node_out_dim, act=act, dropout=dropout)
        else:
            # naive implementation
            # self.node_update = nn.ModuleList([mlp(node_in_dim, mlp_dims, node_out_dim, act=act, dropout=dropout) for i in range(self.num_nodes)])
            # self.edge_update = nn.ModuleList([mlp(node_in_dim + node_in_dim, mlp_dims, node_out_dim, act=act, dropout=dropout) for i in range(self.num_edges)])
            node_dims = [node_in_dim] + mlp_dims + [node_out_dim]
            self.node_update = []
            for i in range(len(node_dims)-1):
                if i == len(node_dims) - 2: layer_act = act # Follow TDMPC2, last layer use customized act
                else: layer_act = nn.Mish(inplace=False)
                if i == 0: layer_dropout = dropout
                else: layer_dropout = 0.
                self.node_update.append(VectorizedLinearLayer(self.num_nodes, node_dims[i], node_dims[i+1], 
                                                              use_layer_norm=True, act=layer_act, dropout=layer_dropout))
            self.node_update = nn.Sequential(*self.node_update)

            edge_dims = [node_in_dim*2] + mlp_dims + [node_out_dim]
            self.edge_update = []
            for i in range(len(edge_dims)-1):
                if i == len(edge_dims) - 2: layer_act = act # Follow TDMPC2, last layer use customized act
                else: layer_act = nn.Mish(inplace=False)
                if i == 0: layer_dropout = dropout
                else: layer_dropout = 0.
                self.edge_update.append(VectorizedLinearLayer(self.num_edges, edge_dims[i], edge_dims[i+1], 
                                                              use_layer_norm=True, act=layer_act, dropout=layer_dropout))
            self.edge_update = nn.Sequential(*self.edge_update)

        # Create fully connected graph edges
        edge_index = torch.combinations(torch.arange(self.num_nodes), r=2).T
        # edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Add reverse edges: [2, num_edges * 2] 
        self.edge_index = edge_index.to('cuda')
        # self.register_buffer('edge_index', edge_index)  # TODO: Use register_buffer to move to CUDA automatically


    def forward(self, node_features):
        """
        Args:
            - node_features: [..., num_nodes, feature_dim]
        """
        # Reshape inputs to 3D vectors
        raw_input_shape = node_features.shape # [..., num_nodes, input_dim]
        node_features = node_features.view(-1, raw_input_shape[-2], raw_input_shape[-1]).contiguous() # [B, num_nodes, input_dim]

        # Update node outputs
        if self.shared_parameters:
            node_outputs = self.node_update(node_features)
        else:
            node_outputs = node_features.transpose(1, 0) # [num_nodes, B, input_dim]
            node_outputs = self.node_update(node_outputs) # [num_nodes, B, node_dim]
            node_outputs = node_outputs.transpose(1, 0) # [B, num_nodes, node_dim]
        # Reshape outputs back to 4D/3D vectors
        node_outputs = node_outputs.view(*raw_input_shape[:-2], self.num_nodes, -1).contiguous() # [..., num_nodes, node_dim]

        # Compute edge features
        src, dest = self.edge_index
        edge_features = torch.cat([node_features[..., src, :], node_features[..., dest, :]], dim=-1) # [B, num_edges*2, input_dim]
        if self.shared_parameters: 
            edge_outputs = self.edge_update(edge_features)
        else:
            edge_outputs = edge_features.transpose(1, 0) # [num_edges*2, B, input_dim]
            edge_outputs = self.edge_update(edge_outputs) # [num_edges*2, B, node_dim]
            edge_outputs = edge_outputs.transpose(1, 0) # [B, num_edges*2, node_dim]
        # Average i->j and j->i
        # edge_outputs = torch.mean(edge_outputs.view(*edge_outputs.shape[:-2], self.num_edges, 2, -1), dim=-2) # [B, num_edges, input_dim]
        # Reshape outputs back to 4D/3D vectors
        edge_outputs = edge_outputs.view(*raw_input_shape[:-2], self.num_edges, -1).contiguous() # [..., num_edges, node_dim]
        return node_outputs, edge_outputs


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

        self.weight = torch.nn.Parameter(
            torch.empty(self._population_size, self._in_features, self._out_features),
            requires_grad=True,
        )
        self.bias = torch.nn.Parameter(
            torch.empty(self._population_size, 1, self._out_features),
            requires_grad=True,
        )

        for member_id in range(population_size):
            torch.nn.init.kaiming_uniform_(self.weight[member_id], a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)

        self._layer_norm = (
            torch.nn.LayerNorm(self._out_features, self._population_size)
            if use_layer_norm
            else None
        )

        # M: add activation and dropout
        self._act = act
        self._dropout = nn.Dropout(dropout, inplace=False) if dropout else None

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: [population_size, batch_size, in_features]
        Returns:
            - x: [population_size, batch_size, out_features]
        """
        assert x.shape[0] == self._population_size
        x = x.matmul(self.weight) + self.bias
        if self._layer_norm is not None:
            x = self._layer_norm(x)
        if self._dropout:
            x = self._dropout(x)
        if self._act:
            return self._act(x)
        else: 
            return x
    
    def __repr__(self):
        repr_dropout = f", dropout={self._dropout.p}" if self._dropout else ", dropout=None"
        repr_act = f"act={self._act.__class__.__name__})" if self._act else "act=None)"
        return f"VectorizedLinearLayer(in_features={self._in_features}, "\
            f"out_features={self._out_features}, "\
            f"bias={self.bias is not None}{repr_dropout}, "\
            f"{repr_act}"


class QMIXNet(nn.Module):
    """
    Inspired from QMIX
    """
    def __init__(self,
                num_agents,
                state_dim,
                num_layers=2,
                mixing_hidden_size=32):

        super(QMIXNet, self).__init__()

        self.num_agents = num_agents
        self.state_dim = state_dim
        self.mixing_hidden_size = mixing_hidden_size
        self.num_layers = num_layers

        # Generate mixing network
        if num_layers == 2:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.num_agents * self.mixing_hidden_size)
            self.hyper_b_1 = nn.Linear(self.state_dim, self.mixing_hidden_size)
            self.l2_dim = self.mixing_hidden_size
        elif num_layers == 1:
            self.l2_dim = self.num_agents
        self.hyper_w_2 = nn.Linear(self.state_dim, self.l2_dim)
        self.hyper_b_2 = nn.Sequential(nn.Linear(self.state_dim, self.mixing_hidden_size),
                               nn.ReLU(),
                               nn.Linear(self.mixing_hidden_size, 1))

    def forward(self, agent_qs, global_state):
        """
        The forward model takes the state to be given to the hyper-networks
        and agent observations as a single tensor (concatenation of
        agent local current observation, one-hot encoded last action, one-hot encoded agent_id)
        """
        state_shape = global_state.shape

        if self.num_layers == 2:
            # First layer 
            w1 = self.hyper_w_1(global_state)
            w1 = w1.view(*state_shape[:-1], self.num_agents, self.mixing_hidden_size)
            # w1 = torch.abs(w1)
            w1 = F.softmax(w1, dim=-1)
            b1 = self.hyper_b_1(global_state)
            b1 = b1.view(*state_shape[:-1], 1, self.mixing_hidden_size)
            q_tot = nn.functional.elu(torch.matmul(agent_qs.unsqueeze(-2), w1) + b1)
        else:
            q_tot = agent_qs.unsqueeze(-2)

        # Second layer 
        w2 = self.hyper_w_2(global_state)
        w2 = w2.view(*state_shape[:-1], self.l2_dim, 1)
        # w2 = torch.abs(w2)
        w2 = F.softmax(w2, dim=-2)
        b2 = self.hyper_b_2(global_state)
        b2 = b2.view(*state_shape[:-1], 1, 1)
        q_tot = torch.matmul(q_tot, w2) + b2
        return q_tot.squeeze(-2)