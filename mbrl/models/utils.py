from typing import Optional

import numpy as np
import torch
from torch import nn

from mbrl import torch_helpers


def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result


def unsorted_segment_mean(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    count = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    count.scatter_add_(0, segment_ids, torch.ones_like(tensor))
    return result / count.clamp(1.0)


def unsorted_segment_sum_ensemble(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    For ensembles, tensor: [e, nB*num_edges, hidden_dim]
    segment_ids: [nB*num_edges]
    """
    result_shape = (tensor.size(0), num_segments, tensor.size(-1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(0).unsqueeze(-1).expand(tensor.size(0), -1, tensor.size(-1))
    result.scatter_add_(1, segment_ids, tensor)
    return result


def unsorted_segment_mean_ensemble(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (tensor.size(0), num_segments, tensor.size(-1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    count = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    # Broadcast segment_ids [nB*num_edges] -> [ensemble_size, nB*num_edges, hidden_dim]
    segment_ids = segment_ids.unsqueeze(0).unsqueeze(-1).expand(tensor.size(0), -1, tensor.size(-1))
    result.scatter_add_(1, segment_ids, tensor)
    count.scatter_add_(1, segment_ids, torch.ones_like(tensor))
    return result / count.clamp(1.0)


def build_mlp(
    input_dim: int,
    output_dim: int,
    size: int,
    num_layers: int,
    activation: str,
    output_activation: Optional[str] = "none",
    layer_norm: bool = True,
):
    activation = torch_helpers.activation_from_string(activation)
    output_activation = torch_helpers.activation_from_string(output_activation)

    all_dims = [input_dim] + [size] * num_layers
    hidden_layers = []
    for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
        hidden_layers.append(nn.Linear(in_dim, out_dim))
        if layer_norm:
            hidden_layers.append(nn.LayerNorm(size))
        if activation is not None:
            hidden_layers.append(activation())
    layers = hidden_layers + [nn.Linear(all_dims[-1], output_dim)]
    if output_activation is not None:
        layers.append(output_activation())

    layers = nn.Sequential(*layers)
    return layers


class LayerNormEnsemble(nn.Module):
    def __init__(self, normal_shape, ensemble_size=1, elementwise_affine=True, epsilon=1e-5):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        """
        super().__init__()

        assert isinstance(normal_shape, int)

        param_shape = (ensemble_size, normal_shape)

        self.normal_shape = normal_shape
        self.param_shape = param_shape
        self.ensemble_size = ensemble_size
        self.epsilon = epsilon

        if elementwise_affine:
            self.gamma = nn.Parameter(torch.empty(param_shape, dtype=torch.float32))
            self.beta = nn.Parameter(torch.empty(param_shape, dtype=torch.float32))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.gamma is not None:
            self.gamma.data.fill_(1.0)
        if self.beta is not None:
            self.beta.data.zero_()

    def forward(self, x):

        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.square(x - mean).mean(dim=-1, keepdim=True)

        y = (x - mean) / torch.sqrt(var + self.epsilon)
        if self.gamma is not None:
            y = torch.mul(y, self.gamma.unsqueeze(1).expand_as(y))
        if self.beta is not None:
            y = y + self.beta.unsqueeze(1).expand_as(y)
        return y

    def extra_repr(self):
        return "normalization_dim={}, ensemble_size={}, gamma={}, beta={}, epsilon={}".format(
            self.normal_shape,
            self.ensemble_size,
            self.gamma is not None,
            self.beta is not None,
            self.epsilon,
        )


def connected_shuffle(list_arrays):
    random_state = np.random.RandomState(0)
    n_samples = list_arrays[0].shape[0]
    num_elements = np.array([a.shape[0] for a in list_arrays])
    if not np.all(num_elements == n_samples):
        raise ValueError("Different number of elements along axis 0", num_elements, n_samples)
    shuffling_indices = random_state.permutation(n_samples)
    list_arrays = [a[shuffling_indices] for a in list_arrays]
    return list_arrays
