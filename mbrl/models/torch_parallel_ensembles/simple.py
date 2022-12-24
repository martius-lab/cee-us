import numpy as np
import torch
from torch import nn

from mbrl import torch_helpers
from mbrl.models.abstract_models import (
    EnsembleModel,
    StochasticModel,
    TorchModel,
    TrainableModel,
)
from mbrl.models.spectral_normalization import ensemble_spectral_norm
from mbrl.models.utils import LayerNormEnsemble

from .utils import MLPParallelEnsembleLayer


class MLPParallelEnsemble(torch.nn.Module):
    def __init__(self, input_dim, output_dim, params, **kwargs):
        """ "
        :models should be a list of torch.Sequential
        """
        super().__init__()

        self._parse_params(**params)

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.build()
        self.initialize()

    def initialize(self):
        self.layers.apply(torch_helpers.initializer_from_string(self.weight_initializer, self.bias_initializer))

    def build(self):

        all_dims = [self.input_dim] + [self.size] * self.num_layers
        hidden_layers = []

        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            layer = [MLPParallelEnsembleLayer(self.n, in_dim, out_dim)]
            if self.use_spectral_normalization:
                layer[-1] = ensemble_spectral_norm(layer[-1])
            if self.use_layer_normalization:
                layer.append(LayerNormEnsemble(out_dim, self.n))
            if self.activation is not None:
                layer.append(self.activation())
            hidden_layers.append(nn.Sequential(*layer))

        layer = [MLPParallelEnsembleLayer(self.n, all_dims[-1], self.output_dim)]
        if self.output_activation is not None:
            layer.append(self.output_activation())

        layers = hidden_layers + [nn.Sequential(*layer)]

        self.layers = nn.Sequential(*layers)

    def _parse_params(
        self,
        n,
        num_layers,
        size,
        activation,
        output_activation,
        l1_reg,
        l2_reg,
        weight_initializer,
        bias_initializer,
        use_spectral_normalization,
        use_layer_normalization,
    ):
        self.n = n
        self.num_layers = num_layers
        self.size = size
        self.activation = torch_helpers.activation_from_string(activation)
        self.output_activation = torch_helpers.activation_from_string(output_activation)
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.use_spectral_normalization = use_spectral_normalization
        self.use_layer_normalization = use_layer_normalization

    def forward(self, x):
        x = x.to(torch_helpers.device)
        return self.layers(x)

    def L1_losses(self):
        l1_losses = 0
        for layer in self.layers:
            l1_losses += torch.sum(layer[0].weight.norm(1, dim=(1, 2)))

        return l1_losses * self.l1_reg

    def L2_losses(self):
        l2_losses = 0
        for layer in self.layers:
            l2_losses += torch.sum(layer[0].weight.norm(2, dim=(1, 2)))

        return l2_losses * self.l2_reg


class SimpleStochasticEnsembleModel(StochasticModel, EnsembleModel, TrainableModel, TorchModel):
    """
    This model emulates an ensemble to estimate the uncertainty (epistemic + aleatoric),
    whereas it's in fact a simple Gaussian distribution outputting model.
    """

    def __init__(self, *, env, **params):
        super().__init__(env)
        self.input_dim = self.env.observation_space_size_preproc + self.env.action_space.shape[0]
        self.output_dim = self.env.targ_proc(
            np.zeros(self.env.observation_space.shape[0]),
            np.zeros(self.env.observation_space.shape[0]),
        ).shape[0]

        self._parse_params(**params)

    def _parse_params(self, ensemble_size, stochastic_nn_params):
        self.ensemble_size = ensemble_size
        self.stochastic_nn_params = stochastic_nn_params
