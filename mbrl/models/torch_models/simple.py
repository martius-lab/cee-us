from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F

from mbrl import torch_helpers
from mbrl.torch_helpers import (
    SimpleNormal,
    activation_from_string,
    initializer_from_string,
    input_to_tensors,
    output_to_numpy,
)

# specific
# PYTORCH

TensorType = Union[torch.Tensor, np.ndarray]


def model_from_string(mdl_str):
    mdl_dict = {
        "mlp": Mlp,
        "gaussian": GaussianNN,
    }
    if mdl_str in mdl_dict:
        return mdl_dict[mdl_str]
    else:
        raise NotImplementedError("Add model class {} to dictionary".format(mdl_str))


class NNModel(ABC):
    @abstractmethod
    def forward(self, obs):
        """forward pass of the computation graph"""
        pass


class DeterministicNNModel(NNModel, ABC):
    is_stochastic = False


class StochasticNNModel(NNModel, ABC):
    is_stochastic = True


class Mlp(DeterministicNNModel, nn.Module):
    def __init__(self, *, input_dim, output_dim, params, **kwargs):

        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = None
        self._parse_params(**params)

        self.build()

        self.initialize()

    def initialize(self):
        self.layers.apply(initializer_from_string(self.weight_initializer, self.bias_initializer))

    def _parse_params(
        self,
        num_layers,
        size,
        activation,
        output_activation,
        l1_reg,
        l2_reg,
        weight_initializer,
        bias_initializer,
    ):
        self.num_layers = num_layers
        self.size = size
        self.activation = activation_from_string(activation)
        self.output_activation = activation_from_string(output_activation)
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

    def build(self):

        all_dims = [self.input_dim] + [self.size] * self.num_layers
        hidden_layers = []
        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            hidden_layers.append(nn.Linear(in_dim, out_dim))
            if self.activation is not None:
                hidden_layers.append(self.activation())
        layers = hidden_layers + [nn.Linear(all_dims[-1], self.output_dim)]
        if self.output_activation is not None:
            layers.append(self.output_activation())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(torch_helpers.device)
        return self.layers(x)

    @input_to_tensors
    @output_to_numpy
    def predict(self, inputs):
        return self.forward(inputs)

    @property
    def L1_loss(self):
        l1_loss = 0
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                l1_loss += layer.weight.norm(1)

        return l1_loss * self.l1_reg

    @property
    def L2_loss(self):
        l2_loss = 0
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                l2_loss += layer.weight.norm(2)

        return l2_loss * self.l2_reg


class GaussianNN(StochasticNNModel, nn.Module):
    def __init__(self, *, input_dim, output_dim, params, **kwargs):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim * 2  # mean + std

        self._parse_params(**params)

        self.base_model = model_from_string(self.base_model_spec)(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            params=self.base_model_params,
        )

        self.logvar = None
        if not self.state_dependent_var:
            self.logvar = nn.Parameter(torch.zeros(self.output_dim // 2))

        self.max_logvar = nn.Parameter(
            torch.ones(1, self.output_dim // 2, dtype=torch.float32) * self.var_clipping_high,
            requires_grad=self.regularize_automatic_var_scaling,
        )
        self.min_logvar = nn.Parameter(
            torch.ones(1, self.output_dim // 2, dtype=torch.float32) * self.var_clipping_low,
            requires_grad=self.regularize_automatic_var_scaling,
        )

        self.dist = None

    def _parse_params(
        self,
        base_model,
        base_model_params,
        state_dependent_var,
        var_clipping_low,
        var_clipping_high,
        distribution,
        regularize_automatic_var_scaling,
    ):
        self.base_model_spec = base_model
        self.base_model_params = base_model_params
        self.state_dependent_var = state_dependent_var
        self.var_clipping_low = var_clipping_low
        self.var_clipping_high = var_clipping_high
        self.distribution = distribution
        self.regularize_automatic_var_scaling = regularize_automatic_var_scaling

    def forward(self, x):
        base_model_predictions = self.base_model.forward(x)
        self._mean = base_model_predictions[..., : self.output_dim // 2]
        logvar = self.logvar
        if logvar is None:
            logvar = base_model_predictions[..., self.output_dim // 2 :]
            # logvar = torch.clamp(
            #     logvar, -10, 4
            # )
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if self.distribution == "MultivariateNormal":
            self.dist = MultivariateNormal(
                self._mean,
                scale_tril=torch.diag_embed(torch.sqrt(logvar.exp_()), offset=0, dim1=-2, dim2=-1),
            )
        elif self.distribution == "Normal":
            self.dist = Normal(self._mean, torch.sqrt(logvar.exp_()))
        elif self.distribution == "none":
            self.dist = SimpleNormal(self._mean, logvar)
        else:
            raise NotImplementedError(f"Distribution {self.distribution} not supported in GaussianNN policy")
        # TODO use full covariance for CEM policy

        return self.dist

    def logstd_reg_losses(self):
        loss = 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)

        return loss

    def L1_losses(self):
        return self.base_model.L1_losses()

    def L2_losses(self):
        return self.base_model.L2_losses()
