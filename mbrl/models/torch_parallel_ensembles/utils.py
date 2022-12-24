import torch

from mbrl import torch_helpers


class MLPParallelEnsembleLayer(torch.nn.Module):
    def __init__(self, ensemble_size, in_features, out_features, bias=True):
        super(MLPParallelEnsembleLayer, self).__init__()

        self.ensemble_size = ensemble_size
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.parameter.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch_helpers.torch_truncated_normal_initializer(self.weight)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        We need it that way, because otherwise the tensorRT model doesn't work. E.g. it expects a
        certain shape of the bias vector
        """

        return ((input @ self.weight).transpose(0, 1) + self.bias[None, ...]).transpose(0, 1)

    def extra_repr(self) -> str:
        return "ensemble_size={}, in_features={}, out_features={}, bias={}".format(
            self.ensemble_size,
            self.in_features,
            self.out_features,
            self.bias is not None,
        )
