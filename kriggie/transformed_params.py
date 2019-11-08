
import torch
from torch import nn
from torch.distributions import transforms


class TransformedParam(nn.Module):
    """
    Wraps a PyTorch parameter with an associated PyTorch Transform
    """
    def __init__(self, param, transform: transforms.Transform, requires_grad=True):
        super().__init__()
        self._unconstained = nn.Parameter(param, requires_grad=requires_grad)
        self._transform = transform

    @property
    def unconstrained(self):
        return self._unconstained

    @unconstrained.setter
    def unconstrained(self, val):
        assert isinstance(val, torch.Tensor) and not isinstance(val, nn.Parameter)
        self._unconstained = nn.Parameter(val)

    @property
    def constrained(self):
        return self._transform(self._unconstained)

    @constrained.setter
    def constrained(self, val):
        assert isinstance(val, torch.Tensor) and not isinstance(val, nn.Parameter)
        val_unconstrained = self._transform.inv(val.data)
        self.unconstrained = val_unconstrained

    def constrained_init_set(self, val):
        val_unconstrained = self._transform.inv(val)
        with torch.no_grad():
            self._unconstained[...] = val_unconstrained

    def forward(self):
        return self.constrained
