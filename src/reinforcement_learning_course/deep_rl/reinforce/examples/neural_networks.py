import torch
import torch.nn as nn
from collections import OrderedDict


class LunarLanderNetwork(nn.Module):
    """LunarLander Deep Q-Network."""

    def __init__(self) -> None:
        """Initializes LunarLanderNetwork with a multi-layer perceptron."""
        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(8, 512)),
                                              ('relu1', nn.ReLU()),
                                              ('linear2', nn.Linear(512, 256)),
                                              ('relu2', nn.ReLU()),
                                              ('linear3', nn.Linear(256, 4))]))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Runs a forward propagation.

        Args:
            x: Input state tensor.

        Returns:
            Output q-values for the state.
        """
        return self.mlp(x)


class CartPoleNetwork(nn.Module):
    """CartPole Deep Q-Network."""

    def __init__(self):
        """Initializes CartPoleNetwork with a multi-layer perceptron."""
        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(4, 512)),
                                              ('relu1', nn.ReLU()),
                                              ('linear2', nn.Linear(512, 256)),
                                              ('relu2', nn.ReLU()),
                                              ('linear3', nn.Linear(256, 2))]))

    def forward(self, x):
        """Runs a forward propagation.

        Args:
            x: Input state tensor.

        Returns:
            Output q-values for the state.
        """
        return self.mlp(x)


class MountainCarNetwork(nn.Module):
    """MountainCar Deep Q-Network."""

    def __init__(self):
        """Initializes MountainCarNetwork with a multi-layer perceptron."""
        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(2, 512)),
                                              ('relu1', nn.ReLU()),
                                              ('linear2', nn.Linear(512, 256)),
                                              ('relu2', nn.ReLU()),
                                              ('linear3', nn.Linear(256, 3))]))

    def forward(self, x):
        """Runs a forward propagation.

        Args:
            x: Input state tensor.

        Returns:
            Output q-values for the state.
        """
        return self.mlp(x)


class AcrobotNetwork(nn.Module):
    """Acrobot Deep Q-Network."""

    def __init__(self):
        """Initializes AcrobotNetwork with a multi-layer perceptron."""
        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(6, 512)),
                                              ('relu1', nn.ReLU()),
                                              ('linear2', nn.Linear(512, 256)),
                                              ('relu2', nn.ReLU()),
                                              ('linear3', nn.Linear(256, 3))]))

    def forward(self, x):
        """Runs a forward propagation.

        Args:
            x: Input state tensor.

        Returns:
            Output q-values for the state.
        """
        return self.mlp(x)
