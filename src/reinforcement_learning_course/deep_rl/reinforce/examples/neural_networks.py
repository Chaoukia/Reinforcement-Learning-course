import torch
import torch.nn as nn
from collections import OrderedDict


class LunarLanderNetwork(nn.Module):
    """LunarLander REINFORCE policy network."""

    def __init__(self) -> None:
        """Initializes LunarLanderNetwork with a multi-layer perceptron."""
        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(8, 512)),
                                              ('tanh1', nn.Tanh()),
                                              ('linear2', nn.Linear(512, 256)),
                                              ('tanh2', nn.Tanh()),
                                              ('linear3', nn.Linear(256, 4))]))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Runs a forward propagation.

        Args:
            x: Input state tensor.

        Returns:
            Output action logits for the state.
        """
        return self.mlp(x)


class CartPoleNetwork(nn.Module):
    """CartPole REINFORCE policy network."""

    def __init__(self):
        """Initializes CartPoleNetwork with a multi-layer perceptron."""
        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(4, 64)),
                                              ('tanh1', nn.Tanh()),
                                              ('linear2', nn.Linear(64, 64)),
                                              ('tanh2', nn.Tanh()),
                                              ('linear3', nn.Linear(64, 2))]))

    def forward(self, x):
        """Runs a forward propagation.

        Args:
            x: Input state tensor.

        Returns:
            Output action logits for the state.
        """
        return self.mlp(x)


class MountainCarNetwork(nn.Module):
    """MountainCar REINFORCE policy network."""

    def __init__(self):
        """Initializes MountainCarNetwork with a multi-layer perceptron."""
        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(2, 64)),
                                              ('tanh1', nn.Tanh()),
                                              ('linear2', nn.Linear(64, 64)),
                                              ('tanh2', nn.Tanh()),
                                              ('linear3', nn.Linear(64, 3))]))

    def forward(self, x):
        """Runs a forward propagation.

        Args:
            x: Input state tensor.

        Returns:
            Output action logits for the state.
        """
        return self.mlp(x)


class AcrobotNetwork(nn.Module):
    """Acrobot REINFORCE policy network."""

    def __init__(self):
        """Initializes AcrobotNetwork with a multi-layer perceptron."""
        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(6, 64)),
                                              ('tanh1', nn.Tanh()),
                                              ('linear2', nn.Linear(64, 64)),
                                              ('tanh2', nn.Tanh()),
                                              ('linear3', nn.Linear(64, 3))]))

    def forward(self, x):
        """Runs a forward propagation.

        Args:
            x: Input state tensor.

        Returns:
            Output action logits for the state.
        """
        return self.mlp(x)
