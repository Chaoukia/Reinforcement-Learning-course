import torch
import torch.nn as nn
from collections import OrderedDict


class LunarLanderNetwork(nn.Module):
    """LunarLander Deep Q-Network.

    Attributes:
        mlp: Sequential MLP mapping 8-dimensional state to 4 Q-values.
    """

    def __init__(self) -> None:
        """Initializes the LunarLander network architecture."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(8, 512)),
                                              ('tanh1', nn.Tanh()),
                                              ('linear2', nn.Linear(512, 256)),
                                              ('tanh2', nn.Tanh()),
                                              ('linear3', nn.Linear(256, 4))]))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Runs a forward propagation through the network.

        Args:
            x: Input state tensor.

        Returns:
            Output Q-values for the input state.
        """

        return self.mlp(x)


class CartPoleNetwork(nn.Module):
    """CartPole Deep Q-Network.

    Attributes:
        mlp: Sequential MLP mapping 4-dimensional state to 2 Q-values.
    """

    def __init__(self):
        """Initializes the CartPole network architecture."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(4, 64)),
                                              ('tanh1', nn.Tanh()),
                                              ('linear2', nn.Linear(64, 64)),
                                              ('tanh2', nn.Tanh()),
                                              ('linear3', nn.Linear(64, 2))]))

    def forward(self, x):
        """Runs a forward propagation through the network.

        Args:
            x: Input state tensor.

        Returns:
            Output Q-values for the input state.
        """

        return self.mlp(x)


class MountainCarNetwork(nn.Module):
    """MountainCar Deep Q-Network.

    Attributes:
        mlp: Sequential MLP mapping 2-dimensional state to 3 Q-values.
    """

    def __init__(self):
        """Initializes the MountainCar network architecture."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(2, 64)),
                                              ('tanh1', nn.Tanh()),
                                              ('linear2', nn.Linear(64, 64)),
                                              ('tanh2', nn.Tanh()),
                                              ('linear3', nn.Linear(64, 3))]))

    def forward(self, x):
        """Runs a forward propagation through the network.

        Args:
            x: Input state tensor.

        Returns:
            Output Q-values for the input state.
        """

        return self.mlp(x)


class AcrobotNetwork(nn.Module):
    """Acrobot Deep Q-Network.

    Attributes:
        mlp: Sequential MLP mapping 6-dimensional state to 3 Q-values.
    """

    def __init__(self):
        """Initializes the Acrobot network architecture."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(6, 64)),
                                              ('tanh1', nn.Tanh()),
                                              ('linear2', nn.Linear(64, 64)),
                                              ('tanh2', nn.Tanh()),
                                              ('linear3', nn.Linear(64, 3))]))

    def forward(self, x):
        """Runs a forward propagation through the network.

        Args:
            x: Input state tensor.

        Returns:
            Output Q-values for the input state.
        """

        return self.mlp(x)
