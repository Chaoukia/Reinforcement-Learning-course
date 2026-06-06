import torch
import torch.nn as nn
from collections import OrderedDict


class LunarLanderPolicyNetwork(nn.Module):
    """LunarLander policy network mapping states to action logits.

    Attributes:
        mlp: Sequential MLP with three linear layers and ReLU activations.
    """

    def __init__(self) -> None:
        """Initializes the LunarLanderPolicyNetwork."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(8, 512)),
                                              ('relu1', nn.ReLU()),
                                              ('linear2', nn.Linear(512, 256)),
                                              ('relu2', nn.ReLU()),
                                              ('linear3', nn.Linear(256, 4))]))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Runs a forward pass through the policy network.

        Args:
            x: Input state tensor.

        Returns:
            Output action logits for the given state.
        """

        return self.mlp(x)


class LunarLanderValueNetwork(nn.Module):
    """LunarLander value network mapping states to scalar values.

    Attributes:
        mlp: Sequential MLP with three linear layers and ReLU activations.
    """

    def __init__(self) -> None:
        """Initializes the LunarLanderValueNetwork."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(8, 512)),
                                              ('relu1', nn.ReLU()),
                                              ('linear2', nn.Linear(512, 256)),
                                              ('relu2', nn.ReLU()),
                                              ('linear3', nn.Linear(256, 1))]))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Runs a forward pass through the value network.

        Args:
            x: Input state tensor.

        Returns:
            Estimated scalar value for the given state.
        """

        return self.mlp(x)


class CartPolePolicyNetwork(nn.Module):
    """CartPole policy network mapping states to action logits.

    Attributes:
        mlp: Sequential MLP with three linear layers and ReLU activations.
    """

    def __init__(self):
        """Initializes the CartPolePolicyNetwork."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(4, 512)),
                                              ('relu1', nn.ReLU()),
                                              ('linear2', nn.Linear(512, 256)),
                                              ('relu2', nn.ReLU()),
                                              ('linear3', nn.Linear(256, 2))]))

    def forward(self, x):
        """Runs a forward pass through the policy network.

        Args:
            x: Input state tensor.

        Returns:
            Output action logits for the given state.
        """

        return self.mlp(x)


class CartPoleValueNetwork(nn.Module):
    """CartPole value network mapping states to scalar values.

    Attributes:
        mlp: Sequential MLP with three linear layers and ReLU activations.
    """

    def __init__(self):
        """Initializes the CartPoleValueNetwork."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(4, 512)),
                                              ('relu1', nn.ReLU()),
                                              ('linear2', nn.Linear(512, 256)),
                                              ('relu2', nn.ReLU()),
                                              ('linear3', nn.Linear(256, 1))]))

    def forward(self, x):
        """Runs a forward pass through the value network.

        Args:
            x: Input state tensor.

        Returns:
            Estimated scalar value for the given state.
        """

        return self.mlp(x)


class MountainCarPolicyNetwork(nn.Module):
    """MountainCar policy network mapping states to action logits.

    Attributes:
        mlp: Sequential MLP with three linear layers and ReLU activations.
    """

    def __init__(self):
        """Initializes the MountainCarPolicyNetwork."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(2, 512)),
                                              ('relu1', nn.ReLU()),
                                              ('linear2', nn.Linear(512, 256)),
                                              ('relu2', nn.ReLU()),
                                              ('linear3', nn.Linear(256, 3))]))

    def forward(self, x):
        """Runs a forward pass through the policy network.

        Args:
            x: Input state tensor.

        Returns:
            Output action logits for the given state.
        """

        return self.mlp(x)


class MountainCarValueNetwork(nn.Module):
    """MountainCar value network mapping states to scalar values.

    Attributes:
        mlp: Sequential MLP with three linear layers and ReLU activations.
    """

    def __init__(self):
        """Initializes the MountainCarValueNetwork."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(2, 512)),
                                              ('relu1', nn.ReLU()),
                                              ('linear2', nn.Linear(512, 256)),
                                              ('relu2', nn.ReLU()),
                                              ('linear3', nn.Linear(256, 1))]))

    def forward(self, x):
        """Runs a forward pass through the value network.

        Args:
            x: Input state tensor.

        Returns:
            Estimated scalar value for the given state.
        """

        return self.mlp(x)


class AcrobotPolicyNetwork(nn.Module):
    """Acrobot policy network mapping states to action logits.

    Attributes:
        mlp: Sequential MLP with three linear layers and ReLU activations.
    """

    def __init__(self):
        """Initializes the AcrobotPolicyNetwork."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(6, 512)),
                                              ('relu1', nn.ReLU()),
                                              ('linear2', nn.Linear(512, 256)),
                                              ('relu2', nn.ReLU()),
                                              ('linear3', nn.Linear(256, 3))]))

    def forward(self, x):
        """Runs a forward pass through the policy network.

        Args:
            x: Input state tensor.

        Returns:
            Output action logits for the given state.
        """

        return self.mlp(x)


class AcrobotValueNetwork(nn.Module):
    """Acrobot value network mapping states to scalar values.

    Attributes:
        mlp: Sequential MLP with three linear layers and ReLU activations.
    """

    def __init__(self):
        """Initializes the AcrobotValueNetwork."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(6, 512)),
                                              ('relu1', nn.ReLU()),
                                              ('linear2', nn.Linear(512, 256)),
                                              ('relu2', nn.ReLU()),
                                              ('linear3', nn.Linear(256, 1))]))

    def forward(self, x):
        """Runs a forward pass through the value network.

        Args:
            x: Input state tensor.

        Returns:
            Estimated scalar value for the given state.
        """

        return self.mlp(x)
