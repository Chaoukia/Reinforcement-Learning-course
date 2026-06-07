import torch
import torch.nn as nn
from collections import OrderedDict


class LunarLanderPolicyNetwork(nn.Module):
    """Policy network for the LunarLander environment.

    Attributes:
        mlp: Sequential MLP mapping 8-dimensional state to 4 action logits.
    """

    def __init__(self) -> None:
        """Initializes LunarLanderPolicyNetwork."""

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
    """Value network for the LunarLander environment.

    Attributes:
        mlp: Sequential MLP mapping 8-dimensional state to a scalar value.
    """

    def __init__(self) -> None:
        """Initializes LunarLanderValueNetwork."""

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
            Estimated state value for the given state.
        """

        return self.mlp(x)


class CartPolePolicyNetwork(nn.Module):
    """Policy network for the CartPole environment.

    Attributes:
        mlp: Sequential MLP mapping 4-dimensional state to 2 action logits.
    """

    def __init__(self):
        """Initializes CartPolePolicyNetwork."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(4, 64)),
                                              ('tanh1', nn.Tanh()),
                                              ('linear2', nn.Linear(64, 64)),
                                              ('tanh2', nn.Tanh()),
                                              ('linear3', nn.Linear(64, 2))]))

    def forward(self, x):
        """Runs a forward pass through the policy network.

        Args:
            x: Input state tensor.

        Returns:
            Output action logits for the given state.
        """

        return self.mlp(x)


class CartPoleValueNetwork(nn.Module):
    """Value network for the CartPole environment.

    Attributes:
        mlp: Sequential MLP mapping 4-dimensional state to a scalar value.
    """

    def __init__(self):
        """Initializes CartPoleValueNetwork."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(4, 64)),
                                              ('tanh1', nn.Tanh()),
                                              ('linear2', nn.Linear(64, 64)),
                                              ('tanh2', nn.Tanh()),
                                              ('linear3', nn.Linear(64, 1))]))

    def forward(self, x):
        """Runs a forward pass through the value network.

        Args:
            x: Input state tensor.

        Returns:
            Estimated state value for the given state.
        """

        return self.mlp(x)


class MountainCarPolicyNetwork(nn.Module):
    """Policy network for the MountainCar environment.

    Attributes:
        mlp: Sequential MLP mapping 2-dimensional state to 3 action logits.
    """

    def __init__(self):
        """Initializes MountainCarPolicyNetwork."""

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
    """Value network for the MountainCar environment.

    Attributes:
        mlp: Sequential MLP mapping 2-dimensional state to a scalar value.
    """

    def __init__(self):
        """Initializes MountainCarValueNetwork."""

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
            Estimated state value for the given state.
        """

        return self.mlp(x)


class AcrobotPolicyNetwork(nn.Module):
    """Policy network for the Acrobot environment.

    Attributes:
        mlp: Sequential MLP mapping 6-dimensional state to 3 action logits.
    """

    def __init__(self):
        """Initializes AcrobotPolicyNetwork."""

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
    """Value network for the Acrobot environment.

    Attributes:
        mlp: Sequential MLP mapping 6-dimensional state to a scalar value.
    """

    def __init__(self):
        """Initializes AcrobotValueNetwork."""

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
            Estimated state value for the given state.
        """

        return self.mlp(x)
