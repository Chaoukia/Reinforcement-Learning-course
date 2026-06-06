import torch
import torch.nn as nn
from collections import OrderedDict


class LunarLanderPolicyNetwork(nn.Module):
    """Policy network for the LunarLander environment.

    Attributes:
        mlp: Sequential MLP with two hidden layers of size 64 and Tanh activations,
            outputting logits for 4 discrete actions.
    """

    def __init__(self) -> None:
        """Initializes LunarLanderPolicyNetwork."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(8, 64)),
                                              ('relu1', nn.Tanh()),
                                              ('linear2', nn.Linear(64, 64)),
                                              ('relu2', nn.Tanh()),
                                              ('linear3', nn.Linear(64, 4))]))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Runs a forward propagation through the policy network.

        Args:
            x: Input state tensor.

        Returns:
            Output logits for each discrete action.
        """

        return self.mlp(x)


class LunarLanderValueNetwork(nn.Module):
    """Value network for the LunarLander environment.

    Attributes:
        mlp: Sequential MLP with two hidden layers of size 64 and Tanh activations,
            outputting a scalar state value.
    """

    def __init__(self) -> None:
        """Initializes LunarLanderValueNetwork."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(8, 64)),
                                              ('relu1', nn.Tanh()),
                                              ('linear2', nn.Linear(64, 64)),
                                              ('relu2', nn.Tanh()),
                                              ('linear3', nn.Linear(64, 1))]))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Runs a forward propagation through the value network.

        Args:
            x: Input state tensor.

        Returns:
            Scalar state value estimate.
        """

        return self.mlp(x)


class CartPolePolicyNetwork(nn.Module):
    """Policy network for the CartPole environment.

    Attributes:
        mlp: Sequential MLP with two hidden layers of size 64 and Tanh activations,
            outputting logits for 2 discrete actions.
    """

    def __init__(self):
        """Initializes CartPolePolicyNetwork."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(4, 64)),
                                              ('relu1', nn.Tanh()),
                                              ('linear2', nn.Linear(64, 64)),
                                              ('relu2', nn.Tanh()),
                                              ('linear3', nn.Linear(64, 2))]))

    def forward(self, x):
        """Runs a forward propagation through the policy network.

        Args:
            x: Input state tensor.

        Returns:
            Output logits for each discrete action.
        """

        return self.mlp(x)


class CartPoleValueNetwork(nn.Module):
    """Value network for the CartPole environment.

    Attributes:
        mlp: Sequential MLP with two hidden layers of size 64 and Tanh activations,
            outputting a scalar state value.
    """

    def __init__(self):
        """Initializes CartPoleValueNetwork."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(4, 64)),
                                              ('relu1', nn.Tanh()),
                                              ('linear2', nn.Linear(64, 64)),
                                              ('relu2', nn.Tanh()),
                                              ('linear3', nn.Linear(64, 1))]))

    def forward(self, x):
        """Runs a forward propagation through the value network.

        Args:
            x: Input state tensor.

        Returns:
            Scalar state value estimate.
        """

        return self.mlp(x)


class MountainCarPolicyNetwork(nn.Module):
    """Policy network for the MountainCar environment.

    Attributes:
        mlp: Sequential MLP with hidden layers of size 512 and 256 and Tanh activations,
            outputting logits for 3 discrete actions.
    """

    def __init__(self):
        """Initializes MountainCarPolicyNetwork."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(2, 512)),
                                              ('relu1', nn.Tanh()),
                                              ('linear2', nn.Linear(512, 256)),
                                              ('relu2', nn.Tanh()),
                                              ('linear3', nn.Linear(256, 3))]))

    def forward(self, x):
        """Runs a forward propagation through the policy network.

        Args:
            x: Input state tensor.

        Returns:
            Output logits for each discrete action.
        """

        return self.mlp(x)


class MountainCarValueNetwork(nn.Module):
    """Value network for the MountainCar environment.

    Attributes:
        mlp: Sequential MLP with hidden layers of size 512 and 256 and Tanh activations,
            outputting a scalar state value.
    """

    def __init__(self):
        """Initializes MountainCarValueNetwork."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(2, 512)),
                                              ('relu1', nn.Tanh()),
                                              ('linear2', nn.Linear(512, 256)),
                                              ('relu2', nn.Tanh()),
                                              ('linear3', nn.Linear(256, 1))]))

    def forward(self, x):
        """Runs a forward propagation through the value network.

        Args:
            x: Input state tensor.

        Returns:
            Scalar state value estimate.
        """

        return self.mlp(x)


class AcrobotPolicyNetwork(nn.Module):
    """Policy network for the Acrobot environment.

    Attributes:
        mlp: Sequential MLP with hidden layers of size 512 and 256 and Tanh activations,
            outputting logits for 3 discrete actions.
    """

    def __init__(self):
        """Initializes AcrobotPolicyNetwork."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(6, 512)),
                                              ('relu1', nn.Tanh()),
                                              ('linear2', nn.Linear(512, 256)),
                                              ('relu2', nn.Tanh()),
                                              ('linear3', nn.Linear(256, 3))]))

    def forward(self, x):
        """Runs a forward propagation through the policy network.

        Args:
            x: Input state tensor.

        Returns:
            Output logits for each discrete action.
        """

        return self.mlp(x)


class AcrobotValueNetwork(nn.Module):
    """Value network for the Acrobot environment.

    Attributes:
        mlp: Sequential MLP with hidden layers of size 512 and 256 and Tanh activations,
            outputting a scalar state value.
    """

    def __init__(self):
        """Initializes AcrobotValueNetwork."""

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(6, 512)),
                                              ('relu1', nn.Tanh()),
                                              ('linear2', nn.Linear(512, 256)),
                                              ('relu2', nn.Tanh()),
                                              ('linear3', nn.Linear(256, 1))]))

    def forward(self, x):
        """Runs a forward propagation through the value network.

        Args:
            x: Input state tensor.

        Returns:
            Scalar state value estimate.
        """

        return self.mlp(x)
