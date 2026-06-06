import numpy as np
import torch.nn as nn
from reinforcement_learning_course.deep_rl.reinforce.algorithms import Reinforce
from reinforcement_learning_course.deep_rl.reinforce.examples.neural_networks import (
    LunarLanderNetwork,
    CartPoleNetwork,
    MountainCarNetwork,
    AcrobotNetwork
)
from gymnasium import Env


class LunarLanderReinforce(Reinforce):
    """REINFORCE agent for the LunarLander environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 gamma: float = 0.99
                 ) -> None:
        """Initializes the LunarLanderReinforce agent.

        Args:
            env: Gymnasium LunarLander environment.
            gamma: Discount factor.
        """
        super().__init__(env, gamma)

    def make_networks(self) -> nn.Module:
        """Initializes the policy network.

        Returns:
            The LunarLander policy network.
        """

        policy_network = LunarLanderNetwork()
        return policy_network


class CartPoleReinforce(Reinforce):
    """REINFORCE agent for the CartPole environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 gamma: float = 0.99
                 ) -> None:
        """Initializes the CartPoleReinforce agent.

        Args:
            env: Gymnasium CartPole environment.
            gamma: Discount factor.
        """
        super().__init__(env, gamma)

    def make_networks(self) -> nn.Module:
        """Initializes the policy network.

        Returns:
            The CartPole policy network.
        """

        policy_network = CartPoleNetwork()
        return policy_network


class MountainCarReinforce(Reinforce):
    """REINFORCE agent for the MountainCar environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 gamma: float = 0.99,
                 ) -> None:
        """Initializes the MountainCarReinforce agent.

        Args:
            env: Gymnasium MountainCar environment.
            gamma: Discount factor.
        """
        super().__init__(env, gamma)

    def make_networks(self) -> nn.Module:
        """Initializes the policy network.

        Returns:
            The MountainCar policy network.
        """

        policy_network = MountainCarNetwork()
        return policy_network


class AcrobotReinforce(Reinforce):
    """REINFORCE agent for the Acrobot environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 gamma: float = 0.99,
                 ) -> None:
        """Initializes the AcrobotReinforce agent.

        Args:
            env: Gymnasium Acrobot environment.
            gamma: Discount factor.
        """
        super().__init__(env, gamma)

    def make_networks(self) -> nn.Module:
        """Initializes the policy network.

        Returns:
            The Acrobot policy network.
        """

        policy_netywork = AcrobotNetwork()
        return policy_netywork
