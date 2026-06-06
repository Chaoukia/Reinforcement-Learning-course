import numpy as np
import torch.nn as nn
from reinforcement_learning_course.deep_rl.reinforce_baseline.algorithms import ReinforceBaseline
from reinforcement_learning_course.deep_rl.reinforce_baseline.examples.neural_networks import (
    LunarLanderPolicyNetwork, LunarLanderValueNetwork,
    CartPolePolicyNetwork, CartPoleValueNetwork,
    MountainCarPolicyNetwork, MountainCarValueNetwork,
    AcrobotPolicyNetwork, AcrobotValueNetwork,
)
from gymnasium import Env


class LunarLanderReinforceBaseline(ReinforceBaseline):
    """ReinforceBaseline agent for the LunarLander environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 gamma: float = 0.99
                 ) -> None:
        """Initializes the LunarLanderReinforceBaseline agent.

        Args:
            env: Gymnasium LunarLander environment.
            gamma: Discount factor.
        """
        super().__init__(env, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Initializes the policy and value networks.

        Returns:
            A tuple containing:
                - policy_network: The policy network.
                - value_network: The value network.
        """

        policy_network = LunarLanderPolicyNetwork()
        value_network = LunarLanderValueNetwork()
        return policy_network, value_network


class CartPoleReinforceBaseline(ReinforceBaseline):
    """ReinforceBaseline agent for the CartPole environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 gamma: float = 0.99
                 ) -> None:
        """Initializes the CartPoleReinforceBaseline agent.

        Args:
            env: Gymnasium CartPole environment.
            gamma: Discount factor.
        """
        super().__init__(env, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Initializes the policy and value networks.

        Returns:
            A tuple containing:
                - policy_network: The policy network.
                - value_network: The value network.
        """

        policy_network = CartPolePolicyNetwork()
        value_network = CartPoleValueNetwork()
        return policy_network, value_network


class MountainCarReinforceBaseline(ReinforceBaseline):
    """ReinforceBaseline agent for the MountainCar environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 gamma: float = 0.99,
                 ) -> None:
        """Initializes the MountainCarReinforceBaseline agent.

        Args:
            env: Gymnasium MountainCar environment.
            gamma: Discount factor.
        """
        super().__init__(env, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Initializes the policy and value networks.

        Returns:
            A tuple containing:
                - policy_network: The policy network.
                - value_network: The value network.
        """

        policy_network = MountainCarPolicyNetwork()
        value_network = MountainCarValueNetwork()
        return policy_network, value_network


class AcrobotReinforceBaseline(ReinforceBaseline):
    """ReinforceBaseline agent for the Acrobot environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 gamma: float = 0.99,
                 ) -> None:
        """Initializes the AcrobotReinforceBaseline agent.

        Args:
            env: Gymnasium Acrobot environment.
            gamma: Discount factor.
        """
        super().__init__(env, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Initializes the policy and value networks.

        Returns:
            A tuple containing:
                - policy_netywork: The policy network.
                - value_network: The value network.
        """

        policy_netywork = AcrobotPolicyNetwork()
        value_network = AcrobotValueNetwork()
        return policy_netywork, value_network

