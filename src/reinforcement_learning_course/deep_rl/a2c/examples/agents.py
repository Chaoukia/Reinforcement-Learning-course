import numpy as np
import torch.nn as nn
import gymnasium as gym
from reinforcement_learning_course.deep_rl.a2c.algorithms import A2CWorker
from reinforcement_learning_course.deep_rl.a2c.examples.neural_networks import (
    LunarLanderPolicyNetwork, LunarLanderValueNetwork,
    CartPolePolicyNetwork, CartPoleValueNetwork,
    MountainCarPolicyNetwork, MountainCarValueNetwork,
    AcrobotPolicyNetwork, AcrobotValueNetwork,
)
from gymnasium import Env


class LunarLanderA2C(A2CWorker):
    """A2CWorker agent for the LunarLander environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 n_workers: int,
                 gamma: float = 0.99
                 ) -> None:
        """Initialize the LunarLanderA2C agent.

        Args:
            env: Gymnasium LunarLander environment.
            n_workers: Number of parallel workers.
            gamma: Discount factor.
        """
        super().__init__(env, n_workers, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Initialize the policy and value networks.

        Returns:
            A tuple of:
                - policy_network: The policy network.
                - value_network: The value network.
        """

        policy_network = LunarLanderPolicyNetwork()
        value_network = LunarLanderValueNetwork()
        return policy_network, value_network


class CartPoleA2C(A2CWorker):
    """A2CWorker agent for the CartPole environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 n_workers: int,
                 gamma: float = 0.99
                 ) -> None:
        """Initialize the CartPoleA2C agent.

        Args:
            env: Gymnasium CartPole environment.
            n_workers: Number of parallel workers.
            gamma: Discount factor.
        """
        super().__init__(env, n_workers, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Initialize the policy and value networks.

        Returns:
            A tuple of:
                - policy_network: The policy network.
                - value_network: The value network.
        """

        policy_network = CartPolePolicyNetwork()
        value_network = CartPoleValueNetwork()
        return policy_network, value_network


class MountainCarA2C(A2CWorker):
    """A2CWorker agent for the MountainCar environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 n_workers: int,
                 gamma: float = 0.99,
                 ) -> None:
        """Initialize the MountainCarA2C agent.

        Args:
            env: Gymnasium MountainCar environment.
            n_workers: Number of parallel workers.
            gamma: Discount factor.
        """
        super().__init__(env, n_workers, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Initialize the policy and value networks.

        Returns:
            A tuple of:
                - policy_network: The policy network.
                - value_network: The value network.
        """

        policy_network = MountainCarPolicyNetwork()
        value_network = MountainCarValueNetwork()
        return policy_network, value_network


class AcrobotA2C(A2CWorker):
    """A2CWorker agent for the Acrobot environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 n_workers: int,
                 gamma: float = 0.99,
                 ) -> None:
        """Initialize the AcrobotA2C agent.

        Args:
            env: Gymnasium Acrobot environment.
            n_workers: Number of parallel workers.
            gamma: Discount factor.
        """
        super().__init__(env, n_workers, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Initialize the policy and value networks.

        Returns:
            A tuple of:
                - policy_network: The policy network.
                - value_network: The value network.
        """

        policy_netywork = AcrobotPolicyNetwork()
        value_network = AcrobotValueNetwork()
        return policy_netywork, value_network

