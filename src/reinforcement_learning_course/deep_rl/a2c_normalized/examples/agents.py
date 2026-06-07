import numpy as np
import torch.nn as nn
import gymnasium as gym
from reinforcement_learning_course.deep_rl.a2c_normalized.algorithms import A2CWorker
from reinforcement_learning_course.deep_rl.a2c_normalized.examples.neural_networks import (
    LunarLanderPolicyNetwork, LunarLanderValueNetwork,
    CartPolePolicyNetwork, CartPoleValueNetwork,
    MountainCarPolicyNetwork, MountainCarValueNetwork,
    AcrobotPolicyNetwork, AcrobotValueNetwork,
)
from gymnasium import Env


class LunarLanderA2C(A2CWorker):
    """A2CWorker LunarLander agent."""

    def __init__(self,
                 env: Env[np.array, int],
                 worker_id: int,
                 n_workers: int,
                 gamma: float = 0.99
                 ) -> None:
        """Initializes the LunarLanderA2C agent.

        Args:
            env: Gymnasium lunar lander environment.
            worker_id: Identifier for this worker.
            n_workers: Total number of workers.
            gamma: Discount factor.
        """
        super().__init__(env, worker_id, n_workers, gamma)

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


class CartPoleA2C(A2CWorker):
    """A2CWorker CartPole agent."""

    def __init__(self,
                 env: Env[np.array, int],
                 worker_id: int,
                 n_workers: int,
                 gamma: float = 0.99
                 ) -> None:
        """Initializes the CartPoleA2C agent.

        Args:
            env: Gymnasium cartpole environment.
            worker_id: Identifier for this worker.
            n_workers: Total number of workers.
            lambd: Lambda parameter for GAE.
            gamma: Discount factor.
        """
        super().__init__(env, worker_id, n_workers, gamma)

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


class MountainCarA2C(A2CWorker):
    """A2CWorker MountainCar agent."""

    def __init__(self,
                 env: Env[np.array, int],
                 worker_id: int,
                 n_workers: int,
                 gamma: float = 0.99,
                 ) -> None:
        """Initializes the MountainCarA2C agent.

        Args:
            env: Gymnasium mountain-car environment.
            worker_id: Identifier for this worker.
            n_workers: Total number of workers.
            gamma: Discount factor.
        """
        super().__init__(env, worker_id, n_workers, gamma)

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


class AcrobotA2C(A2CWorker):
    """A2CWorker Acrobot agent."""

    def __init__(self,
                 env: Env[np.array, int],
                 worker_id: int,
                 n_workers: int,
                 gamma: float = 0.99,
                 ) -> None:
        """Initializes the AcrobotA2C agent.

        Args:
            env: Gymnasium acrobot environment.
            worker_id: Identifier for this worker.
            n_workers: Total number of workers.
            gamma: Discount factor.
        """
        super().__init__(env, worker_id, n_workers, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Initializes the policy and value networks.

        Returns:
            A tuple containing:
                - policy_network: The policy network.
                - value_network: The value network.
        """

        policy_netywork = AcrobotPolicyNetwork()
        value_network = AcrobotValueNetwork()
        return policy_netywork, value_network
