import numpy as np
import torch.nn as nn
import gymnasium as gym
from reinforcement_learning_course.deep_rl.ppo.algorithms import PPOWorker
from reinforcement_learning_course.deep_rl.ppo.examples.neural_networks import (
    LunarLanderPolicyNetwork, LunarLanderValueNetwork,
    CartPolePolicyNetwork, CartPoleValueNetwork,
    MountainCarPolicyNetwork, MountainCarValueNetwork,
    AcrobotPolicyNetwork, AcrobotValueNetwork,
)
from gymnasium import Env


class LunarLanderPPO(PPOWorker):
    """PPOWorker LunarLander agent."""

    def __init__(self,
                 env: Env[np.array, int],
                 worker_id: int,
                 n_workers: int,
                 epsilon: float,
                 lambd: float,
                 gamma: float = 0.99
                 ) -> None:
        """Initializes the LunarLanderPPO agent.

        Args:
            env: Gymnasium lunar lander environment.
            worker_id: Identifier for this worker.
            n_workers: Total number of workers.
            epsilon: Clipping parameter for PPO.
            lambd: Lambda parameter for GAE.
            gamma: Discount factor.
        """
        super().__init__(env, worker_id, n_workers, epsilon, lambd, gamma)

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


class CartPolePPO(PPOWorker):
    """PPOWorker CartPole agent."""

    def __init__(self,
                 env: Env[np.array, int],
                 worker_id: int,
                 n_workers: int,
                 epsilon: float,
                 lambd: float,
                 gamma: float = 0.99
                 ) -> None:
        """Initializes the CartPolePPO agent.

        Args:
            env: Gymnasium cartpole environment.
            worker_id: Identifier for this worker.
            n_workers: Total number of workers.
            epsilon: Clipping parameter for PPO.
            lambd: Lambda parameter for GAE.
            gamma: Discount factor.
        """
        super().__init__(env, worker_id, n_workers, epsilon, lambd, gamma)

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


class MountainCarPPO(PPOWorker):
    """PPOWorker MountainCar agent."""

    def __init__(self,
                 env: Env[np.array, int],
                 worker_id: int,
                 n_workers: int,
                 epsilon: float,
                 lambd: float,
                 gamma: float = 0.99,
                 ) -> None:
        """Initializes the MountainCarPPO agent.

        Args:
            env: Gymnasium mountain-car environment.
            worker_id: Identifier for this worker.
            n_workers: Total number of workers.
            epsilon: Clipping parameter for PPO.
            lambd: Lambda parameter for GAE.
            gamma: Discount factor.
        """
        super().__init__(env, worker_id, n_workers, epsilon, lambd, gamma)

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


class AcrobotPPO(PPOWorker):
    """PPOWorker Acrobot agent."""

    def __init__(self,
                 env: Env[np.array, int],
                 worker_id: int,
                 n_workers: int,
                 epsilon: float,
                 lambd: float,
                 gamma: float = 0.99,
                 ) -> None:
        """Initializes the AcrobotPPO agent.

        Args:
            env: Gymnasium acrobot environment.
            worker_id: Identifier for this worker.
            n_workers: Total number of workers.
            epsilon: Clipping parameter for PPO.
            lambd: Lambda parameter for GAE.
            gamma: Discount factor.
        """
        super().__init__(env, worker_id, n_workers, epsilon, lambd, gamma)

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
