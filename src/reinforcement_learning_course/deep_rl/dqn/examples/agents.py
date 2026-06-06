import numpy as np
import torch.nn as nn
from reinforcement_learning_course.deep_rl.dqn.algorithms import DQN
from reinforcement_learning_course.deep_rl.dqn.examples.neural_networks import (
    LunarLanderNetwork,
    CartPoleNetwork,
    MountainCarNetwork,
    AcrobotNetwork
)
from gymnasium import Env


class LunarLanderDQN(DQN):
    """DQN agent for the LunarLander environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 max_size: int = 100000,
                 gamma: float = 0.99,
                 double_learning: bool = False
                 ) -> None:
        """Initialize the LunarLanderDQN agent.

        Args:
            env: Gymnasium LunarLander environment.
            max_size: Maximum size of the replay buffer.
            gamma: Discount factor.
            double_learning: Whether to use double DQN learning.
        """
        super().__init__(env, max_size, gamma, double_learning)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Make the main and target networks.

        Returns:
            A tuple containing:
                - q_network: Main network.
                - q_network_target: Target network.
        """

        q_network = LunarLanderNetwork()
        q_network_target = LunarLanderNetwork()
        q_network_target.load_state_dict(q_network.state_dict())
        return q_network, q_network_target


class CartPoleDQN(DQN):
    """DQN agent for the CartPole environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 max_size: int = 500000,
                 gamma: float = 0.99,
                 double_learning: bool = False
                 ) -> None:
        """Initialize the CartPoleDQN agent.

        Args:
            env: Gymnasium CartPole environment.
            max_size: Maximum size of the replay buffer.
            gamma: Discount factor.
            double_learning: Whether to use double DQN learning.
        """
        super().__init__(env, max_size, gamma, double_learning)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Make the main and target networks.

        Returns:
            A tuple containing:
                - q_network: Main network.
                - q_network_target: Target network.
        """

        q_network = CartPoleNetwork()
        q_network_target = CartPoleNetwork()
        q_network_target.load_state_dict(q_network.state_dict())
        return q_network, q_network_target


class MountainCarDQN(DQN):
    """DQN agent for the MountainCar environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 max_size: int = 500000,
                 gamma: float = 0.99,
                 double_learning: bool = False
                 ) -> None:
        """Initialize the MountainCarDQN agent.

        Args:
            env: Gymnasium MountainCar environment.
            max_size: Maximum size of the replay buffer.
            gamma: Discount factor.
            double_learning: Whether to use double DQN learning.
        """
        super().__init__(env, max_size, gamma, double_learning)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Make the main and target networks.

        Returns:
            A tuple containing:
                - q_network: Main network.
                - q_network_target: Target network.
        """

        q_network = MountainCarNetwork()
        q_network_target = MountainCarNetwork()
        q_network_target.load_state_dict(q_network.state_dict())
        return q_network, q_network_target


class AcrobotDQN(DQN):
    """DQN agent for the Acrobot environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 max_size: int = 500000,
                 gamma: float = 0.99,
                 double_learning: bool = False
                 ) -> None:
        """Initialize the AcrobotDQN agent.

        Args:
            env: Gymnasium Acrobot environment.
            max_size: Maximum size of the replay buffer.
            gamma: Discount factor.
            double_learning: Whether to use double DQN learning.
        """
        super().__init__(env, max_size, gamma, double_learning)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Make the main and target networks.

        Returns:
            A tuple containing:
                - q_network: Main network.
                - q_network_target: Target network.
        """

        q_network = AcrobotNetwork()
        q_network_target = AcrobotNetwork()
        q_network_target.load_state_dict(q_network.state_dict())
        return q_network, q_network_target

