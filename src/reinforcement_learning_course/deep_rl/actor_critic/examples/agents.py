import numpy as np
import torch.nn as nn
from reinforcement_learning_course.deep_rl.actor_critic.algorithms import ActorCritic
from reinforcement_learning_course.deep_rl.actor_critic.examples.neural_networks import (
    LunarLanderPolicyNetwork, LunarLanderValueNetwork,
    CartPolePolicyNetwork, CartPoleValueNetwork,
    MountainCarPolicyNetwork, MountainCarValueNetwork,
    AcrobotPolicyNetwork, AcrobotValueNetwork,
)
from gymnasium import Env


class LunarLanderActorCritic(ActorCritic):
    """ActorCritic agent for the LunarLander environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 gamma: float = 0.99
                 ) -> None:
        """Initialize the LunarLander ActorCritic agent.

        Args:
            env: Gymnasium LunarLander environment.
            gamma: Discount factor.
        """
        super().__init__(env, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Initialize the policy and value networks.

        Returns:
            A tuple containing:
                - policy_network: The policy network.
                - value_network: The value network.
        """

        policy_network = LunarLanderPolicyNetwork()
        value_network = LunarLanderValueNetwork()
        return policy_network, value_network


class CartPoleActorCritic(ActorCritic):
    """ActorCritic agent for the CartPole environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 gamma: float = 0.99
                 ) -> None:
        """Initialize the CartPole ActorCritic agent.

        Args:
            env: Gymnasium CartPole environment.
            gamma: Discount factor.
        """
        super().__init__(env, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Initialize the policy and value networks.

        Returns:
            A tuple containing:
                - policy_network: The policy network.
                - value_network: The value network.
        """

        policy_network = CartPolePolicyNetwork()
        value_network = CartPoleValueNetwork()
        return policy_network, value_network


class MountainCarActorCritic(ActorCritic):
    """ActorCritic agent for the MountainCar environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 gamma: float = 0.99,
                 ) -> None:
        """Initialize the MountainCar ActorCritic agent.

        Args:
            env: Gymnasium MountainCar environment.
            gamma: Discount factor.
        """
        super().__init__(env, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Initialize the policy and value networks.

        Returns:
            A tuple containing:
                - policy_network: The policy network.
                - value_network: The value network.
        """

        policy_network = MountainCarPolicyNetwork()
        value_network = MountainCarValueNetwork()
        return policy_network, value_network


class AcrobotActorCritic(ActorCritic):
    """ActorCritic agent for the Acrobot environment."""

    def __init__(self,
                 env: Env[np.array, int],
                 gamma: float = 0.99,
                 ) -> None:
        """Initialize the Acrobot ActorCritic agent.

        Args:
            env: Gymnasium Acrobot environment.
            gamma: Discount factor.
        """
        super().__init__(env, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Initialize the policy and value networks.

        Returns:
            A tuple containing:
                - policy_network: The policy network.
                - value_network: The value network.
        """

        policy_netywork = AcrobotPolicyNetwork()
        value_network = AcrobotValueNetwork()
        return policy_netywork, value_network
