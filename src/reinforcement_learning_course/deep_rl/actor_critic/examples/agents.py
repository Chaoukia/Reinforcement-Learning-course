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
    """
    ActorCritic LunarLander agent.
    """

    def __init__(self, 
                 env: Env[np.array, int], 
                 gamma: float = 0.99
                 ) -> None:
        """
        Description
        ------------------------------
        Constructor.

        Parameters
        ------------------------------
        env      : gymnasium lunar lander environment.
        gamma    : Float, discount factor.

        Returns
        ------------------------------
        """
        super().__init__(env, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """
        Description
        ------------------------------
        Initialize the policy network.

        Parameters
        ------------------------------

        Returns
        ------------------------------
        policy_network : nn.Module, policy network.
        value_network  : nn.Module, value network.
        """

        policy_network = LunarLanderPolicyNetwork()
        value_network = LunarLanderValueNetwork()
        return policy_network, value_network
        

class CartPoleActorCritic(ActorCritic):
    """
    ActorCritic CartPole agent.
    """

    def __init__(self, 
                 env: Env[np.array, int], 
                 gamma: float = 0.99
                 ) -> None:
        """
        Description
        ------------------------------
        Constructor.

        Parameters
        ------------------------------
        env      : gymnasium cartpole environment.
        gamma    : Float, discount factor.

        Returns
        ------------------------------
        """
        super().__init__(env, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """
        Description
        ------------------------------
        Make the main and target networks.

        Parameters
        ------------------------------

        Returns
        ------------------------------
        policy_network : nn.Module, policy network.
        """

        policy_network = CartPolePolicyNetwork()
        value_network = CartPoleValueNetwork()
        return policy_network, value_network
    

class MountainCarActorCritic(ActorCritic):
    """
    ActorCritic MountainCar agent.
    """

    def __init__(self, 
                 env: Env[np.array, int], 
                 gamma: float = 0.99, 
                 ) -> None:
        """
        Description
        ------------------------------
        Constructor.

        Parameters
        ------------------------------
        env      : gymnasium mountain-car environment.
        gamma    : Float, discount factor.

        Returns
        ------------------------------
        """
        super().__init__(env, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """
        Description
        ------------------------------
        Make the main and target networks.

        Parameters
        ------------------------------

        Returns
        ------------------------------
        policy_network : nn.Module, policy network.
        """

        policy_network = MountainCarPolicyNetwork()
        value_network = MountainCarValueNetwork()
        return policy_network, value_network
    

class AcrobotActorCritic(ActorCritic):
    """
    ActorCritic Acrobot agent.
    """

    def __init__(self, 
                 env: Env[np.array, int], 
                 gamma: float = 0.99, 
                 ) -> None:
        """
        Description
        ------------------------------
        Constructor.

        Parameters
        ------------------------------
        env      : gymnasium acrobot environment.
        gamma    : Float, discount factor.

        Returns
        ------------------------------
        """
        super().__init__(env, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """
        Description
        ------------------------------
        Make the main and target networks.

        Parameters
        ------------------------------

        Returns
        ------------------------------
        policy_netywork : nn.Module, policy network.
        """

        policy_netywork = AcrobotPolicyNetwork()
        value_network = AcrobotValueNetwork()
        return policy_netywork, value_network
    
