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
    """
    Reinforce LunarLander agent.
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

    def make_networks(self) -> nn.Module:
        """
        Description
        ------------------------------
        Initialize the policy network.

        Parameters
        ------------------------------

        Returns
        ------------------------------
        policy_network : nn.Module, policy network.
        """

        policy_network = LunarLanderNetwork()
        return policy_network
        

class CartPoleReinforce(Reinforce):
    """
    Reinforce CartPole agent.
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

    def make_networks(self) -> nn.Module:
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

        policy_network = CartPoleNetwork()
        return policy_network
    

class MountainCarReinforce(Reinforce):
    """
    Reinforce MountainCar agent.
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

    def make_networks(self) -> nn.Module:
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

        policy_network = MountainCarNetwork()
        return policy_network
    

class AcrobotReinforce(Reinforce):
    """
    Reinforce Acrobot agent.
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

    def make_networks(self) -> nn.Module:
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

        policy_netywork = AcrobotNetwork()
        return policy_netywork
    
