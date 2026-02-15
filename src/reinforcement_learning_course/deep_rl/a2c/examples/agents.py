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
    """
    A2CWorker LunarLander agent.
    """

    def __init__(self, 
                 env: Env[np.array, int], 
                 n_workers: int,
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
        super().__init__(env, n_workers, gamma)

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
            

class CartPoleA2C(A2CWorker):
    """
    A2CWorker CartPole agent.
    """

    def __init__(self, 
                 env: Env[np.array, int], 
                 n_workers: int,
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
        super().__init__(env, n_workers, gamma)

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
    

class MountainCarA2C(A2CWorker):
    """
    A2CWorker MountainCar agent.
    """

    def __init__(self, 
                 env: Env[np.array, int], 
                 n_workers: int,
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
        super().__init__(env, n_workers, gamma)

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
    

class AcrobotA2C(A2CWorker):
    """
    A2CWorker Acrobot agent.
    """

    def __init__(self, 
                 env: Env[np.array, int], 
                 n_workers: int,
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
        super().__init__(env, n_workers, gamma)

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
    
