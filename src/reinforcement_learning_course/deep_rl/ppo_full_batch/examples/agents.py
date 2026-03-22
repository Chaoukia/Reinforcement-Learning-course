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
    """
    PPOWorker LunarLander agent.
    """

    def __init__(self, 
                 env: Env[np.array, int], 
                 n_workers: int,
                 epsilon: float,
                 lambd: float,
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
        super().__init__(env, n_workers, epsilon, lambd, gamma)

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
            

class CartPolePPO(PPOWorker):
    """
    PPOWorker CartPole agent.
    """

    def __init__(self, 
                 env: Env[np.array, int], 
                 n_workers: int,
                 epsilon: float,
                 lambd: float,
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
        super().__init__(env, n_workers, epsilon, lambd, gamma)

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
    

class MountainCarPPO(PPOWorker):
    """
    PPOWorker MountainCar agent.
    """

    def __init__(self, 
                 env: Env[np.array, int], 
                 n_workers: int,
                 epsilon: float,
                 lambd: float,
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
        super().__init__(env, n_workers, epsilon, lambd, gamma)

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
    

class AcrobotPPO(PPOWorker):
    """
    PPOWorker Acrobot agent.
    """

    def __init__(self, 
                 env: Env[np.array, int], 
                 n_workers: int,
                 epsilon: float,
                 lambd: float,
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
        super().__init__(env, n_workers, epsilon, lambd, gamma)

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
    
