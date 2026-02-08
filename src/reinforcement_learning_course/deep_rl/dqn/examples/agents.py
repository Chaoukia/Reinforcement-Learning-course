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
    """
    DQN LunarLander agent.
    """

    def __init__(self, 
                 env: Env[np.array, int], 
                 max_size: int = 100000, 
                 gamma: float = 0.99
                 ) -> None:
        """
        Description
        ------------------------------
        Constructor.

        Parameters
        ------------------------------
        env      : gymnasium lunar lander environment.
        max_size : Int, maximum size of the replay buffer.
        gamma    : Float, discount factor.

        Returns
        ------------------------------
        """
        super().__init__(env, max_size, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """
        Description
        ------------------------------
        Make the main and target networks.

        Parameters
        ------------------------------

        Returns
        ------------------------------
        q_network        : nn.Module, main network.
        q_network_target : nn.Module, target network.
        """

        q_network = LunarLanderNetwork()
        q_network_target = LunarLanderNetwork()
        q_network_target.load_state_dict(q_network.state_dict())
        return q_network, q_network_target
    

class CartPoleDQN(DQN):
    """
    DQN CartPole agent.
    """

    def __init__(self, 
                 env: Env[np.array, int], 
                 max_size: int = 500000, 
                 gamma: float = 0.99
                 ) -> None:
        """
        Description
        ------------------------------
        Constructor.

        Parameters
        ------------------------------
        env      : gymnasium cartpole environment.
        max_size : Int, maximum size of the replay buffer.
        gamma    : Float, discount factor.

        Returns
        ------------------------------
        """
        super().__init__(env, max_size, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """
        Description
        ------------------------------
        Make the main and target networks.

        Parameters
        ------------------------------

        Returns
        ------------------------------
        q_network        : nn.Module, main network.
        q_network_target : nn.Module, target network.
        """

        q_network = CartPoleNetwork()
        q_network_target = CartPoleNetwork()
        q_network_target.load_state_dict(q_network.state_dict())
        return q_network, q_network_target
    

class MountainCarDQN(DQN):
    """
    DQN MountainCar agent.
    """

    def __init__(self, 
                 env: Env[np.array, int], 
                 max_size: int = 500000, 
                 gamma: float = 0.99
                 ) -> None:
        """
        Description
        ------------------------------
        Constructor.

        Parameters
        ------------------------------
        env      : gymnasium mountain-car environment.
        max_size : Int, maximum size of the replay buffer.
        gamma    : Float, discount factor.

        Returns
        ------------------------------
        """
        super().__init__(env, max_size, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """
        Description
        ------------------------------
        Make the main and target networks.

        Parameters
        ------------------------------

        Returns
        ------------------------------
        q_network        : nn.Module, main network.
        q_network_target : nn.Module, target network.
        """

        q_network = MountainCarNetwork()
        q_network_target = MountainCarNetwork()
        q_network_target.load_state_dict(q_network.state_dict())
        return q_network, q_network_target
    

class AcrobotDQN(DQN):
    """
    DQN Acrobot agent.
    """

    def __init__(self, 
                 env: Env[np.array, int], 
                 max_size: int = 500000, 
                 gamma: float = 0.99
                 ) -> None:
        """
        Description
        ------------------------------
        Constructor.

        Parameters
        ------------------------------
        env      : gymnasium acrobot environment.
        max_size : Int, maximum size of the replay buffer.
        gamma    : Float, discount factor.

        Returns
        ------------------------------
        """
        super().__init__(env, max_size, gamma)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """
        Description
        ------------------------------
        Make the main and target networks.

        Parameters
        ------------------------------

        Returns
        ------------------------------
        q_network        : nn.Module, main network.
        q_network_target : nn.Module, target network.
        """

        q_network = AcrobotNetwork()
        q_network_target = AcrobotNetwork()
        q_network_target.load_state_dict(q_network.state_dict())
        return q_network, q_network_target
    
