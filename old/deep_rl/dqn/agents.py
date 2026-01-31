from dqn import DQN
from networks import LunarLanderNetwork, CartPoleNetwork, MountainCarNetwork, AcrobotNetwork

class LunarLanderDQN(DQN):
    """
    Description
    --------------
    Class describing an DQN LunarLander agent.
    """

    def __init__(self, env, gamma=0.99, max_size=500000, double_learning=False):
        """
        Description
        --------------
        Constructor of class LunarLanderDQN.
        
        Arguments
        --------------
        env      : LunarLander-v3 gymnasium environment.
        gamma    : Float in [0, 1] generally close to 1, discount factor.
        max_size : Int, maximum size of the replay buffer.
        """

        super(LunarLanderDQN, self).__init__(env, gamma, max_size, double_learning)
        self.q_network = LunarLanderNetwork()
        self.q_network_target = LunarLanderNetwork()
        self.update_q_network_target()

class CartPoleDQN(DQN):
    """
    Description
    --------------
    Class describing an DQN CartPoleDQN agent.
    """

    def __init__(self, env, gamma=0.99, max_size=500000, double_learning=False):
        """
        Description
        --------------
        Constructor of class CartPoleDQN.
        
        Arguments
        --------------
        env      : LunarLander-v3 gymnasium environment.
        gamma    : Float in [0, 1] generally close to 1, discount factor.
        max_size : Int, maximum size of the replay buffer.
        """

        super(CartPoleDQN, self).__init__(env, gamma, max_size, double_learning)
        self.q_network = CartPoleNetwork()
        self.q_network_target = CartPoleNetwork()
        self.update_q_network_target()

class MountainCarDQN(DQN):
    """
    Description
    --------------
    Class describing an DQN LunarLander agent.
    """

    def __init__(self, env, gamma=0.99, max_size=500000, double_learning=False):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env      : LunarLander-v3 gymnasium environment.
        gamma    : Float in [0, 1] generally close to 1, discount factor.
        max_size : Int, maximum size of the replay buffer.
        """

        super(MountainCarDQN, self).__init__(env, gamma, max_size, double_learning)
        self.q_network = MountainCarNetwork()
        self.q_network_target = MountainCarNetwork()
        self.update_q_network_target()

class AcrobotDQN(DQN):
    """
    Description
    --------------
    Class describing an DQN LunarLander agent.
    """

    def __init__(self, env, gamma=0.99, max_size=500000, double_learning=False):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env      : LunarLander-v3 gymnasium environment.
        gamma    : Float in [0, 1] generally close to 1, discount factor.
        max_size : Int, maximum size of the replay buffer.
        """

        super(AcrobotDQN, self).__init__(env, gamma, max_size, double_learning)
        self.q_network = AcrobotNetwork()
        self.q_network_target = AcrobotNetwork()
        self.update_q_network_target()