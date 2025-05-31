from dqn import DQN
from networks import LunarLanderNetwork

class LunarLanderDQN(DQN):
    """
    Description
    --------------
    Class describing an DQN LunarLander agent.
    """

    def __init__(self, env, gamma=0.99, max_size=500000):
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

        super(LunarLanderDQN, self).__init__(env, gamma, max_size)
        self.q_network = LunarLanderNetwork()