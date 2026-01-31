from reinforce import Reinforce
from networks import LunarLanderNetwork, CartPoleNetwork, MountainCarNetwork, AcrobotNetwork

class LunarLanderReinforce(Reinforce):
    """
    Description
    --------------
    Class describing an Reinforce LunarLander agent.
    """

    def __init__(self, gamma=0.99):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        gamma    : Float in [0, 1] generally close to 1, discount factor.
        """

        super(LunarLanderReinforce, self).__init__(gamma)
        self.policy_network = LunarLanderNetwork()

class CartPoleReinforce(Reinforce):
    """
    Description
    --------------
    Class describing an Reinforce LunarLander agent.
    """

    def __init__(self, gamma=0.99):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        gamma    : Float in [0, 1] generally close to 1, discount factor.
        """

        super(CartPoleReinforce, self).__init__(gamma)
        self.policy_network = CartPoleNetwork()

class MountainCarReinforce(Reinforce):
    """
    Description
    --------------
    Class describing an Reinforce LunarLander agent.
    """

    def __init__(self, gamma=0.99):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        gamma    : Float in [0, 1] generally close to 1, discount factor.
        """

        super(MountainCarReinforce, self).__init__(gamma)
        self.policy_network = MountainCarNetwork()

class AcrobotReinforce(Reinforce):
    """
    Description
    --------------
    Class describing an Reinforce LunarLander agent.
    """

    def __init__(self, gamma=0.99):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        gamma    : Float in [0, 1] generally close to 1, discount factor.
        """

        super(AcrobotReinforce, self).__init__(gamma)
        self.policy_network = AcrobotNetwork()