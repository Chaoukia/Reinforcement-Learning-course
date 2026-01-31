from reinforce_baseline import ReinforceBaseline
from networks import LunarLanderNetwork, CartPoleNetwork, MountainCarNetwork, AcrobotNetwork

class LunarLanderReinforce(ReinforceBaseline):
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
        self.policy_value_network = LunarLanderNetwork()

class CartPoleReinforce(ReinforceBaseline):
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
        self.policy_value_network = CartPoleNetwork()

class MountainCarReinforce(ReinforceBaseline):
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
        self.policy_value_network = MountainCarNetwork()

class AcrobotReinforce(ReinforceBaseline):
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
        self.policy_value_network = AcrobotNetwork()