from mc import *

class CliffWalkingMC(MonteCarlo):
    """
    Description
    --------------
    Class describing an agent operating in the CliffWalking environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(CliffWalkingMC, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class FrozenLakeMC(MonteCarlo):
    """
    Description
    --------------
    Class describing an agent operating in the FrozenLake environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(FrozenLakeMC, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class BlackJackMC(MonteCarlo):
    """
    Description
    --------------
    Class describing an agent operating in the FrozenLake environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(BlackJackMC, self).__init__(env, gamma)
        self.n_actions = env.action_space.n
        self.reset()