from q_learning import QLearning
from sarsa import SARSA
from expected_sarsa import ExpectedSARSA
from double_q_learning import DoubleQLearning

class FrozenLakeQLearning(QLearning):
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
        
        super(FrozenLakeQLearning, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class CliffWalkingQLearning(QLearning):
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
        
        super(CliffWalkingQLearning, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class TaxiQLearning(QLearning):
    """
    Description
    --------------
    Class describing an agent operating in the Taxi environment.
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
        
        super(TaxiQLearning, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class BlackJackQLearning(QLearning):
    """
    Description
    --------------
    Class describing an agent operating in the BlackJack environment.
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
        
        super(BlackJackQLearning, self).__init__(env, gamma)
        self.n_actions = env.action_space.n
        self.reset()

class FrozenLakeSARSA(SARSA):
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
        
        super(FrozenLakeSARSA, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class CliffWalkingSARSA(SARSA):
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
        
        super(CliffWalkingSARSA, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class TaxiSARSA(SARSA):
    """
    Description
    --------------
    Class describing an agent operating in the Taxi environment.
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
        
        super(TaxiSARSA, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class BlackJackSARSA(SARSA):
    """
    Description
    --------------
    Class describing an agent operating in the BlackJack environment.
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
        
        super(BlackJackSARSA, self).__init__(env, gamma)
        self.n_actions = env.action_space.n
        self.reset()

class FrozenLakeExpectedSarsa(ExpectedSARSA):
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
        
        super(FrozenLakeExpectedSarsa, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class CliffWalkingExpectedSARSA(ExpectedSARSA):
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
        
        super(CliffWalkingExpectedSARSA, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class TaxiExpectedSARSA(ExpectedSARSA):
    """
    Description
    --------------
    Class describing an agent operating in the Taxi environment.
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
        
        super(TaxiExpectedSARSA, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class BlackJackExpectedSARSA(ExpectedSARSA):
    """
    Description
    --------------
    Class describing an agent operating in the BlackJack environment.
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
        
        super(BlackJackExpectedSARSA, self).__init__(env, gamma)
        self.n_actions = env.action_space.n
        self.reset()

class FrozenLakeDoubleQLearning(DoubleQLearning):
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
        
        super(FrozenLakeDoubleQLearning, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class CliffWalkingDoubleQLearning(DoubleQLearning):
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
        
        super(CliffWalkingDoubleQLearning, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class TaxiDoubleQLearning(DoubleQLearning):
    """
    Description
    --------------
    Class describing an agent operating in the Taxi environment.
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
        
        super(TaxiDoubleQLearning, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class BlackJackDoubleQLearning(DoubleQLearning):
    """
    Description
    --------------
    Class describing an agent operating in the BlackJack environment.
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
        
        super(BlackJackDoubleQLearning, self).__init__(env, gamma)
        self.n_actions = env.action_space.n
        self.reset()