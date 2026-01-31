from astar import *

class CliffWalkingAstar(Astar):
    """
    Description
    --------------
    Class describing an A* agent operating in the FrozenLake environment.
    """
    
    def __init__(self, env):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        p_transition : np.array of shape (n_state, n_actions, n_states), the transition probabilities matrix.
        r_transition : np.array of shape (n_state, n_actions, n_states), the transition rewards matrix.
        value        : np.array of shape (n_states,) or None, state values.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        policy       : np.array of shape (n_states,) or None, policy.
        """
        
        super(CliffWalkingAstar, self).__init__(env)
        self.shape = (4, 12)
        self.goals = set([47])

    def heuristic(self, state):
        """
        Description
        --------------
        Return an upper bound on the optimal value at state.

        Arguments
        --------------
        state : Int, a state.

        Returns
        --------------
        Float, the heuristic value of state.
        """

        return 0

    def split(self, state, action):
        """
        Description
        --------------
        Return the reward and next state induced by taking an action in a state.

        Arguments
        --------------
        state  : Int, a state.
        action : Int, an action.

        Returns
        --------------
        Float, the induced reward.
        Int, the corresponding next state.
        """

        state_index = np.unravel_index(state, self.shape)
        if action == 0: # Go Up.
            next_state_index = (max(0, state_index[0] - 1), state_index[1])

        elif action == 1: # Go right.
            next_state_index = (state_index[0], min(11, state_index[1] + 1))

        elif action == 2: # Go down.
            next_state_index = (min(3, state_index[0] + 1), state_index[1])

        elif action == 3: # Go left.
            next_state_index = (state_index[0], max(0, state_index[1] - 1))
            
        # Going over the cliff sends us back immediately to the initial state and incurs a reward of -100.
        if next_state_index[0] == 3 and next_state_index[1] in set(range(1, 11)):
            reward, next_state = -100, 36

        # In all other situations we incur a reward of -1.
        else:
            reward, next_state = -1, next_state_index[0]*self.shape[1] + next_state_index[1]

        return reward, next_state
    
class FrozenLakeAstar(Astar):
    """
    Description
    --------------
    Class describing an A* agent operating in the FrozenLake environment.
    """
    
    def __init__(self, env):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env   : FrozenLake-v1 environment.
        map   : np.array, the grid map of the environment.
        shape : Tuple, shape of the map grid.
        goals : Set of goal states.
        holes : Set of holes, these are bad absorbing states.
        """
        
        super(FrozenLakeAstar, self).__init__(env)
        self.map = env.unwrapped.desc.astype(str)
        self.shape = self.map.shape
        self.goals = set(np.arange(self.map.shape[0]*self.map.shape[1]).reshape((self.map.shape[0], self.map.shape[1]))[self.map == 'G'])
        self.holes = set(np.arange(self.map.shape[0]*self.map.shape[1]).reshape((self.map.shape[0], self.map.shape[1]))[self.map == 'H'])

    def heuristic(self, state):
        """
        Description
        --------------
        Return an upper bound on the optimal value at state.
        Arguments
        --------------
        state : Int, a state.

        Returns
        --------------
        Float, the heuristic value of state.
        """

        if state in self.holes:
            return 0

        return 1

    def split(self, state, action):
        """
        Description
        --------------
        Return the reward and next state induced by taking an action in a state.

        Arguments
        --------------
        state  : Int, a state.
        action : Int, an action.

        Returns
        --------------
        Float, the induced reward.
        Int, the corresponding next state.
        """

        # A hole is an absorbing state.
        if state in self.holes:
            return 0, state

        state_index = np.unravel_index(state, self.shape)
        if action == 0: # Go left.
            next_state_index = np.array(state_index)
            next_state_index[1] = max(0, next_state_index[1] - 1)
            next_state = next_state_index[0]*self.shape[0] + next_state_index[1]

        elif action == 2: # Go right.
            next_state_index = np.array(state_index)
            next_state_index[1] = min(self.shape[1] - 1, next_state_index[1] + 1)
            next_state = next_state_index[0]*self.shape[0] + next_state_index[1]

        elif action == 1: # Go down.
            next_state_index = np.array(state_index)
            next_state_index[0] = min(self.shape[0] - 1, next_state_index[0] + 1)
            next_state = next_state_index[0]*self.shape[0] + next_state_index[1]

        elif action == 3: # Go up.
            next_state_index = np.array(state_index)
            next_state_index[0] = max(0, next_state_index[0] - 1)
            next_state = next_state_index[0]*self.shape[0] + next_state_index[1]

        reward = 1 if next_state in self.goals else 0
        return reward, next_state