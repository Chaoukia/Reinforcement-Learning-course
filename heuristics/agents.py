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

    def expand(self, info_state):
        """
        Description
        --------------
        Expand a state by returning the states that stem from it by taking each possible action.

        Arguments
        --------------
        info_state : List of three elements.
                        - value   : Float, heuristic value of state. Equal to the sum of rewards following actions from the root plus an upper bound on the return of the optimal policy from state.
                        - reward  : Float, sum of rewards following actions from the root.
                        - actions : List of actions that lead to state from the initial state (root).
                        - state   : Int, a state.

        Returns
        --------------
        info_children : List of lists of the form [value, actions, next_state] where:
                        - value      : Float, heuristic value of next_state. Equal to the sum of rewards following actions from the root plus an upper bound on the return of the optimal policy from next_state.
                        - reward     : Float, sum of rewards following actions from the root.
                        - actions    : List of actions that lead to next_state from the initial state (root).
                        - next_state : Int, a next (child) state that stems from taking an action in state.
        """

        _, reward_neg, actions, state = info_state
        reward = -reward_neg
        info_children = []
        for action in range(self.n_actions):
            r, child = self.split(state, action)
            reward_child = reward + r
            value_child = reward_child + self.heuristic(child)
            info_child = [-value_child, -reward_child, actions + [action], child]
            info_children.append(info_child)

        return info_children