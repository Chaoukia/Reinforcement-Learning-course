from dynamic_programming import *

class FrozenLakeDP(DynamicProgramming):
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
        env          : FrozenLake-v1 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        map          : np.array of shape, the grid map of the environment.
        shape        : Tuple, shape of the map grid.
        p_transition : np.array of shape (n_state, n_actions, n_states), the transition probabilities matrix.
        r_transition : np.array of shape (n_state, n_actions, n_states), the transition rewards matrix.
        """
        
        super(FrozenLakeDP, self).__init__(env, gamma)
        self.map = env.unwrapped.desc.astype(str)
        self.shape = self.map.shape
        self.p_transition, self.r_transition = self.make_transition_matrices()
        
    def make_transition_matrices(self):
        """
        Description
        --------------
        Construct the probability and reward transition matrices.
        
        Arguments
        --------------
        
        Returns
        --------------
        p_transition : np.array of shape (n_state, n_actions, n_states), the transition probabilities matrix.
        r_transition : np.array of shape (n_state, n_actions, n_states), the transition rewards matrix.
        """
        
        p_transition, r_transition = np.zeros((self.n_states, self.n_actions, self.n_states)), np.zeros((self.n_states, self.n_actions, self.n_states))
        for state in range(self.n_states):
            state_index = np.unravel_index(state, self.shape)
            # Holes are absorbing states.
            if self.map[state_index] == 'H':
                p_transition[state, :, state] = 1
                
            else:
                for action in range(self.n_actions):
                    if action == 0: # Go left.
                        # Going left.
                        next_state_index = np.array(state_index)
                        next_state_index[1] = max(0, next_state_index[1] - 1)
                        next_state = next_state_index[0]*self.shape[0] + next_state_index[1]
                        p_transition[state, action, next_state] += 1/3

                        # Slipping up.
                        next_state_index = np.array(state_index)
                        next_state_index[0] = max(0, next_state_index[0] - 1)
                        next_state = next_state_index[0]*self.shape[0] + next_state_index[1]
                        p_transition[state, action, next_state] += 1/3

                        # Slipping down.
                        next_state_index = np.array(state_index)
                        next_state_index[0] = min(self.shape[0] - 1, next_state_index[0] + 1)
                        next_state = next_state_index[0]*self.shape[0] + next_state_index[1]
                        p_transition[state, action, next_state] += 1/3

                    elif action == 2: # Go right.
                        # Going right.
                        next_state_index = np.array(state_index)
                        next_state_index[1] = min(self.shape[1] - 1, next_state_index[1] + 1)
                        next_state = next_state_index[0]*self.shape[0] + next_state_index[1]
                        p_transition[state, action, next_state] += 1/3

                        # Slipping up.
                        next_state_index = np.array(state_index)
                        next_state_index[0] = max(0, next_state_index[0] - 1)
                        next_state = next_state_index[0]*self.shape[0] + next_state_index[1]
                        p_transition[state, action, next_state] += 1/3

                        # Slipping down.
                        next_state_index = np.array(state_index)
                        next_state_index[0] = min(self.shape[0] - 1, next_state_index[0] + 1)
                        next_state = next_state_index[0]*self.shape[0] + next_state_index[1]
                        p_transition[state, action, next_state] += 1/3

                    elif action == 1: # Go down.
                        # Slipping left.
                        next_state_index = np.array(state_index)
                        next_state_index[1] = max(0, next_state_index[1] - 1)
                        next_state = next_state_index[0]*self.shape[0] + next_state_index[1]
                        p_transition[state, action, next_state] += 1/3

                        # Going down.
                        next_state_index = np.array(state_index)
                        next_state_index[0] = min(self.shape[0] - 1, next_state_index[0] + 1)
                        next_state = next_state_index[0]*self.shape[0] + next_state_index[1]
                        p_transition[state, action, next_state] += 1/3

                        # Slipping right.
                        next_state_index = np.array(state_index)
                        next_state_index[1] = min(self.shape[1] - 1, next_state_index[1] + 1)
                        next_state = next_state_index[0]*self.shape[0] + next_state_index[1]
                        p_transition[state, action, next_state] += 1/3

                    elif action == 3: # Go up.
                        # Slipping left.
                        next_state_index = np.array(state_index)
                        next_state_index[1] = max(0, next_state_index[1] - 1)
                        next_state = next_state_index[0]*self.shape[0] + next_state_index[1]
                        p_transition[state, action, next_state] += 1/3

                        # Going up.
                        next_state_index = np.array(state_index)
                        next_state_index[0] = max(0, next_state_index[0] - 1)
                        next_state = next_state_index[0]*self.shape[0] + next_state_index[1]
                        p_transition[state, action, next_state] += 1/3

                        # Slipping right.
                        next_state_index = np.array(state_index)
                        next_state_index[1] = min(self.shape[1] - 1, next_state_index[1] + 1)
                        next_state = next_state_index[0]*self.shape[0] + next_state_index[1]
                        p_transition[state, action, next_state] += 1/3
                
                    for next_state in range(self.n_states):
                        next_state_index = np.unravel_index(next_state, self.shape)
                        if self.map[next_state_index] == 'G':
                            r_transition[state, action, next_state] = 1
                        
        return p_transition, r_transition
    
class CliffWalkingDP(DynamicProgramming):
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
        p_transition : np.array of shape (n_state, n_actions, n_states), the transition probabilities matrix.
        r_transition : np.array of shape (n_state, n_actions, n_states), the transition rewards matrix.
        value        : np.array of shape (n_states,) or None, state values.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        policy       : np.array of shape (n_states,) or None, policy.
        """
        
        super(CliffWalkingDP, self).__init__(env, gamma)
        self.shape = (4, 12)
        self.p_transition, self.r_transition = self.make_transition_matrices()
        
    def make_transition_matrices(self):
        """
        Description
        --------------
        Construct the probability and reward transition matrices.
        
        Arguments
        --------------
        
        Returns
        --------------
        p_transition : np.array of shape (n_state, n_actions, n_states), the transition probabilities matrix.
        r_transition : np.array of shape (n_state, n_actions, n_states), the transition rewards matrix.
        """
        
        p_transition, r_transition = np.zeros((self.n_states, self.n_actions, self.n_states)), np.full((self.n_states, self.n_actions, self.n_states), -1)
        for state in range(self.n_states):
            state_index = np.unravel_index(state, self.shape)
            # Going over the cliff sends us back immediately to the initial state.
            if state_index[0] == 3 and state_index[1] in set(range(1, 11)):
                p_transition[state, :, 36] = 1
                
            # The goal state is an absorbing state.
            elif state_index == (3, 11):
                p_transition[state, :, state] = 1
                r_transition[state, :, :] = 0

            else:
                for action in range(self.n_actions):
                    if action == 0: # Go Up.
                        next_state_index = (max(0, state_index[0] - 1), state_index[1])
                        next_state = next_state_index[0]*self.shape[1] + next_state_index[1]
                        p_transition[state, action, next_state] = 1

                    elif action == 1: # Go right.
                        next_state_index = (state_index[0], min(11, state_index[1] + 1))
                        next_state = next_state_index[0]*self.shape[1] + next_state_index[1]
                        p_transition[state, action, next_state] = 1

                    elif action == 2: # Go down.
                        next_state_index = (min(3, state_index[0] + 1), state_index[1])
                        next_state = next_state_index[0]*self.shape[1] + next_state_index[1]
                        p_transition[state, action, next_state] = 1

                    elif action == 3: # Go left.
                        next_state_index = (state_index[0], max(0, state_index[1] - 1))
                        next_state = next_state_index[0]*self.shape[1] + next_state_index[1]
                        p_transition[state, action, next_state] = 1
                
                    for next_state in range(self.n_states):
                        next_state_index = np.unravel_index(next_state, self.shape)
                        # Going over the cliff incurs -100 reward.
                        if next_state_index[0] == 3 and next_state_index[1] in set(range(1, 11)):
                            r_transition[state, action, next_state] = -100
                        
        return p_transition, r_transition