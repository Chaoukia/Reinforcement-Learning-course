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
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.map = env.unwrapped.desc.astype(str)
        self.shape = self.map.shape
        self.p_transition, self.r_transition = self.make_transition_matrices()
        self.reset()
        
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
        p_transition : np.array of shape (n_state, n_actions, n_states), the transition probabilities matrix.
        r_transition : np.array of shape (n_state, n_actions, n_states), the transition rewards matrix.
        value        : np.array of shape (n_states,) or None, state values.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        policy       : np.array of shape (n_states,) or None, policy.
        """
        
        super(CliffWalkingDP, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.shape = (4, 12)
        self.p_transition, self.r_transition = self.make_transition_matrices()
        self.reset()
        
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
    
class TaxiDP(DynamicProgramming):
    """
    Description
    --------------
    Class describing an agent operating in the Taxi-v3 environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : Taxi-v3 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        map          : np.array of shape, the grid map of the environment.
        shape        : Tuple, shape of the map grid.
        p_transition : np.array of shape (n_state, n_actions, n_states), the transition probabilities matrix.
        r_transition : np.array of shape (n_state, n_actions, n_states), the transition rewards matrix.
        """
        
        super(TaxiDP, self).__init__(env, gamma)
        self.n_states = env.observation_space.n + 1
        self.n_actions = env.action_space.n
        self.p_transition, self.r_transition = self.make_transition_matrices()
        self.reset()
        
    def make_transition_matrices(self):
        """
        Description
        --------------
        Construct the probability and reward transition matrices. Contrary to other environments such as FrozenLake and CliffWalking, we include an additional state in
        our transition matrices. This state represents an absorbing state. The reason behind this choice is that any state in the Taxi environment can be absorbing once we
        perform a drop off action in it. It is thus convenient to dedicate a special absorbing state in this context.
        
        Arguments
        --------------
        
        Returns
        --------------
        p_transition : np.array of shape (n_state+1, n_actions, n_states+1), the transition probabilities matrix.
        r_transition : np.array of shape (n_state+1, n_actions, n_states+1), the transition rewards matrix.
        """
        
        p_transition, r_transition = np.zeros((self.n_states, self.n_actions, self.n_states)), np.zeros((self.n_states, self.n_actions, self.n_states))
        for state in range(self.n_states):
            # self.n_states-1 is the absorbing state.
            if state == self.n_states - 1:
                p_transition[state, :, state] = 1
                
            else:
                taxi_row, taxi_col, passenger_loc, destination = self.env.unwrapped.decode(state)
                action_mask = self.env.unwrapped.action_mask(state)
                actions_allowed = np.arange(self.n_actions)[action_mask == 1]
                for action in actions_allowed:
                    if action == 0: # Go south.
                        taxi_row_new, taxi_col_new, passenger_loc_new = taxi_row + 1, taxi_col, passenger_loc
                        next_state = self.env.unwrapped.encode(taxi_row_new, taxi_col_new, passenger_loc_new, destination)

                        p_transition[state, action, next_state] = 1
                        r_transition[state, action, next_state] = -1

                    elif action == 1: # Go north.
                        taxi_row_new, taxi_col_new, passenger_loc_new = taxi_row - 1, taxi_col, passenger_loc
                        next_state = self.env.unwrapped.encode(taxi_row_new, taxi_col_new, passenger_loc_new, destination)
                        p_transition[state, action, next_state] = 1
                        r_transition[state, action, next_state] = -1

                    elif action == 2: # Go east.
                        taxi_row_new, taxi_col_new, passenger_loc_new = taxi_row, taxi_col + 1, passenger_loc
                        next_state = self.env.unwrapped.encode(taxi_row_new, taxi_col_new, passenger_loc_new, destination)
                        p_transition[state, action, next_state] = 1
                        r_transition[state, action, next_state] = -1

                    elif action == 3: # Go west.
                        taxi_row_new, taxi_col_new, passenger_loc_new = taxi_row, taxi_col - 1, passenger_loc
                        next_state = self.env.unwrapped.encode(taxi_row_new, taxi_col_new, passenger_loc_new, destination)
                        p_transition[state, action, next_state] = 1
                        r_transition[state, action, next_state] = -1

                    elif action == 4: # Pickup passenger.
                        taxi_row_new, taxi_col_new, passenger_loc_new = taxi_row, taxi_col, 4
                        next_state = self.env.unwrapped.encode(taxi_row_new, taxi_col_new, passenger_loc_new, destination)
                        p_transition[state, action, next_state] = 1
                        r_transition[state, action, next_state] = -1

                    elif action == 5: # Drop off passenger and transition to the absorbing state.
                        p_transition[state, action, self.n_states - 1] = 1
                        # If the passenger is dropped at the correct destination, incur a reward of 20. Otherwise incur a reward of -10.
                        if (taxi_row, taxi_col) == self.env.unwrapped.locs[destination]:
                            r_transition[state, action, self.n_states - 1] = 20

                        else:
                            r_transition[state, action, self.n_states - 1] = -10
                        
        return p_transition, r_transition