import numpy as np
from PIL import Image

class DynamicProgramming:
    """
    Description
    --------------
    Class describing Dynamic Programming algorithms.
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
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        p_transition : np.array of shape (n_state, n_actions, n_states), the transition probabilities matrix.
        r_transition : np.array of shape (n_state, n_actions, n_states), the transition rewards matrix.
        values       : np.array of shape (n_states,) or None, state values.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        policy       : np.array of shape (n_states,) or None, policy.
        """
        
        self.env = env
        self.gamma = gamma
        self.map = None
        self.shape = None
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.p_transition, self.r_transition = None, None
        self.value = None
        self.q_values = None
        self.policy = np.zeros(self.n_states).astype(int)
        
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
        
        raise NotImplementedError
        
    def value_iteration(self, epsilon=1e-12, n=1000):
        """
        Description
        --------------
        Run the Value iteration algorithm.
        
        Arguments
        --------------
        epsilon : Float, small threshold 0 < epsilon << 1, stop the algorithm if the norm difference between two consecutive values
                  drops below epsilon.
        n       : Int, number of iterations.
        
        Returns
        --------------
        """
        
        i=0
        delta = 1
        self.value = np.zeros(self.n_states)
        while (i < n) and (delta >= epsilon):
            value_next = ((self.p_transition*(self.r_transition + self.gamma*self.value.reshape((1, 1, -1)))).sum(axis=-1)).max(axis=-1)
            delta = np.linalg.norm(value_next - self.value)
            self.value = value_next
            i += 1
            
        self.policy = ((self.p_transition*(self.r_transition + self.gamma*self.value.reshape((1, 1, -1)))).sum(axis=-1)).argmax(axis=-1)
        print('Termination condition achieved after %d iterations.' %i)
        
    def q_iteration(self, epsilon=1e-12, n=1000):
        """
        Description
        --------------
        Run the Q iteration algorithm.
        
        Arguments
        --------------
        epsilon : Float, small threshold 0 < epsilon << 1, stop the algorithm if the norm difference between two consecutive values
                  drops below epsilon.
        n       : Int, number of iterations.
        
        Returns
        --------------
        """
        
        i=0
        delta = 1
        self.q_values = np.zeros((self.n_states, self.n_actions))
        while (i < n) and (delta >= epsilon):
            p_transition_reshape = np.expand_dims(self.p_transition, axis=3)
            r_transition_reshape = np.expand_dims(self.r_transition, axis=3)
            q_values_reshape = np.expand_dims(self.q_values, axis=(0, 1))
            q_values_next = ((p_transition_reshape*(r_transition_reshape + self.gamma*q_values_reshape)).sum(axis=-2)).max(axis=-1)
            delta = np.linalg.norm(q_values_next - self.q_values)
            self.q_values = q_values_next
            i += 1
            
        self.policy = self.q_values.argmax(axis=-1)
        print('Termination condition achieved after %d iterations.' %i)
        
    def policy_iteration(self, epsilon=1e-12, n=1000):
        """
        Description
        --------------
        Run the Policy iteration algorithm.
        
        Arguments
        --------------
        epsilon : Float, small threshold 0 < epsilon << 1, stop the algorithm if the norm difference between two consecutive values
                  drops below epsilon.
        n       : Int, number of iterations.
        
        Returns
        --------------
        """
        
        i=0
        delta = 1
        self.value = np.zeros(self.n_states)
        while (i < n) and (delta >= epsilon):
            p_transition_policy = self.p_transition[np.arange(self.n_states), self.policy, :]
            r_transition_policy = self.r_transition[np.arange(self.n_states), self.policy, :]
            p_r_policy = (p_transition_policy*r_transition_policy).sum(axis=-1)
            value_policy = np.linalg.solve(np.eye(self.n_states) - self.gamma*p_transition_policy, p_r_policy)
            policy_next = (((self.p_transition*(self.r_transition + self.gamma*value_policy.reshape((1, 1, -1)))).sum(axis=-1)).argmax(axis=-1)).astype(int)
            delta = np.linalg.norm(value_policy - self.value)
            self.policy = policy_next
            self.value = value_policy
            i += 1
            
        print('Termination condition achieved after %d iterations.' %i)
        
    def action(self, state):
        """
        Description
        --------------
        Take an action according to the estimated optimal policy.
        
        Arguments
        --------------
        state : np.array, state.
        
        Returns
        --------------
        Int, action.
        """
        
        return self.policy[state]
    
    def test(self, env, n_episodes=1000):
        """
        Description
        --------------
        Test the agent.
        
        Arguments
        --------------
        env        : gym environment.
        n_episodes : Int, number of test episodes.
        
        Returns
        --------------
        """
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            done = False
            R = 0
            n_steps = 0
            while not done:
                action = self.action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = (terminated or truncated)
                state = next_state
                R += reward
                n_steps += 1
                
            print('Episode : %d, length : %d, reward : %.3F' %(episode, n_steps, R))
            
    def save_gif(self, env, file_name='frozen-lake.gif'):
        """
        Description
        --------------
        Test the agent and save a gif.
        
        Arguments
        --------------
        env        : gym environment.
        
        Returns
        --------------
        """
        
        frames = []
        state, _ = env.reset()
        done = False
        R = 0
        n_steps = 0
        while not done:
            frames.append(Image.fromarray(env.render(), mode='RGB'))
            action = self.action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = (terminated or truncated)
            state = next_state
            R += reward
            n_steps += 1
        
        frames[0].save(file_name, save_all=True, append_images=frames[1:], optimize=False, duration=150, loop=0)
    

class FrozenLake(DynamicProgramming):
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
        map          : np.array of shape (4, 4) or (5, 5), the grid map of the environment.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        p_transition : np.array of shape (n_state, n_actions, n_states), the transition probabilities matrix.
        r_transition : np.array of shape (n_state, n_actions, n_states), the transition rewards matrix.
        values       : np.array of shape (n_states,) or None, state values.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        policy       : np.array of shape (n_states,) or None, policy.
        """
        
        super(FrozenLake, self).__init__(env, gamma)
        self.map = env.desc.astype(str)
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