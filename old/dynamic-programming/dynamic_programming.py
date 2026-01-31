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
        env          : gymnasium environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        p_transition : np.array of shape (n_state, n_actions, n_states), the transition probabilities matrix.
        r_transition : np.array of shape (n_state, n_actions, n_states), the transition rewards matrix.
        value        : np.array of shape (n_states,) or None, state values.
        q_values     : np.array of shape (n_states, n_actions) or None, state-action values.
        policy       : np.array of shape (n_states,) or None, policy.
        """
        
        self.env = env
        self.gamma = gamma
        self.n_states = None
        self.n_actions = None
        self.p_transition, self.r_transition = None, None

    def reset(self):
        """
        Description
        --------------
        Reinitialize the state value function, the state-action value function and the policy.

        Arguments
        --------------

        Returns
        --------------
        """

        self.value = np.zeros(self.n_states)
        self.q_values = np.zeros((self.n_states, self.n_actions))
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
        epsilon : Float, small threshold 0 < epsilon << 1. Stop the algorithm if the norm difference between two consecutive values drops below epsilon.
        n       : Int, number of iterations.
        
        Returns
        --------------
        """
        
        i=0
        delta = 1
        while i < n:
            value_next = ((self.p_transition*(self.r_transition + self.gamma*self.value.reshape((1, 1, -1)))).sum(axis=-1)).max(axis=-1)
            delta = np.linalg.norm(value_next - self.value)
            self.value = value_next
            i += 1
            if delta < epsilon:
                self.policy = ((self.p_transition*(self.r_transition + self.gamma*self.value.reshape((1, 1, -1)))).sum(axis=-1)).argmax(axis=-1)
                print('Termination condition achieved after %d iterations.' %i)
                return
            
        self.policy = ((self.p_transition*(self.r_transition + self.gamma*self.value.reshape((1, 1, -1)))).sum(axis=-1)).argmax(axis=-1)
        print('The termination condition has not been achieved after %d iterations.' %i)
        
    def q_iteration(self, epsilon=1e-12, n=1000):
        """
        Description
        --------------
        Run the Q-iteration algorithm.
        
        Arguments
        --------------
        epsilon : Float, small threshold 0 < epsilon << 1, stop the algorithm if the norm difference between two consecutive values drops below epsilon.
        n       : Int, number of iterations.
        
        Returns
        --------------
        """
        
        i=0
        delta = 1
        while i < n:
            p_transition_reshape = np.expand_dims(self.p_transition, axis=3)
            r_transition_reshape = np.expand_dims(self.r_transition, axis=3)
            q_values_reshape = np.expand_dims(self.q_values, axis=(0, 1))
            q_values_next = ((p_transition_reshape*(r_transition_reshape + self.gamma*q_values_reshape)).sum(axis=-2)).max(axis=-1)
            delta = np.linalg.norm(q_values_next - self.q_values)
            self.q_values = q_values_next
            i += 1
            if delta < epsilon:
                self.policy = self.q_values.argmax(axis=-1)
                print('Termination condition achieved after %d iterations.' %i)
                return
            
        self.policy = self.q_values.argmax(axis=-1)
        print('The termination condition has not been achieved after %d iterations.' %i)
        
    def policy_iteration(self, epsilon=1e-12, n=1000):
        """
        Description
        --------------
        Run the Policy iteration algorithm.
        
        Arguments
        --------------
        epsilon : Float, small threshold 0 < epsilon << 1, stop the algorithm if the norm difference between two consecutive values drops below epsilon.
        n       : Int, number of iterations.
        
        Returns
        --------------
        """
        
        i=0
        delta = 1
        while i < n:
            p_transition_policy = self.p_transition[np.arange(self.n_states), self.policy, :]
            r_transition_policy = self.r_transition[np.arange(self.n_states), self.policy, :]
            p_r_policy = (p_transition_policy*r_transition_policy).sum(axis=-1)
            value_policy = np.linalg.solve(np.eye(self.n_states) - self.gamma*p_transition_policy, p_r_policy)
            policy_next = (((self.p_transition*(self.r_transition + self.gamma*value_policy.reshape((1, 1, -1)))).sum(axis=-1)).argmax(axis=-1)).astype(int)
            delta = np.linalg.norm(value_policy - self.value)
            delta_policy = (policy_next != self.policy).sum()
            self.policy = policy_next
            self.value = value_policy
            i += 1
            if delta_policy == 0 or delta < epsilon:
                print('Termination condition achieved after %d iterations.' %i)
                return
            
        print('The termination condition has not been achieved after %d iterations.' %i)
        
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
    
    def test(self, env, n_episodes=1000, verbose=False):
        """
        Description
        --------------
        Test the agent.
        
        Arguments
        --------------
        env        : gymnasium environment.
        n_episodes : Int, number of test episodes.
        verbose    : Boolean, if True, print the episode index and its corresponding length and return.
        
        Returns
        --------------
        """
        
        returns = np.empty(n_episodes)
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
                
            returns[episode] = R
            if verbose:
                print('Episode : %d, length : %d, return : %.3F' %(episode, n_steps, R))

        return_avg, return_std = returns.mean(), returns.std()
        print('avg : %.3f, std : %.3f' %(return_avg, return_std))
        return return_avg, return_std
            
    def save_gif(self, env, file_name, n_episodes=1):
        """
        Description
        --------------
        Test the agent and save a gif.
        
        Arguments
        --------------
        env       : gymnasium environment.
        file_name : String, path to the saved gif.
        
        Returns
        --------------
        """
        
        frames = []
        for i in range(n_episodes):
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

            frames.append(Image.fromarray(env.render(), mode='RGB'))
            
        frames[0].save(file_name, save_all=True, append_images=frames[1:], optimize=False, duration=150, loop=0)