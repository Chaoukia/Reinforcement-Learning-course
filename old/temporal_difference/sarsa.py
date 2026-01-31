import numpy as np
from PIL import Image

class SARSA:
    
    """
    Class of the SARSA algorithm.
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
        q_values     : np.array of shape (n_states, n_actions) or None, state-action values.
        """
        
        self.env = env
        self.gamma = gamma
        self.n_states = None
        self.n_actions = None

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

        self.q_values, self.visits = {}, {}
            
    def action_explore(self, state, epsilon):
        """
        Description
        --------------
        Take an action according to an epsilon-greedy policy.
        
        Arguments
        --------------
        state   : np.array, state.
        epsilon : Float in ]0, 1[, probability of taking a suboptimal action.
        
        Returns
        --------------
        Int, action to perform.
        """

        action_max = self.action(state)
        bern = np.random.binomial(1, 1 - epsilon)
        if bern == 1:
            return action_max
        
        return self.env.action_space.sample()
        
    def action(self, state):
        """
        Description
        --------------
        Take an action according to the estimated optimal policy.
        
        Arguments
        --------------
        state : Int, state.
        
        Returns
        --------------
        Int, estimated optimal action.
        """

        try:
            return self.q_values[state].argmax()
        
        except KeyError:
            return self.env.action_space.sample()
        
    def update_q_value(self, state, action, reward, next_state, alpha, epsilon):
        """
        Description
        --------------
        Perform a sarsa update of experience (state, action, reward, next_state).
        
        Arguments
        --------------
        
        Returns
        --------------
        """

        try:
            action_next = self.action_explore(next_state, epsilon)
            q_max = self.q_values[next_state][action_next]

        except KeyError:
            self.q_values[next_state], self.visits[next_state] = np.zeros(self.n_actions), np.zeros(self.n_actions)
            q_max = 0

        try:
            td = reward + self.gamma*q_max - self.q_values[state][action]

        except KeyError:
            self.q_values[state], self.visits[state] = np.zeros(self.n_actions), np.zeros(self.n_actions)
            td = reward + self.gamma*q_max - self.q_values[state][action]
            
        self.visits[state][action] += 1
        if alpha is None:
            alpha = 1/(self.visits[state][action])

        self.q_values[state][action] += alpha*td
        
    def unroll(self, alpha, epsilon):
        """
        Description
        --------------
        Unroll the current epsilon-greedy policy from state and update the q-values at each step.
        
        Arguments
        --------------
        alpha   : Float in ]0, 1[, learning rate.
        epsilon : Float in ]0, 1[, parameter of the epsilon-greedy policy.
        
        Returns
        --------------
        """

        state, _ = self.env.reset()
        done = False
        while not done:
            action = self.action_explore(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = (terminated or truncated)
            self.update_q_value(state, action, reward, next_state, alpha, epsilon)
            state = next_state
            
    def train(self, alpha=0.1, epsilon_start=1, epsilon_stop=0.1, decay_rate=1e-3, n_train=1000, print_iter=10):
        """
        Description
        --------------
        Train an on-policy first-visit MC algorithm.
        
        Arguments
        --------------
        alpha         : Float in ]0, 1[, learning rate.
        epsilon_start : Float in ]0, 1[, initial value of epsilon.
        epsilon_stopt : Float in ]0, 1[, final value of epsilon.
        decay_rate    : Float, decay rate of epsilon from epsilon_start to epsilon_stop.
        n_train       : Int, total number of iterations.
        print_iter    : Int, number of iterations between two successive prints.
        
        Returns
        --------------
        """

        for i in range(n_train):
            epsilon = epsilon_stop + (epsilon_start - epsilon_stop)*np.exp(-decay_rate*i)
            self.unroll(alpha, epsilon)
            if i%print_iter == 0:
                print('Iteration : %d' %i)
                print('Epsilon   : %.5f' %epsilon)
                print('\n')
            
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