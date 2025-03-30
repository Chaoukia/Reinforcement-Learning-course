import numpy as np
from PIL import Image

class MonteCarlo:
    """
    Description
    --------------
    Class describing Monte Carlo algorithms.
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

        self.q_values = {}

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
        
        return np.random.choice(self.n_actions)
        
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
    
    def unroll(self, epsilon):
        """
        Description
        --------------
        Unroll the current epsilon-greedy policy from state.
        
        Arguments
        --------------
        state : Int, an initial state.
        
        Returns
        --------------
        """

        states, actions, rewards = [], [], []
        state, _ = self.env.reset()
        done = False
        while not done:
            action = self.action_explore(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = (terminated or truncated)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        return states, actions, rewards
    
    def train(self, epsilon_start=1, epsilon_stop=0.1, decay_rate=1e-3, n_train=1000, first_visit=True, print_iter=10):
        """
        Description
        --------------
        Train an on-policy first-visit MC algorithm.
        
        Arguments
        --------------
        epsilon     : Float in ]0, 1[, probability of taking a suboptimal action according to an epsilon-greedy policy.
        n_train     : Int, number of training episodes.
        first_visit : Boolean, whether to apply first visit MC or every visit MC.
        
        Returns
        --------------
        """

        visits = {}
        for i in range(n_train):
            epsilon = epsilon_stop + (epsilon_start - epsilon_stop)*np.exp(-decay_rate*i)
            states, actions, rewards = self.unroll(epsilon)
            G = 0
            for t in range(-1, -len(states)-1, -1):
                state, action, reward = states[t], actions[t], rewards[t]
                G = self.gamma*G + reward
                if (first_visit and state not in set(states[:t])) or (not first_visit):
                    try:
                        self.q_values[state][action] = (self.q_values[state][action]*visits[state][action] + G)/(visits[state][action] + 1)
                        visits[state][action] = visits[state][action] + 1

                    except KeyError:
                        self.q_values[state] = np.zeros(self.n_actions)
                        visits[state] = np.zeros(self.n_actions)
                        self.q_values[state][action] = G
                        visits[state][action] = 1

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