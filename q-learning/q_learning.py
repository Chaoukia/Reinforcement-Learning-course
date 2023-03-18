from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import pickle as pkl
from collections import deque
from PIL import Image

class QLearning:
    
    """
    Class of the Q-Learning algorithm. This is a ready class that discretises all dimensions of the state space.
    """
    
    def __init__(self, state_dim, n_actions, gamma=.9):
        """
        Description
        -------------------------
        Constructor of class QLearning.
        
        Arguments & Attributes
        -------------------------
        state_dim : Int, dimension of the state space.
        n_actions : Int, number of actions.
        gamma     : Float, discount factor.
        bins      : List of bins, each bin corresponds to a state dimension.
        q_values  : np.array of shape (|state_1|, ..., |state_p|, n_actions), where |state_i| is the number of categories that
                    the ith state coordinate can take. State-Action value function.
        visits    : np.array of shape (|state_1|, ..., |state_p|, n_actions), number of visits of each state-action pair.
        """
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.bins = None
        self.q_values = None
        self.visits = None
        
    def hist(self, env, n_bins=20, n_episodes=10):
        """
        Description
        --------------
        Run a pretraining phase where each state dimension is discretised with bins.
        
        Arguments
        --------------
        env        : gym environment.
        n_bins     : Int, number of bins to discretise each state dimension.
        n_episodes : Int, number of episodes in the pretraining phase.
        
        Returns
        --------------
        List of the derived bins.
        """
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            done = False
            states = [[state_i] for state_i in state] # List containing the lists of observations of each state dimension.
            while not done:
                action = env.action_space.sample()
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = (terminated) or (truncated)
                for i in range(self.state_dim):
                    states[i].append(next_state[i])
                    
        return [np.histogram(states[i], bins=n_bins)[1] for i in range(self.state_dim)]
        
    def initialize(self, env, n_bins=20, n_episodes=10):
        """
        Description
        --------------
        Initialize the Q-values and number of visits of each state-action pair.
        
        Arguments
        --------------
        env        : gym environment.
        n_bins     : Int, number of bins to discretise each state dimension.
        n_episodes : Int, number of episodes in the pretraining phase.
        
        Returns
        --------------
        """
        
        # Initialise the bins.
        self.bins = self.hist(env, n_bins, n_episodes)
        self.q_values = np.zeros(tuple([n_bins+2 for i in range(self.state_dim)] + [self.n_actions]))
        self.visits = np.zeros(tuple([n_bins+2 for i in range(self.state_dim)] + [self.n_actions]))
            
    def action_explore(self, state_discretised, epsilon=0.1):
        """
        Description
        --------------
        Take an action according to the epsilon-greedy policy and update arrays q_values and visits.
        
        Arguments
        --------------
        state_discretised   : np.array, discretised state.
        epsilon             : Float in [0, 1], parameter of the epsilon-greedy policy.
        
        Returns
        --------------
        action : Int, action.
        """
        
        bern = np.random.binomial(1, 1 - epsilon)
        if bern == 1:
            action = np.argmax(self.q_values[state_discretised])
            
        else:
            action = np.random.choice(self.n_actions)
            
        self.visits[state_discretised][action] += 1  # Update the number of visits of this state-action pair.
        return action
    
    def action(self, state):
        """
        Description
        --------------
        Take the action maximising the estimated Q-value.
        
        Arguments
        --------------
        state : np.array, state.
        
        Returns
        --------------
        Int, action to take.
        """
        
        state_discretised = self.discretise(state)
        return np.argmax(self.q_values[state_discretised])
    
    def discretise(self, state):
        """
        Description
        --------------
        Discretise a state according to the bins derived during the pretraining phase.
        
        Arguments
        --------------
        state : np.array, state.
        
        Returns
        --------------
        Tuple, discretised state.
        """
        
        return tuple(np.digitize(state[i], self.bins[i]) for i in range(self.state_dim))
            
    def train(self, env, n_episodes=100, epsilon_start=1, epsilon_stop=0.1, decay_rate=1e-5, log_dir='runs_qlearning', thresh=450):
        """
        Description
        --------------
        Train the agent.
        
        Arguments
        --------------
        env           : gym environment.
        n_espisodes   : Int, number of training episodes.
        epsilon_start : Float in [0, 1], initial value of epsilon.
        epsilon_stop  : Float in [0, 1], final value of epsilon.
        decay_rate    : Float in [0, 1], decay rate of epsilon.
        log_dir       : String, path of the folder where to store tensorboard events.
        thresh        : Float, lower bound on the average of the last 10 training episodes above which early stopping is activated.
        
        Returns
        --------------
        """
        
        writer = SummaryWriter(log_dir=log_dir)
        returns = deque(maxlen=10)
        epsilon = epsilon_start
        it = 0
        for episode in range(n_episodes):
            state, _ = env.reset()
            state_discretised = self.discretise(state)
            done = False
            R = 0
            while not done:
                action = self.action_explore(state_discretised, epsilon)
                epsilon = epsilon_stop + (epsilon_start - epsilon_stop)*np.exp(-decay_rate*it)
                # Get the corresponding transition from the environment.
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = (terminated) or (truncated)
                next_state_discretised = self.discretise(next_state)
                R += reward
                # Compute the temporal difference.                    
                td = reward + self.gamma*np.max(self.q_values[next_state_discretised])*(1 - done) - self.q_values[state_discretised][action]
                # Update the Q-value.
                self.q_values[state_discretised][action] = self.q_values[state_discretised][action] + td/self.visits[state_discretised][action]
                state = next_state
                state_discretised = self.discretise(state)
                it += 1
                
            returns.append(R)
            R_mean = np.mean(returns)
            writer.add_scalar('return', R_mean, episode)
            print('Episode : %d, epsilon : %.3f, return : %.3F' %(episode, epsilon, R))
            
            if R_mean > thresh:
                print('Early stopping achieved after %d episodes' %episode)
                break
            
    def test(self, env, n_episodes=1000):
        """
        Description
        --------------
        Test the agent.
        
        Arguments
        --------------
        env         : gym environment.
        n_espisodes : Int, number of training episodes.
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
            
    def save_gif(self, env, file_name='cartpole.gif'):
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
        
        frames[0].save(file_name, save_all=True, append_images=frames[1:], optimize=True, duration=40, loop=0)
                                            
    def save_weights(self, path):
        """
        Description
        --------------
        Save the agents' q-values.
        
        Parameters
        --------------
        path: String, path to a .pkl file containing q-dictionary.
        
        Returns
        --------------
        """
        
        with open(path, 'wb') as f:
            pkl.dump(self.q_values, f)
    
    def load_weights(self, path):
        """
        Description
        --------------
        Load the agents' q-values.
        
        Parameters
        --------------
        path: String, path to a .pkl file containing q-dictionary.
        
        Returns
        --------------
        """
        
        with open(path, 'rb') as f:
            self.q_values = pkl.load(f)
            
            
class CartPole(QLearning):
    
    """
    Class of Cartopole Agent. We define this subclass with custom rewards and discretisation of the state space.
    """
    
    def __init__(self, state_dim, n_actions, gamma=.9):
        """
        Description
        -------------------------
        Constructor of class QLearning.
        
        Arguments & Attributes
        -------------------------
        state_dim : Int, dimension of the state space.
        n_actions : Int, number of actions.
        gamma     : Float, discount factor.
        bins      : List of bins, each bin corresponds to a state dimension.
        q_values  : np.array of shape (|state_1|, ..., |state_p|, n_actions), where |state_i| is the number of categories that
                    the ith state coordinate can take. State-Action value function.
        visits    : np.array of shape (|state_1|, ..., |state_p|, n_actions), number of visits of each state-action pair.
        """
        
        super(CartPole, self).__init__(state_dim, n_actions, gamma)
        
    def hist(self, env, n_bins=20, n_episodes=10):
        """
        Description
        --------------
        Run a pretraining phase where each state dimension is discretised with bins.
        
        Arguments
        --------------
        env        : gym environment.
        n_bins     : Int, number of bins to discretise each state dimension.
        n_episodes : Int, number of episodes in the pretraining phase.
        
        Returns
        --------------
        List of the derived bins.
        """
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            done = False
            states = [[state_i] for state_i in state] # List containing the lists of observations of each state dimension.
            while not done:
                action = env.action_space.sample()
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = (terminated) or (truncated)
                for i in range(self.state_dim):
                    states[i].append(next_state[i])
                    
        return [np.histogram(states[i], bins=n_bins)[1] for i in range(1, self.state_dim)]
        
    def initialize(self, env, n_bins=20, n_episodes=10):
        """
        Description
        --------------
        Initialize the Q-values and number of visits of each state-action pair.
        
        Arguments
        --------------
        env        : gym environment.
        n_bins     : Int, number of bins to discretise each state dimension.
        n_episodes : Int, number of episodes in the pretraining phase.
        
        Returns
        --------------
        """
        
        # Initialise the bins.
        self.bins = self.hist(env, n_bins, n_episodes)
        self.q_values = np.zeros(tuple([2] + [n_bins+2 for i in range(1, self.state_dim)] + [self.n_actions]))
        self.visits = np.zeros(tuple([2] + [n_bins+2 for i in range(1, self.state_dim)] + [self.n_actions]))
        
    def discretise(self, state):
        """
        Description
        --------------
        Discretise a state according to the bins derived during the pretraining phase.
        
        Arguments
        --------------
        state : np.array, state.
        
        Returns
        --------------
        Tuple, discretised state.
        """
        
        return tuple([int(state[0] >= 0)] + list(np.digitize(state[i], self.bins[i-1]) for i in range(1, self.state_dim)))
    
    def train(self, env, n_episodes=100, epsilon_start=1, epsilon_stop=0.1, decay_rate=1e-5, log_dir='runs_qlearning', thresh=450):
        """
        Description
        --------------
        Custom training of a CartPole agent.
        
        Arguments
        --------------
        env           : gym environment.
        n_espisodes   : Int, number of training episodes.
        epsilon_start : Float in [0, 1], initial value of epsilon.
        epsilon_stop  : Float in [0, 1], final value of epsilon.
        decay_rate    : Float in [0, 1], decay rate of epsilon.
        log_dir       : String, path of the folder where to store tensorboard events.
        thresh        : Float, lower bound on the average of the last 10 training episodes above which early stopping is activated.
        
        Returns
        --------------
        """
        
        writer = SummaryWriter(log_dir=log_dir)
        returns = deque(maxlen=10)
        epsilon = epsilon_start
        it = 0
        for episode in range(n_episodes):
            state, _ = env.reset()
            state_discretised = self.discretise(state)
            done = False
            R = 0
            while not done:
                action = self.action_explore(state_discretised, epsilon)
                epsilon = epsilon_stop + (epsilon_start - epsilon_stop)*np.exp(-decay_rate*it)
                # Get the corresponding transition from the environment.
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = (terminated) or (truncated)
                next_state_discretised = self.discretise(next_state)
                R += reward
                if done and R < 180:
                    reward = -300
                    
                elif done and 180 < R < 350:
                    reward = -100
                    
                # Compute the temporal difference.                    
                td = reward + self.gamma*np.max(self.q_values[next_state_discretised])*(1 - done) - self.q_values[state_discretised][action]
                # Update the Q-value.
                self.q_values[state_discretised][action] = self.q_values[state_discretised][action] + td/self.visits[state_discretised][action]
                state = next_state
                state_discretised = self.discretise(state)
                it += 1
                
            returns.append(R)
            R_mean = np.mean(returns)
            writer.add_scalar('return', R_mean, episode)
            print('Episode : %d, epsilon : %.3f, return : %.3F' %(episode, epsilon, R))
            
            if R_mean > thresh:
                print('Early stopping achieved after %d episodes' %episode)
                break
        
        
        
        
        
        
        
        
        
        