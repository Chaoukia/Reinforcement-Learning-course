import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils import *
from PIL import Image

class DQN:
    
    """
    Class of the DQN algorithm.
    """
    
    def __init__(self, env, gamma=0.99, max_size=500000, double_learning=False):
        """
        Description
        -------------------------
        Constructor of class DQN.
        
        Arguments and Attributes
        -------------------------
        gamma      : Float in [0, 1] generally very close to 1, discount factor.
        max_size   : Int, maximum size of the replay memory.
        """
        
        self.env = env
        self.gamma = gamma
        self.max_size = max_size
        self.double_learning = double_learning
        self.q_network = None
        self.q_network_target = None
        self.buffer = Memory(max_size=max_size)
        
    def action_explore(self, state, epsilon):
        """
        Description
        --------------
        Take an action according to the epsilon-greedy policy.
        
        Arguments
        --------------
        state   : np.array, state from a gym environment.
        epsilon : Float in [0, 1], parameter of the epsilon-greedy policy.
        
        Returns
        --------------
        action : Int, action to take.
        """
        
        bern = np.random.binomial(1, epsilon)
        if bern == 1: return self.env.action_space.sample()

        return self.action(state)

    def action(self, state):
        """
        Description
        --------------
        Take the action maximising the highest currently estimated Q-value.
        
        Arguments
        --------------
        state : np.array, state from a gym environment.
        
        Returns
        --------------
        Int, action to take.
        """
        
        with torch.no_grad():
            q_values = self.q_network(torch.from_numpy(state))
            return torch.argmax(q_values).item()
        
    def sample_batch(self, batch_size):
        """
        Description
        -------------
        Sample a batch from the replay buffer.
        
        Arguments
        -------------
        batch_size : Int, the batch size.
        
        Returns
        --------------
        states_batch      : torch.tensor of shape (batch_size, state_shape).
        actions_batch     : np.array of shape (batch_size,).
        rewards_batch     : torch.tensor of shape (batch_size, 1).
        next_states_batch : torch.tensor of shape (batch_size, state_shape).
        dones_batch       : torch.tensor of shape (batch_size,).
        """

        batch = self.buffer.sample(batch_size)
        states_batch = torch.tensor(np.stack(batch.state), dtype=torch.float32)
        actions_batch = np.array(batch.action)
        rewards_batch = torch.tensor(batch.reward, dtype=torch.float32)
        next_states_batch = torch.tensor(np.stack(batch.next_state), dtype=torch.float32)
        dones_batch = torch.tensor(batch.done, dtype=torch.int)
        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch
        
    def update_q_network(self, batch_size, optimizer, writer, it):
        """
        Description
        -------------
        Update the weights of the q network.
        
        Arguments
        -------------
        batch_size : Int, the batch size.
        
        Returns
        --------------
        """

        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = self.sample_batch(batch_size)
        # q_values = self.q_network(states_batch)[np.arange(batch_size), actions_batch]
        q_values_states = self.q_network(states_batch)
        q_values = q_values_states[np.arange(batch_size), actions_batch]
        with torch.no_grad(): 
            q_values_next_states = self.q_network_target(next_states_batch)
            # DQN targets
            if not self.double_learning:
                q_targets = rewards_batch + self.gamma*torch.max(q_values_next_states, dim=1).values*dones_batch

            # Double DQN target.
            else:
                actions_max = torch.argmax(q_values_states, dim=1)
                q_targets = rewards_batch + self.gamma*q_values_next_states[np.arange(batch_size), actions_max]*dones_batch

        loss = F.mse_loss(q_values, q_targets)
        writer.add_scalar('loss', loss.item(), it)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def update_q_network_target(self):
        """
        Description
        -------------
        Update the weights of the q network.
        
        Arguments
        -------------
        batch_size : Int, the batch size.
        
        Returns
        --------------
        """

        self.q_network_target.load_state_dict(self.q_network.state_dict())

    def update_epsilon(self, epsilon_start, epsilon_stop, decay_rate, it):
        """
        Description
        --------------
        Update the epsilon parameter of the epsilon-greedy policy.
        
        Arguments
        --------------
        
        Returns
        --------------
        """

        return epsilon_stop + (epsilon_start - epsilon_stop)*np.exp(-decay_rate*it)

    def unroll(self, epsilon_start, epsilon_stop, epsilon, decay_rate, it, n_learn, tau, batch_size, optimizer, writer):
        """
        Description
        --------------
        Unroll an episode.
        
        Arguments
        --------------
        
        Returns
        --------------
        """

        state, _ = self.env.reset()
        done = False
        reward_episode = 0
        while not done:
            action = self.action_explore(state, epsilon)
            it += 1
            epsilon = self.update_epsilon(epsilon_start, epsilon_stop, decay_rate, it)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            reward_episode += reward
            done = (terminated or truncated)
            self.buffer.add(state, action, reward, next_state, 1 - done)
            state = next_state
            
            # Learning phase
            if it%n_learn == 0: self.update_q_network(batch_size, optimizer, writer, it)

            # Target network update phase
            if it%tau == 0: self.update_q_network_target()

        return reward_episode, epsilon, it
    
    def pretrain(self, n_episodes, pretrain=100):
        """
        Description
        --------------
        Pre-fill the replay buffer by running a pretraining phase where actions are taken uniformly at random.
        
        Arguments
        --------------
        n_episodes : Int, number of episodes in the pretraining phase.
        
        Returns
        --------------
        """
        
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = (terminated or truncated)
                self.buffer.add(state, action, reward, next_state, 1 - done)
                state = next_state
                
            if episode%pretrain == 0:
                print('pretrain : %d' %episode)
                
    def train(self, n_episodes=1000, n_pretrain=100, epsilon_start=1, epsilon_stop=0.01, decay_rate=1e-3, 
              n_learn=5, tau=50, batch_size=32, lr=1e-4, thresh=250, file_save='dqn.pth', log_dir='runs/', print_iter=10):
        """
        Description
        --------------
        Train the agent.
        
        Arguments
        --------------
        n_episodes    : Int, number of episodes in training phase.
        n_pretrain    : Int, number of episodes in the pretraining phase.
        epsilon_start : Float in [0, 1], initial value of epsilon.
        epsilon_stop  : Float in [0, 1], final value of epsilon.
        decay_rate    : Float in [0, 1], decay rate of epsilon.
        n_learn       : Int, number of iterations between two consecutive updates of the main network.
        tau           : Int, number of iterations between two consecutive updates of the target network.
        batch_size    : Int, size of the batch to sample from the replay buffer.
        lr            : Float, learning rate.
        max_tau       : Int, number of iterations between two consecutive updates of the target network.
        thresh        : Float, lower bound on the average of the last 10 training episodes above which early stopping is activated.
        file_save     : String, name of the file containingt the saved network weights.
        print_iter    : Int, number of episodes between two consecutive prints.
        
        Returns
        --------------
        """

        optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        writer = SummaryWriter(log_dir=log_dir)

        # Pretraining.
        self.pretrain(n_pretrain)
        epsilon, it = epsilon_start, 0

        # Training.
        rewards = deque(maxlen=100)
        reward_mean = None
        for episode in range(n_episodes):
            reward_episode, epsilon, it = self.unroll(epsilon_start, epsilon_stop, epsilon, decay_rate, it, n_learn, tau, batch_size, optimizer, writer)
            writer.add_scalar('return', reward_episode, episode)
            if len(rewards) < rewards.maxlen:
                rewards.append(reward_episode)

            else:
                reward_mean = np.mean(rewards) if reward_mean is None else reward_mean + (reward_episode - rewards[0])/rewards.maxlen
                rewards.append(reward_episode)
                if reward_mean >= thresh:
                    print('Early stopping achieved after %d episodes' %episode)
                    self.save_weights(file_save)
                    return
                
            if episode%print_iter == 0 and reward_mean is not None:
                print('Episode : %d, epsilon : %.3f, return : %.3F' %(episode, epsilon, reward_mean))
            
        print('Training finished without early stopping.')
        self.save_weights(file_save)
                    
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
        print('mean : %.3f, std : %.3f' %(return_avg, return_std))
        return return_avg, return_std
            
    def save_gif(self, env, file_name, n_episodes=1, duration=40):
        """
        Description
        --------------
        Test the agent and save a gif.
        
        Arguments
        --------------
        env        : gym environment.
        file_name  : String, path where to save the test gifs.
        n_episodes : Int, number of test episodes.
        
        Returns
        --------------
        """
        
        frames = []
        for i in range(n_episodes):
            state, _ = env.reset()
            done = False
            while not done:
                frames.append(Image.fromarray(env.render(), mode='RGB'))
                action = self.action(state)
                next_state, _, terminated, truncated, _ = env.step(action)
                done = (terminated or truncated)
                state = next_state
        
        frames[0].save(file_name, save_all=True, append_images=frames[1:], optimize=True, duration=duration, loop=0)
        
    def save_weights(self, path):
        """
        Description
        --------------
        Save the weights of the main network.
        
        Parameters
        --------------
        path: String, path to a .pth file containing the weights of the main network.
        
        Returns
        --------------
        """
        
        torch.save(self.q_network.state_dict(), path)
    
    def load_weights(self, path):
        """
        Description
        --------------
        Load stored weights onto the main network.
        
        Parameters
        --------------
        path: String, path to a .pth file containing the weights of the main network.
        
        Returns
        --------------
        """
        
        self.q_network.load_state_dict(torch.load(path))