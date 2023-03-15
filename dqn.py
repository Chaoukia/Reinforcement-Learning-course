import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from utils import *
from torch.utils.tensorboard import SummaryWriter
from PIL import Image


class DQN:
    
    """
    Class of the DQN algorithm.
    """
    
    def __init__(self, gamma=0.99, input_size=8, out=4, max_size=500000):
        """
        Description
        -------------------------
        Constructor of class DQN.
        
        Arguments and Attributes
        -------------------------
        gamma      : Float in [0, 1] generally very close to 1, discount factor.
        input_size : Int, dimension of the state space.
        out        : Int, output dimension, equal to the number of possible actions.
        max_size   : Int, maximum size of the replay memory.
        """
        
        self.gamma = gamma
        self.q_network = DQNetwork(input_size=input_size, out=out)
        self.q_network_target = DQNetwork(input_size=input_size, out=out)
        self.buffer = Memory(max_size=max_size)
        
    def action_explore(self, env, state, epsilon):
        """
        Description
        --------------
        Take an action according to the epsilon-greedy policy.
        
        Arguments
        --------------
        env     : gym environment.
        state   : np.array, state from a gym environment.
        epsilon : Float in [0, 1], parameter of the epsilon-greedy policy.
        
        Returns
        --------------
        action : Int, action to take.
        """
        
        bern = np.random.binomial(1, epsilon)
        if bern == 1:
            action = env.action_space.sample()

        else:
            with torch.no_grad():
                q_values = self.q_network(torch.from_numpy(state))
                action = torch.argmax(q_values).item()

        return action

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
        
    def update_target(self):
        """
        Description
        -------------
        Update the weights of the target network with those of the main network.
        
        Arguments
        -------------
        
        Returns
        --------------
        """
        
        self.q_network_target.load_state_dict(self.q_network.state_dict())
    
    def pretrain(self, env, n_episodes):
        """
        Description
        --------------
        Pre-fill the replay buffer by running a pretraining phase where actions are taken uniformly at random.
        
        Arguments
        --------------
        env        : gym environment.
        n_episodes : Int, number of episodes in the pretraining phase.
        
        Returns
        --------------
        """
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = (terminated or truncated)
                self.buffer.add(state, action, reward, next_state, 1 - done)
                state = next_state
                
            if episode%10 == 0:
                print('pretrain : %d' %episode)
                
    def train(self, env, n_episodes=1000, n_pretrain=100, epsilon_start=1, epsilon_stop=0.01, decay_rate=1e-3, n_learn=5, batch_size=32, lr=1e-4, max_tau=50, log_dir='runs/', 
             thresh=250, file_save='dqn'):
        """
        Description
        --------------
        Train the agent.
        
        Arguments
        --------------
        env           : gym environment.
        n_episodes    : Int, number of episodes in training phase.
        n_pretrain    : Int, number of episodes in the pretraining phase.
        epsilon_start : Float in [0, 1], initial value of epsilon.
        epsilon_stop  : Float in [0, 1], final value of epsilon.
        decay_rate    : Float in [0, 1], decay rate of epsilon.
        n_learn       : Int, number of iterations between two consecutive updates of the main network.
        batch_size    : Int, size of the batch to sample from the replay buffer.
        lr            : Float, learning rate.
        max_tau       : Int, number of iterations between two consecutive updates of the target network.
        log_dir       : String, path of the folder where to store tensorboard events.
        thresh        : Float, lower bound on the average of the last 10 training episodes above which early stopping is activated.
        file_save     : String, name of the file containingt the saved network weights.
        
        Returns
        --------------
        """

        # Pretraining.
        writer = SummaryWriter(log_dir=log_dir)
        variables = {'loss' : deque(maxlen=100), 'return' : deque(maxlen=10)}
        self.pretrain(env, n_pretrain)
        epsilon = epsilon_start
        it = 0
        optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        tau = 0
        # Training.
        for episode in range(n_episodes):
            state, _ = env.reset()
            done = False
            R = 0
            n_steps = 0
            while not done:
                action = self.action_explore(env, state, epsilon)
                it += 1
                tau += 1
                epsilon = epsilon_stop + (epsilon_start - epsilon_stop)*np.exp(-decay_rate*it)
                next_state, reward, terminated, truncated, _ = env.step(action)
                R += reward
                done = (terminated or truncated)
                self.buffer.add(state, action, reward, next_state, 1 - done)
                state = next_state
                n_steps += 1
                
                # Learning phase
                if it%n_learn == 0:
                    batch = self.buffer.sample(batch_size)
                    states_batch = torch.cat(batch.state)
                    actions_batch = np.concatenate(batch.action)
                    rewards_batch = torch.cat(batch.reward)
                    next_states_batch = torch.cat(batch.next_state)
                    dones_batch = torch.cat(batch.done)
                    with torch.no_grad():
                        q_values_next_states = self.q_network_target(next_states_batch)
                        q_targets = rewards_batch + self.gamma*torch.max(q_values_next_states, dim=1).values*dones_batch

                    q_values = self.q_network(states_batch)[np.arange(batch_size), actions_batch]
                    loss = F.mse_loss(q_values, q_targets)
                    variables['loss'].append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if it%(10*n_learn) == 0:
                        writer.add_scalar('Loss', np.mean(variables['loss']), it)

                # Update the weights of the target network.
                if tau == max_tau:
                    self.update_target()
                    tau = 0
                    
            variables['return'].append(R)
            R_mean = np.mean(variables['return'])
            writer.add_scalar('return', R_mean, episode)                   
            if episode%1 == 0:
                print('Episode : %d, epsilon : %.3f, length : %d, return : %.3F' %(episode, epsilon, n_steps, R))
                
            if R_mean > thresh:
                print('Early stopping achieved after %d episodes' %episode)
                self.save_weights(file_save + '.pth')
                break
                    
    def test(self, env, n_episodes=10):
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
            
    def save_gif(self, env, file_name='lunar-lander.gif'):
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