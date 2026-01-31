import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from PIL import Image

class ReinforceBaseline:
    """
    Class of the Reinforce algorithm.
    """

    def __init__(self, gamma=0.99):
        """
        Description
        -------------------
        Constructor of class Reinforce.

        Parameters & Attributes
        -------------------
        gamma : Float in [0, 1] generally very close to 1, discount factor.
        """

        self.gamma = gamma
        self.policy_value_network = None
        self.output_dim = None

    def action(self, state, train=True):
        """
        Description
        -------------------
        Constructor of class Reinforce.

        Parameters
        -------------------
        state : np.array, a state.
        train : Boolean, whether to calculate gradients during the forward propagation through the policy network or not.

        Returns
        -------------------
        action         : Int, action taken by the policy.
        action_logprob : torch.tensor, logit of the performed action.
        entropy        : torch.tensor, entropy of the current policy.
        """

        if train:
            logits, value = self.policy_value_network(torch.from_numpy(state))

        else:
            with torch.no_grad():
                logits, value = self.policy_value_network(torch.from_numpy(state))
             
        dist = Categorical(logits=logits)
        entropy = dist.entropy()
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob, entropy, value
        
    def unroll(self, env, train=True):
        """
        Description
        -------------------
        Unroll an episode with the current policy.

        Parameters
        -------------------
        env   : gymnasium environment for training.
        train : Boolean, whether to calculate gradients during the forward propagation through the policy network or not.

        Returns
        -------------------
        states           : List of visited states.
        actions_logprobs : List of logits of the performed actions.
        rewards          : List of received rewards.
        entropies        : List of entropies of the policy evaluated at the visited states.
        reward_episode   : Float, return of the episode.
        """

        state, _ = env.reset()
        states, actions_logprobs, rewards, entropies, values = [state], [], [], [], []
        done = False
        reward_episode = 0
        while not done:
            action, action_logprob, entropy, value = self.action(state, train)
            next_state, reward, terminated, truncated, _ = env.step(action)
            reward_episode += reward
            done = (terminated or truncated)
            state = next_state
            actions_logprobs.append(action_logprob)
            rewards.append(reward)
            entropies.append(entropy)
            values.append(value)
            states.append(state)

        return states[:-1], actions_logprobs, rewards, entropies, reward_episode, values
    
    def learn(self, states, actions_logprobs, rewards, entropies, values, alpha_entropy, optimizer):
        """
        Description
        -------------------
        Update the policy network parameters.

        Parameters
        -------------------
        states           : List of visited states.
        actions_logprobs : List of logits of the performed actions.
        rewards          : List of received rewards.
        entropies        : List of entropies of the policy evaluated at the visited states.
        optimizer        : torch.optim.Optimizer object that updates the network parameters.

        Returns
        -------------------
        loss_policy  : torch.tensor, the loss term corresponding to improving the policy.
        loss_entropy : torch.tensor, the loss term corresponding to maximising the entropy of the policy.
        loss         : torch.tensor, the global loss term.
        """

        R = 0
        optimizer.zero_grad()
        for t in range(len(states)-1, -1, -1):
            R = rewards[t] + self.gamma*R
            advantage = (R - values[t])
            loss_policy = -self.gamma**t*advantage.detach()*actions_logprobs[t]
            loss_entropy = -entropies[t]
            loss_value = advantage**2
            loss = loss_policy + loss_value + alpha_entropy*loss_entropy
            loss.backward()

        optimizer.step()
        return loss_policy, loss_entropy, loss_value, loss

    def train(self, env, n_episodes=1000, lr=1e-4, alpha_entropy=0.1, thresh=250, file_save='dqn.pth', log_dir='runs/', print_iter=10):
        """
        Description
        --------------
        Train the agent.
        
        Arguments
        --------------
        env           : gymnasium environment for training.
        n_episodes    : Int, number of episodes in training phase.
        lr            : Float, learning rate.
        thresh        : Float, lower bound on the average of the last 10 training episodes above which early stopping is activated.
        file_save     : String, name of the file containingt the saved network weights.
        print_iter    : Int, number of episodes between two consecutive prints.
        
        Returns
        --------------
        """

        optimizer = optim.Adam(self.policy_value_network.parameters(), lr=lr)
        writer = SummaryWriter(log_dir=log_dir)
        rewards_episodes = deque(maxlen=100)
        reward_mean = None
        for episode in range(n_episodes):
            states, actions_logprobs, rewards, entropies, reward_episode, values = self.unroll(env)
            loss_policy, loss_entropy, loss_value, loss = self.learn(states, actions_logprobs, rewards, entropies, values, alpha_entropy, optimizer)
            writer.add_scalar('loss_policy', loss_policy, episode)
            writer.add_scalar('loss_entropy', loss_entropy, episode)
            writer.add_scalar('loss_value', loss_value, episode)
            writer.add_scalar('loss', loss, episode)
            writer.add_scalar('return', reward_episode, episode)
            if len(rewards_episodes) < rewards_episodes.maxlen:
                rewards_episodes.append(reward_episode)

            else:
                reward_mean = np.mean(rewards_episodes) if reward_mean is None else reward_mean + (reward_episode - rewards_episodes[0])/rewards_episodes.maxlen
                rewards_episodes.append(reward_episode)
                if reward_mean >= thresh:
                    print('Early stopping achieved after %d episodes' %episode)
                    self.save_weights(file_save)
                    return
                
            if episode%print_iter == 0 and reward_mean is not None:
                print('Episode : %d, return : %.3F' %(episode, reward_mean))
            
        print('Training finished without early stopping.')
        self.save_weights(file_save)

    def test(self, env, n_episodes=1000, verbose=False):
        """
        Description
        --------------
        Test the agent.
        
        Arguments
        --------------
        env        : gymnasium environment for testing.
        n_episodes : Int, number of test episodes.
        verbose    : Boolean, if True, print the episode index and its corresponding length and return.
        
        Returns
        --------------
        """
        
        returns = np.empty(n_episodes)
        for episode in range(n_episodes):
            _, _, _, _, reward_episode, _ = self.unroll(env, train=False)
            returns[episode] = reward_episode
            if verbose:
                print('Episode : %d, return : %.3F' %(episode, reward_episode))

        return_mean, return_std = returns.mean(), returns.std()
        print('mean : %.3f, std : %.3f' %(return_mean, return_std))
        return return_mean, return_std
            
    def save_gif(self, env, file_name, n_episodes=1, duration=40):
        """
        Description
        --------------
        Test the agent and save a gif.
        
        Arguments
        --------------
        env        : gymnasium environment for testing.
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
                action, _ = self.action(state, train=False)
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
        
        torch.save(self.policy_value_network.state_dict(), path)
    
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
        
        self.policy_value_network.load_state_dict(torch.load(path))

        