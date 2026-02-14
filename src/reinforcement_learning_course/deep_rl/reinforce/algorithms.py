import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from reinforcement_learning_course.core import Agent
from gymnasium import Env


class Reinforce(Agent[np.array, int]):
    """
    Reinforce agent.
    """

    def __init__(self, env: Env[np.array, int], gamma: float = 0.99) -> None:
        """
        Description
        -------------------------------
        Constructor.

        Parameters
        -------------------------------
        env   : gymnasium environment.
        gamma : Float, discount factor.

        Returns
        -------------------------------
        """
        super().__init__(env, gamma)
        self.policy_network = self.make_networks()

    def make_networks(self) -> nn.Module:
        """
        Description
        -------------------------------
        Initialize the policy network.

        Parameters
        -------------------------------

        Returns
        -------------------------------
        """

        raise NotImplementedError
    
    def action_explore(self, state: np.array) -> tuple[int, torch.tensor, torch.tensor]:
        """
        Description
        -------------------
        Choose an action according to the policy network and return the necessary elements for training. To use when training.

        Parameters
        -------------------
        state : np.array, a state.

        Returns
        -------------------
        action         : Int, action taken by the policy.
        action_logprob : torch.tensor, logit of the performed action.
        entropy        : torch.tensor, entropy of the current policy.
        """

        logits = self.policy_network(torch.from_numpy(state))
        dist = Categorical(logits=logits)
        entropy = dist.entropy()
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob, entropy
    
    def action(self, state: np.array) -> int:
        """
        Description
        -------------------
        Choose an action according to the policy network. To use when testing.

        Parameters
        -------------------
        state : np.array, a state.

        Returns
        -------------------
        action         : Int, action taken by the policy.
        """

        with torch.no_grad():
            logits = self.policy_network(torch.from_numpy(state))
            dist = Categorical(logits=logits)
            action = dist.sample()
            return action.item()
        
    def unroll(self) -> tuple[list[np.array], list[torch.tensor], list[float], list[torch.tensor], float]:
        """
        Description
        -------------------
        Unroll an episode with the current policy.

        Parameters
        -------------------

        Returns
        -------------------
        states           : List of visited states.
        actions_logprobs : List of logits of the performed actions.
        rewards          : List of received rewards.
        entropies        : List of entropies of the policy evaluated at the visited states.
        reward_episode   : Float, return of the episode.
        """

        state, _ = self.env.reset()
        states, actions_logprobs, rewards, entropies = [state], [], [], []
        done = False
        reward_episode = 0
        while not done:
            action, action_logprob, entropy = self.action_explore(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            reward_episode += reward
            done = (terminated or truncated)
            state = next_state
            actions_logprobs.append(action_logprob)
            rewards.append(reward)
            entropies.append(entropy)
            states.append(state)

        return states, actions_logprobs, rewards, entropies, reward_episode
    
    def learn(self, 
              states: list[np.array], 
              actions_logprobs: list[torch.tensor], 
              rewards: list[float], 
              entropies: list[torch.tensor], 
              alpha_entropy: float, 
              optimizer: optim
              ) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
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
        loss_policy  : Float, the loss term corresponding to improving the policy.
        loss_entropy : Float, the loss term corresponding to maximising the entropy of the policy.
        loss         : Float, the global loss term.
        """

        R = 0
        optimizer.zero_grad()
        loss_policy, loss_entropy = 0, 0
        for t in range(len(states)-2, -1, -1):
            R = rewards[t] + self.gamma*R
            loss_policy += -self.gamma**t*R*actions_logprobs[t]
            loss_entropy += -entropies[t]

        loss_policy /= len(states) - 1
        loss_entropy /= len(states) - 1
        loss = loss_policy + alpha_entropy*loss_entropy
        loss.backward()
        optimizer.step()
        return loss_policy.item(), loss_entropy.item(), loss.item()

    def train(self, 
              n_episodes: int = 1000, 
              lr: float = 1e-4, 
              alpha_entropy: float = 0.1, 
              thresh: float = 250, 
              log_dir: str = 'runs/', 
              print_iter: int = 10
              ) -> None:
        """
        Description
        --------------
        Train the agent.
        
        Arguments
        --------------
        n_episodes    : Int, number of episodes in training phase.
        lr            : Float, learning rate.
        thresh        : Float, lower bound on the average of the last 10 training episodes above which early stopping is activated.
        file_save     : String, name of the file containingt the saved network weights.
        print_iter    : Int, number of episodes between two consecutive prints.
        
        Returns
        --------------
        """

        optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        writer = SummaryWriter(log_dir=log_dir)
        rewards_episodes = deque(maxlen=100)
        reward_mean = None
        for episode in range(n_episodes):
            states, actions_logprobs, rewards, entropies, reward_episode = self.unroll()
            loss_policy, loss_entropy, loss = self.learn(states, actions_logprobs, rewards, entropies, alpha_entropy, optimizer)
            writer.add_scalar('loss_policy', loss_policy, episode)
            writer.add_scalar('loss_entropy', loss_entropy, episode)
            writer.add_scalar('loss', loss, episode)
            writer.add_scalar('return', reward_episode, episode)
            if len(rewards_episodes) < rewards_episodes.maxlen:
                rewards_episodes.append(reward_episode)

            else:
                reward_mean = np.mean(rewards_episodes) if reward_mean is None else reward_mean + (reward_episode - rewards_episodes[0])/rewards_episodes.maxlen
                rewards_episodes.append(reward_episode)
                if reward_mean >= thresh:
                    print('Early stopping achieved after %d episodes' %episode)
                    return
                
            if episode%print_iter == 0 and reward_mean is not None:
                print('Episode : %d, return : %.3F' %(episode, reward_mean))
            
        print('Training finished without early stopping.')

    def save(self, path: str) -> None:
        """
        Description
        --------------
        Save the weights of the policy network.
        
        Parameters
        --------------
        path: String, path to the weights of the policy network.
        
        Returns
        --------------
        """
        
        torch.save(self.policy_network.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Description
        --------------
        Load the weights of the policy network.
        
        Parameters
        --------------
        path: String, path to the weights of the policy network.
        
        Returns
        --------------
        """
        
        self.policy_network.load_state_dict(torch.load(path))
        
