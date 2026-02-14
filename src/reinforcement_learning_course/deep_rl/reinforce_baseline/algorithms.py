import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from reinforcement_learning_course.core import Agent
from gymnasium import Env


class ReinforceBaseline(Agent[np.array, int]):
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
        self.policy_network, self.value_network = self.make_networks()

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """
        Description
        -------------------------------
        Initialize the policy network and value networks.

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
        action : Int, action taken by the policy.
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
              optimizer_policy: optim, 
              optimizer_value: optim
              ) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
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
        loss_value   : Float, the loss term for the value.
        """

        R = 0
        optimizer_policy.zero_grad()
        optimizer_value.zero_grad()
        loss_policy, loss_entropy, loss_value = 0, 0, 0
        for t in range(len(states)-2, -1, -1):
            R = rewards[t] + self.gamma*R
            delta = R - self.value_network(torch.from_numpy(states[t]))
            loss_policy += -self.gamma**t*delta.detach()*actions_logprobs[t]
            loss_entropy += -entropies[t]
            loss_value += delta**2
            
        loss_policy /= len(states) - 1
        loss_entropy /= len(states) - 1
        loss = loss_policy + alpha_entropy*loss_entropy
        loss_value /= len(states) - 1
        loss.backward()
        optimizer_policy.step()
        loss_value.backward()
        optimizer_value.step()
        return loss_policy.item(), loss_entropy.item(), loss.item(), loss_value.item()

    def train(self, 
              n_episodes: int = 1000, 
              lr_policy: float = 1e-4, 
              lr_value: float = 1e-4, 
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
        n_episodes : Int, number of episodes in training phase.
        lr_policy  : Float, learning rate for the policy network.
        lr_value   : Float, learning rate for the value network.
        thresh     : Float, lower bound on the average of the last 10 training episodes above which early stopping is activated.
        file_save  : String, name of the file containingt the saved network weights.
        print_iter : Int, number of episodes between two consecutive prints.
        
        Returns
        --------------
        """

        optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=lr_policy)
        optimizer_value = optim.Adam(self.value_network.parameters(), lr=lr_value)
        writer = SummaryWriter(log_dir=log_dir)
        rewards_episodes = deque(maxlen=100)
        reward_mean = None
        for episode in range(n_episodes):
            states, actions_logprobs, rewards, entropies, reward_episode = self.unroll()
            loss_policy, loss_entropy, loss, loss_value = self.learn(states, actions_logprobs, rewards, entropies, alpha_entropy, optimizer_policy, optimizer_value)
            writer.add_scalar('Policy/loss_policy', loss_policy, episode)
            writer.add_scalar('Policy/loss_entropy', loss_entropy, episode)
            writer.add_scalar('Policy/loss', loss, episode)
            writer.add_scalar('Value/loss', loss_value, episode)
            writer.add_scalar('Return/return', reward_episode, episode)
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

    def save(self, path: str | Path) -> None:
        """
        Description
        --------------
        Save the weights of the policy and value networks.
        
        Parameters
        --------------
        path : String or Path, path to the directory storing the weights of the policy and value networks.
        
        Returns
        --------------
        """

        os.makedirs(str(path), exist_ok=True)
        path = Path(path)
        path_policy = path / "policy.pt"
        path_value = path / "value.pt"
        torch.save(self.policy_network.state_dict(), path_policy)
        torch.save(self.policy_network.state_dict(), path_value)

    def load(self, path: str | Path) -> None:
        """
        Description
        --------------
        Load the weights of the policy and value networks.
        
        Parameters
        --------------
        path : String or Path, path to the directory storing the weights of the policy and value networks.
        
        Returns
        --------------
        """

        if not os.path.isdir(str(path)):
            raise ValueError(f"{str(path)} is not a directory")
        
        path = Path(path)
        path_policy = path / "policy.pt"
        path_value = path / "value.pt"
        self.policy_network.load_state_dict(torch.load(path_policy))
        self.value_network.load_state_dict(torch.load(path_value))
        
