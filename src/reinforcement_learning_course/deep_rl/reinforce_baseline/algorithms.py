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
    """REINFORCE with baseline (advantage) learning agent.
    
    Implements REINFORCE algorithm with learned value baseline to reduce
    variance in policy gradient estimates.
    """

    def __init__(self, env: Env[np.array, int], gamma: float = 0.99) -> None:
        """Initialize the REINFORCE Baseline agent.
        
        Args:
            env: Gymnasium environment wrapper.
            gamma: Discount factor. Defaults to 0.99.
        """
        super().__init__(env, gamma)
        self.policy_network, self.value_network = self.make_networks()

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Initialize the policy and value networks.
        
        Must be implemented by subclasses with appropriate architectures.
        
        Returns:
            Tuple of (policy_network, value_network) modules.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError
    
    def action_explore(self, state: np.array) -> tuple[int, torch.tensor, torch.tensor]:
        """Select action during training with policy exploration.
        
        Samples action from policy distribution and returns required training information.
        
        Args:
            state: Current observation as numpy array.
        
        Returns:
            action: Integer action sampled from policy.
            action_logprob: Log probability of action under current policy.
            entropy: Entropy of policy distribution at this state.
        """

        logits = self.policy_network(torch.from_numpy(state))
        dist = Categorical(logits=logits)
        entropy = dist.entropy()
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob, entropy
    
    def action(self, state: np.array) -> int:
        """Select action during testing using learned policy.
        
        Args:
            state: Current observation as numpy array.
        
        Returns:
            Integer action sampled from policy (greedy without exploration noise).
        """

        with torch.no_grad():
            logits = self.policy_network(torch.from_numpy(state))
            dist = Categorical(logits=logits)
            action = dist.sample()
            return action.item()
        
    def unroll(self) -> tuple[list[np.array], list[torch.tensor], list[float], list[torch.tensor], float]:
        """Execute one episode with current policy.
        
        Collects trajectory data needed for policy gradient computation.
        
        Returns:
            states: List of observed states throughout episode.
            actions_logprobs: List of log probabilities for taken actions.
            rewards: List of immediate rewards received.
            entropies: List of policy entropies at each state.
            reward_episode: Total discounted return for the episode.
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
        """Update policy and value networks using REINFORCE with baseline.
        
        Updates policy using advantage estimates from baseline, and trains
        value network as baseline function.
        
        Args:
            states: List of states visited in episode.
            actions_logprobs: Log probabilities of actions taken.
            rewards: Rewards received in episode.
            entropies: Policy entropies at each state.
            alpha_entropy: Weight of entropy regularization term.
            optimizer_policy: PyTorch optimizer for policy network.
            optimizer_value: PyTorch optimizer for value network.
        
        Returns:
            loss_policy: Policy gradient loss value.
            loss_entropy: Entropy regularization loss value.
            loss: Total policy loss (policy + entropy).
            loss_value: Value network (baseline) loss value.
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
        """Train the REINFORCEBaseline agent.
        
        Runs multiple episodes of training with periodic evaluation and early stopping.
        
        Args:
            n_episodes: Maximum number of training episodes. Defaults to 1000.
            lr_policy: Learning rate for policy optimizer. Defaults to 1e-4.
            lr_value: Learning rate for value network optimizer. Defaults to 1e-4.
            alpha_entropy: Weight of entropy regularization. Defaults to 0.1.
            thresh: Target average return for early stopping. Defaults to 250.
            log_dir: Directory for TensorBoard logs. Defaults to 'runs/'.
            print_iter: Episodes between progress prints. Defaults to 10.
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
        """Save policy and value network weights to disk.
        
        Args:
            path: Directory path where network weights will be saved.
        """

        os.makedirs(str(path), exist_ok=True)
        path = Path(path)
        path_policy = path / "policy.pt"
        path_value = path / "value.pt"
        torch.save(self.policy_network.state_dict(), path_policy)
        torch.save(self.value_network.state_dict(), path_value)

    def load(self, path: str | Path) -> None:
        """Load policy and value network weights from disk.
        
        Args:
            path: Directory path containing saved network weights.
        """

        if not os.path.isdir(str(path)):
            raise ValueError(f"{str(path)} is not a directory")
        
        path = Path(path)
        path_policy = path / "policy.pt"
        path_value = path / "value.pt"
        self.policy_network.load_state_dict(torch.load(path_policy))
        self.value_network.load_state_dict(torch.load(path_value))
        
