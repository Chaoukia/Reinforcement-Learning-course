import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from threading import BrokenBarrierError
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from reinforcement_learning_course.core import Agent
from gymnasium import Env


class A2CWorker(Agent[np.array, int]):

    def __init__(self, 
                 env: Env[np.array, int], 
                 n_workers: int, 
                 gamma: float = 0.99
                 ) -> None:
        """Initialize the Actor-Critic agent.
        
        Args:
            env: Gymnasium environment with numpy array observations and integer actions.
            gamma: Discount factor. Defaults to 0.99.
        
        Attributes:
            policy_network: Neural network representing the actor (policy).
            value_network: Neural network representing the critic (value function).
        """

        super().__init__(env, gamma)
        self.n_workers = n_workers
        self.policy_network, self.value_network = self.make_networks()

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """
        """

        raise NotImplementedError
    
    def action_explore(self, state: np.array) -> tuple[int, torch.tensor, torch.tensor]:
        """Sample an action from the policy with entropy information.
        
        Used during training to enable both exploration and entropy regularization.
        
        Args:
            state: NumPy array representing the current state.
        
        Returns:
            A tuple containing:
                - action: Integer action sampled from the policy.
                - action_logprob: Log probability of the sampled action.
                - entropy: Entropy of the policy distribution.
        """

        logits = self.policy_network(torch.from_numpy(state))
        dist = Categorical(logits=logits)
        entropy = dist.entropy()
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob, entropy
    
    def action(self, state: np.array) -> int:
        """Select an action deterministically from the policy.
        
        Used during testing/evaluation to get the best action without sampling.
        
        Args:
            state: NumPy array representing the current state.
        
        Returns:
            Integer action sampled from the policy distribution.
        """

        with torch.no_grad():
            logits = self.policy_network(torch.from_numpy(state))
            dist = Categorical(logits=logits)
            action = dist.sample()
            return action.item()
        
    def unroll(self, state: np.array, t_max: int) -> tuple[list[np.array], list[torch.tensor], list[float], list[torch.tensor], float, bool]:
        """Collect a trajectory of length up to t_max steps.
        
        Runs the policy for up to t_max steps or until a terminal state,
        collecting states, actions, rewards and policy information.
        
        Args:
            state: Starting state.
            t_max: Maximum number of steps before truncating the episode.
        
        Returns:
            A tuple containing:
                - states: List of visited states.
                - actions_logprobs: List of log probabilities of sampled actions.
                - rewards: List of rewards received.
                - entropies: List of policy entropies at each step.
                - episode_return: Cumulative reward from this rollout.
                - done: Whether the episode terminated.
        """

        states, actions_logprobs, rewards, entropies = [state], [], [], []
        done = False
        R = 0
        t = 0
        while not done and t < t_max:
            action, action_logprob, entropy = self.action_explore(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            R += reward
            done = (terminated or truncated)
            state = next_state
            actions_logprobs.append(action_logprob)
            rewards.append(reward)
            entropies.append(entropy)
            states.append(state)
            t += 1

        return states, actions_logprobs, rewards, entropies, R, done
    
    def backward(self, 
                states: list[np.array], 
                actions_logprobs: list[torch.tensor], 
                rewards: list[float], 
                entropies: list[torch.tensor], 
                done: bool,
                alpha_entropy: float, 
                optimizer_policy: optim, 
                optimizer_value: optim
                ) -> tuple[float, float, float, float]:
        """Update both policy and value networks using a collected trajectory.
        
        Computes advantage estimates and performs gradient updates for both networks.
        
        Args:
            states: List of states visited in the trajectory.
            actions_logprobs: Log probabilities of sampled actions.
            rewards: Rewards received at each step.
            entropies: Policy entropies at each step.
            done: Whether the episode terminated or was truncated.
            alpha_entropy: Coefficient for entropy regularization term.
            optimizer_policy: Optimizer for policy network.
            optimizer_value: Optimizer for value network.
        
        Returns:
            A tuple containing:
                - loss_policy: Loss for policy gradient term.
                - loss_entropy: Loss for entropy regularization.
                - loss_actor: Total loss (policy + entropy).
                - loss_critic: Loss for value function.
        """

        if done:
            R = 0

        else:
            with torch.no_grad():
                R = self.value_network(torch.from_numpy(states[-1]))
                
        optimizer_policy.zero_grad()
        optimizer_value.zero_grad()
        loss_policy, loss_entropy, loss_critic = 0, 0, 0
        for t in range(len(states)-2, -1, -1):
            R = rewards[t] + self.gamma*R
            advantage = R - self.value_network(torch.from_numpy(states[t]))
            loss_policy += -self.gamma**t*advantage.detach()*actions_logprobs[t]
            loss_entropy += -entropies[t]
            loss_critic += advantage**2

        loss_policy /= (len(states) - 1)*self.n_workers
        loss_entropy /= (len(states) - 1)*self.n_workers
        loss_actor = loss_policy + alpha_entropy*loss_entropy
        loss_critic /= (len(states) - 1)*self.n_workers
        loss_actor.backward()
        loss_critic.backward()

        return loss_policy.item(), loss_entropy.item(), loss_actor.item(), loss_critic.item()
    

def share_adam_optimizer(optimizer):
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state[param]
            state["step"] = torch.tensor(0.0).share_memory_()
            state["exp_avg"] = torch.zeros_like(param).share_memory_()
            state["exp_avg_sq"] = torch.zeros_like(param).share_memory_()


def train_a2c_worker(worker_id, 
                     policy_network,
                     value_network, 
                     policy_optimizer, 
                     value_optimizer, 
                     env,
                     agent,
                     t_max,
                     n_episodes,
                     alpha_entropy,
                     thresh,
                     print_iter,
                     log_dir,
                     barrier, 
                     lock, 
                     ) -> None:
    
    agent.policy_network = policy_network
    agent.value_network = value_network
    writer = SummaryWriter(log_dir=log_dir)
    rewards_episodes = deque(maxlen=100)
    reward_mean = None
    it = 0
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        reward_episode = 0
        while not done:
            # The worker must wait for the other workers to update the global parameters before moving to the next iteration
            try:
                barrier.wait()

            except BrokenBarrierError:
                print(f"\nWorker {worker_id}: Barrier was broken, likely due to another worker finishing earlier.")
                return # Stop training because one worker finished early
            
            states, actions_logprobs, rewards, entropies, R, done = agent.unroll(state, t_max)
            reward_episode += R
            loss_policy, loss_entropy, loss_actor, loss_critic = agent.backward(
                states, 
                actions_logprobs, 
                rewards, 
                entropies, 
                done,
                alpha_entropy, 
                policy_optimizer, 
                value_optimizer,
            ) 
            state = states[-1]
            writer.add_scalar(f'worker_{worker_id}/loss_policy', loss_policy, it)
            writer.add_scalar(f'worker_{worker_id}/loss_entropy', loss_entropy, it)
            writer.add_scalar(f'worker_{worker_id}/loss_actor', loss_actor, it)
            writer.add_scalar(f'worker_{worker_id}/loss_critic', loss_critic, it)
            it += 1
            # Update the global parameters
            with lock:
                policy_optimizer.step()
                value_optimizer.step()

        writer.add_scalar(f'worker_{worker_id}/return', reward_episode, episode)
        if len(rewards_episodes) < rewards_episodes.maxlen:
            rewards_episodes.append(reward_episode)

        else:
            reward_mean = np.mean(rewards_episodes) if reward_mean is None else reward_mean + (reward_episode - rewards_episodes[0])/rewards_episodes.maxlen
            rewards_episodes.append(reward_episode)
            if reward_mean >= thresh:
                print(f"\nWorker {worker_id} achieved early stopping achieved after {episode} episodes")
                barrier.abort()     # Break the barrier because the worker finished
                return
            
        if episode%print_iter == 0 and reward_mean is not None and worker_id == 0:
            print('Worker : %d , episode : %d , return : %.3F' %(worker_id, episode, reward_mean))
        
    print(f'\nWorker {worker_id} finished training without early stopping.')
    barrier.abort()     # Break the barrier because the worker finished
