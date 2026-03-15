import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from threading import BrokenBarrierError
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from reinforcement_learning_course.core import Agent
from gymnasium import Env


class PPOWorker(Agent[np.array, int]):

    def __init__(self, 
                 env: Env[np.array, int], 
                 n_workers: int, 
                 epsilon: float,
                 lambd: float,
                 gamma: float = 0.99
                 ) -> None:
        """Initialize the PPO worker.
        
        Args:
            env: Gymnasium environment with numpy array observations and integer actions.
            gamma: Discount factor. Defaults to 0.99.
        
        Attributes:
            policy_network: Neural network representing the actor (policy).
            value_network: Neural network representing the critic (value function).
        """

        super().__init__(env, gamma)
        self.n_workers = n_workers
        self.epsilon = epsilon
        self.lambd = lambd
        self.policy_network, self.value_network = self.make_networks()

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """
        """

        raise NotImplementedError
    
    def action_explore(self, state: np.array) -> tuple[int, float]:
        """Sample an action from the policy with entropy information.
        
        Used during training to enable both exploration and entropy regularization.
        
        Args:
            state: NumPy array representing the current state.
        
        Returns:
            A tuple containing:
                - action: Integer action sampled from the policy.
                - action_prob: Probability of the sampled action.
        """

        with torch.no_grad():
            logits = self.policy_network(torch.from_numpy(state))
            dist = Categorical(logits=logits)
            action = dist.sample()
            action_prob = dist.probs[action]
            return action.item(), action_prob.item()
    
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
        
    def unroll(self, state: np.array, t_max: int) -> tuple[list[np.array], list[int], list[float], list[float], float, bool]:
        """Collect a trajectory of length up to t_max steps.
        
        Runs the policy for up to t_max steps or until a terminal state,
        collecting states, actions, rewards and policy information.
        
        Args:
            state: Starting state.
            t_max: Maximum number of steps before truncating the episode.
        
        Returns:
            A tuple containing:
                - states: List of visited states.
                - actions_probs: List of log probabilities of sampled actions.
                - rewards: List of rewards received.
                - entropies: List of policy entropies at each step.
                - episode_return: Cumulative reward from this rollout.
                - done: Whether the episode terminated.
        """

        states, actions, actions_probs, rewards = [state], [], [], []
        done = False
        R = 0
        t = 0
        while not done and t < t_max:
            action, action_prob = self.action_explore(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            R += reward
            done = (terminated or truncated)
            state = next_state
            actions.append(action)
            actions_probs.append(action_prob)
            rewards.append(reward)
            states.append(state)
            t += 1

        return states, actions, actions_probs, rewards, R, done
    
    def calculate_advantages(self, 
                             shared_advantages: np.array,
                             index: int,
                             states: list[np.array], 
                             rewards: list[float],
                             done: bool,
                             ) -> tuple[list[torch.tensor], torch.tensor, int]:
        """Calculate the advantages and store them in the shared array of advantages, 
        the purpose being to calculate the mean and std of advantages.
        Return the list of advantages in tensor form, this will be used later
        to update the critic.
        Return len(states)-2, the n-step length of bootstrapping, it will be used for
        calculating the advantage mean and std.
        """
        
        # Initialize the advantages list
        advantages_list = [0 for i in range(len(states)-1)]
        advantage = 0
        # If done, do not calculate the value of the last state, it is terminal and thus its value is 0
        with torch.no_grad():
            state_values = self.value_network(torch.from_numpy(np.array(states)))

        V_next_state = 0 if done else state_values[-1].item()
        for t in range(len(states)-2, -1, -1):
            V_state = state_values[t]
            td = rewards[t] + self.gamma*V_next_state - V_state
            advantage = td + (self.gamma*self.lambd)*advantage
            shared_advantages[index, t] = advantage.item()
            advantages_list[t] = advantage
            V_next_state = V_state

        return advantages_list, state_values[:-1], len(states)-1
    
    def backward(self, 
                 states: list[np.array], 
                 actions: list[int],
                 actions_probs_old: list[float], 
                 advantages: list[torch.tensor],
                 advantage_mean, 
                 advantage_std, 
                 state_values_old: torch.tensor,
                 alpha_entropy: float,
                 lock
                 ) -> tuple[float, float, float, float]:
        """Update both policy and value networks using a collected trajectory.
        
        Computes advantage estimates and performs gradient updates for both networks.
        
        Args:
            states: List of states visited in the trajectory.
            actions_probs_old: Probabilities of the sampled actions according to the old policy.
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

        # Take the softmax along dim=1 to transform logits into probabilities.
        policy_probs = torch.softmax(self.policy_network(torch.from_numpy(np.array(states[:-1]))), dim=1)
        # Action probabilities according to the current policy
        actions_probs = policy_probs[range(policy_probs.shape[0]), actions]
        actions_probs_old_tensor = torch.from_numpy(np.array(actions_probs_old))
        ratio = actions_probs/actions_probs_old_tensor
        advantages_tensor = torch.hstack(advantages)
        advantage_normalized = (advantages_tensor - advantage_mean.value)/(advantage_std.value + 1e-8)
        loss_policy = -torch.minimum(
            ratio*advantage_normalized, torch.clip(ratio, 1-self.epsilon, 1+self.epsilon)*advantage_normalized
        ).mean()
        loss_entropy = 0
        for i in range(len(states)-1):
            dist = Categorical(logits=policy_probs[i, :])
            loss_entropy -= dist.entropy()

        loss_entropy /= (len(states)-1)
        loss_actor = loss_policy + alpha_entropy*loss_entropy
        state_values = self.value_network(torch.from_numpy(np.array(states[:-1])))
        state_values_target = state_values_old + advantages_tensor.unsqueeze(1)
        loss_critic = F.huber_loss(state_values, state_values_target)
        with lock:
            loss_actor.backward()
            loss_critic.backward()

        return loss_policy.item(), loss_entropy.item(), loss_actor.item(), loss_critic.item()
    

def reset_shared_advantages(shared_advantages: np.array) -> None:
    for i, j in itertools.product(
        range(shared_advantages.shape[0]), 
        range(shared_advantages.shape[1])
        ):
        shared_advantages[i, j] = 0
    

def share_adam_optimizer(optimizer):
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state[param]
            state["step"] = torch.tensor(0.0).share_memory_()
            state["exp_avg"] = torch.zeros_like(param).share_memory_()
            state["exp_avg_sq"] = torch.zeros_like(param).share_memory_()


def advantage_mean_std(
        shared_advantages, 
        advantage_mean, 
        advantage_std, 
        n_advantages_total, 
    ) -> None:
    advantage_mean.value = 0
    advantage_std.value = 0
    for i, j in itertools.product(
            range(shared_advantages.shape[0]), 
            range(shared_advantages.shape[1]),
        ):
        advantage = shared_advantages[i, j]
        advantage_mean.value += advantage

    advantage_mean.value /= n_advantages_total.value
    for i, j in itertools.product(
            range(shared_advantages.shape[0]), 
            range(shared_advantages.shape[1]),
        ):
        advantage = shared_advantages[i, j]
        advantage_std.value += (advantage.item() - advantage_mean.value)**2

    advantage_std.value = np.sqrt(advantage_std.value/n_advantages_total.value)
    reset_shared_advantages(shared_advantages)
    n_advantages_total.value = 0


def train_ppo_worker(worker_id, 
                     shared_advantages, 
                     advantage_mean, 
                     advantage_std,
                     n_advantages_total,
                     policy_network,
                     value_network, 
                     policy_optimizer, 
                     value_optimizer, 
                     env,
                     agent,
                     t_max,
                     n_episodes,
                     alpha_entropy,
                     epochs,
                     thresh,
                     print_iter,
                     log_dir,
                     barrier, 
                     lock, 
                     ) -> None:
    
    agent.policy_network = policy_network
    agent.value_network = value_network
    writer = SummaryWriter(log_dir=log_dir)
    rewards_episodes = deque(maxlen=10)
    reward_mean = None
    it = 0
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        reward_episode = 0
        while not done:
            try:
                # Synchronize all workers (same weights) before unrolling the policy
                barrier.wait()
                states, actions, actions_probs_old, rewards, R, done = agent.unroll(state, t_max)
                state = states[-1]
                reward_episode += R

            except BrokenBarrierError:
                print(f"\nWorker {worker_id}: Barrier was broken, likely due to another worker finishing earlier.")
                return # Stop training because one worker finished early
            
            advantages, state_values_old, n_advantages = agent.calculate_advantages(
                shared_advantages, 
                worker_id, 
                states, 
                rewards, 
                done
            )
            # Update the total number of advantages gathered by the workers
            with lock:
                n_advantages_total.value += n_advantages

            # All workers must finish calculating their advantages before worker 0 calculates their mean and std
            barrier.wait()
            if worker_id == 0:
                advantage_mean_std(
                    shared_advantages, 
                    advantage_mean,
                    advantage_std, 
                    n_advantages_total
                )

            # Wait for worker 0 to finish calculating the advantage mean and std.
            barrier.wait()
            for i in range(epochs):
                loss_policy, loss_entropy, loss_actor, loss_critic = agent.backward(
                    states, 
                    actions,
                    actions_probs_old, 
                    advantages, 
                    advantage_mean, 
                    advantage_std,
                    state_values_old,
                    alpha_entropy,
                    lock,
                )
                writer.add_scalar(f'worker_{worker_id}/loss_policy', loss_policy, it)
                writer.add_scalar(f'worker_{worker_id}/loss_entropy', loss_entropy, it)
                writer.add_scalar(f'worker_{worker_id}/loss_actor', loss_actor, it)
                writer.add_scalar(f'worker_{worker_id}/loss_critic', loss_critic, it)
                if worker_id == 0:
                    writer.add_scalar(f'worker_{worker_id}/advantage_mean', advantage_mean.value, it)
                    writer.add_scalar(f'worker_{worker_id}/advantage_std', advantage_std.value, it)

                it += 1
                # Wait for all workers to do a backward pass before performing a gradient update.
                barrier.wait()
                if worker_id == 0:
                    torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=0.5)
                    torch.nn.utils.clip_grad_norm_(value_network.parameters(), max_norm=0.5)
                    policy_optimizer.step()
                    value_optimizer.step()
                    policy_optimizer.zero_grad()
                    value_optimizer.zero_grad()

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
