import itertools
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from threading import BrokenBarrierError
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from pathlib import Path
from reinforcement_learning_course.core import Agent
from gymnasium import Env


class PPOWorker(Agent[np.array, int]):

    def __init__(self, 
                 env: Env[np.array, int], 
                 worker_id: int,
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
            actor_network: Neural network representing the actor (policy).
            critic_network: Neural network representing the critic (value function).
        """

        super().__init__(env, gamma)
        self.worker_id = worker_id
        self.n_workers = n_workers
        self.epsilon = epsilon
        self.lambd = lambd
        self.actor_network, self.critic_network = self.make_networks()

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
            logits = self.actor_network(torch.from_numpy(state))
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
            logits = self.actor_network(torch.from_numpy(state))
            dist = Categorical(logits=logits)
            action = dist.sample()
            return action.item()
        
    def unroll(
            self, 
            env: Env[np.array, int], 
            state: np.array, 
            t_max: int, 
            episode: int,
            R: float, 
            rewards_episodes: deque, 
            reward_mean: float, 
            writer: SummaryWriter
        ) -> tuple[list[np.array], list[int], list[float], list[float], list[bool], float, float]:
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

        states, actions, actions_probs, rewards, dones = [state], [], [], [], []
        t = 0
        while t < t_max:
            action, action_prob = self.action_explore(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = (terminated or truncated)
            R += reward
            actions.append(action)
            actions_probs.append(action_prob)
            rewards.append(reward)
            dones.append(done)
            state = next_state if not done else env.reset()[0]
            states.append(state)
            t += 1
            if done:
                writer.add_scalar(f'worker_{self.worker_id}/return', R, episode)
                if len(rewards_episodes) >= rewards_episodes.maxlen:
                    reward_mean = np.mean(rewards_episodes) if reward_mean is None else reward_mean + (R - rewards_episodes[0])/rewards_episodes.maxlen

                rewards_episodes.append(R)
                R = 0
                episode += 1

        return states, actions, actions_probs, rewards, dones, episode, R, reward_mean
    
    def calculate_advantages(
            self, 
            shared_advantages: np.array,
            state_values: torch.tensor,
            rewards: list[float],
            dones: list[bool]    
        ) -> list[torch.tensor]:
        """Calculate the advantages and store them in the shared array of advantages, 
        the purpose being to calculate the mean and std of advantages.
        Return the list of advantages in tensor form, this will be used later
        to update the critic.
        Return len(states)-2, the n-step length of bootstrapping, it will be used for
        calculating the advantage mean and std.
        """
        
        # Initialize the advantages list
        n_transitions = shared_advantages.shape[1]  # equal to t_max
        advantages_list = [0 for i in range(n_transitions)]
        advantage = 0
        # state_values.shape[0] = n _transitions+1, thus V_next_state is the value of the final state after all transitions.
        V_next_state = state_values[-1]
        for t in range(n_transitions-1, -1, -1):
            V_state = state_values[t]
            td = rewards[t] + self.gamma*V_next_state*(1 - dones[t]) - V_state
            advantage = td + (self.gamma*self.lambd)*advantage*(1 - dones[t])
            shared_advantages[self.worker_id, t] = advantage.item()
            advantages_list[t] = advantage
            V_next_state = V_state

        return advantages_list
    
    def backward(
            self, 
            states_batch: torch.tensor, 
            actions_batch: list[int],
            actions_probs_old_batch: torch.tensor, 
            advantages_batch: torch.tensor,
            advantage_mean, 
            advantage_std, 
            state_values_old_batch: torch.tensor,
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

        batch_size = states_batch.shape[0]
        # Take the softmax along dim=1 to transform logits into probabilities.
        policy_probs = torch.softmax(self.actor_network(states_batch), dim=1)
        # Action probabilities according to the current policy
        actions_probs_batch = policy_probs[range(batch_size), actions_batch]
        ratio = actions_probs_batch/actions_probs_old_batch
        advantage_normalized_batch = (advantages_batch - advantage_mean.value)/(advantage_std.value + 1e-8)
        loss_policy = -torch.minimum(
            ratio*advantage_normalized_batch, torch.clip(ratio, 1-self.epsilon, 1+self.epsilon)*advantage_normalized_batch
        ).mean()/self.n_workers
        loss_entropy = 0
        for i in range(batch_size):
            dist = Categorical(logits=policy_probs[i, :])
            loss_entropy -= dist.entropy()

        loss_entropy /= batch_size*self.n_workers
        loss_actor = loss_policy + alpha_entropy*loss_entropy
        state_values = self.critic_network(states_batch)
        state_values_target = state_values_old_batch + advantages_batch
        loss_critic = F.smooth_l1_loss(state_values, state_values_target)/self.n_workers
        with lock:
            loss_actor.backward()
            loss_critic.backward()

        return loss_policy.item(), loss_entropy.item(), loss_actor.item(), loss_critic.item()

    def save(self, path: str | Path) -> None:
        """Save the policy and value network weights to disk.
        
        Saves both networks as PyTorch state dictionaries.
        
        Args:
            path: Directory path where the network weights will be saved.
        """

        os.makedirs(str(path), exist_ok=True)
        path = Path(path)
        path_actor = path / "actor.pt"
        path_critic = path / "critic.pt"
        torch.save(self.actor_network.state_dict(), path_actor)
        torch.save(self.actor_network.state_dict(), path_critic)

    def load(self, path: str | Path) -> None:
        """Load network weights from disk.
        
        Restores both policy and value networks from saved state dictionaries.
        
        Args:
            path: Directory containing the saved network weights.
        
        Raises:
            ValueError: If the path does not exist or is not a directory.
        """

        if not os.path.isdir(str(path)):
            raise ValueError(f"{str(path)} is not a directory")
        
        path = Path(path)
        path_actor = path / "actor.pt"
        path_critic = path / "critic.pt"
        self.actor_network.load_state_dict(torch.load(path_actor))
        self.critic_network.load_state_dict(torch.load(path_critic))
    

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
    ) -> None:
    advantage_mean.value = 0
    advantage_std.value = 0
    n_advantages = shared_advantages.shape[0]*shared_advantages.shape[1]
    for i, j in itertools.product(
            range(shared_advantages.shape[0]), 
            range(shared_advantages.shape[1]),
        ):
        advantage = shared_advantages[i, j]
        advantage_mean.value += advantage

    advantage_mean.value /= n_advantages
    for i, j in itertools.product(
            range(shared_advantages.shape[0]), 
            range(shared_advantages.shape[1]),
        ):
        advantage = shared_advantages[i, j]
        advantage_std.value += (advantage.item() - advantage_mean.value)**2

    advantage_std.value = np.sqrt(advantage_std.value/n_advantages)
    reset_shared_advantages(shared_advantages)


def train_ppo_worker(
        shared_advantages, 
        advantage_mean, 
        advantage_std,
        actor_network: nn.Module,
        critic_network: nn.Module, 
        policy_optimizer: torch.optim.Optimizer, 
        value_optimizer: torch.optim.Optimizer, 
        env: Env[np.array, int],
        agent: PPOWorker,
        t_max: int,
        n_train: int,
        batch_size: int,
        alpha_entropy: float,
        epochs: int,
        thresh: int,
        print_iter: int,
        log_dir: str | Path,
        barrier, 
        lock,              
    ) -> None:
    
    torch.set_num_threads(1)
    agent.actor_network = actor_network
    agent.critic_network = critic_network
    writer = SummaryWriter(log_dir=log_dir)
    episode = 0
    R = 0
    reward_mean = None
    rewards_episodes = deque(maxlen=100)
    state, _ = env.reset()
    for it in range(n_train):
        try:
            # Synchronize all workers (same weights) before unrolling the policy
            barrier.wait()
            states, actions, actions_probs_old, rewards, dones, episode, R, reward_mean = agent.unroll(
                env, 
                state, 
                t_max, 
                episode, 
                R, 
                rewards_episodes, 
                reward_mean, 
                writer
            )
            if reward_mean is not None and reward_mean >= thresh:
                print(f"\nWorker {agent.worker_id} achieved early stopping achieved after {episode} episodes")
                barrier.abort()     # Break the barrier because the worker finished
                return
            
            if episode%print_iter == 0 and reward_mean is not None and agent.worker_id == 0:
                print('Worker : %d , episode : %d , return : %.3F' %(agent.worker_id, episode, reward_mean))

        except BrokenBarrierError:
            print(f"\nWorker {agent.worker_id}: Barrier was broken, likely due to another worker finishing earlier.")
            return # Stop training because one worker finished early
        
        state = states[-1]
        states = torch.from_numpy(np.array(states))
        actions_probs_old = torch.from_numpy(np.array(actions_probs_old))
        with torch.no_grad():
            state_values_old = agent.critic_network(states)

        advantages = agent.calculate_advantages(
            shared_advantages, 
            state_values_old, 
            rewards, 
            dones
        )
        advantages = torch.tensor(advantages).unsqueeze(1)
        # All workers must finish calculating their advantages before worker 0 calculates their mean and std
        barrier.wait()
        if agent.worker_id == 0:
            advantage_mean_std(
                shared_advantages, 
                advantage_mean,
                advantage_std
            )
            writer.add_scalar(f'worker_{agent.worker_id}/advantage_mean', advantage_mean.value, it)
            writer.add_scalar(f'worker_{agent.worker_id}/advantage_std', advantage_std.value, it)

        # Wait for worker 0 to finish calculating the advantage mean and std.
        barrier.wait()
        for epoch in range(epochs):
            indices = np.random.permutation(t_max)
            for batch in range(t_max//batch_size):
                indices_batch = indices[batch*batch_size : (batch+1)*batch_size]
                states_batch = states[indices_batch]
                actions_batch = [actions[index] for index in indices_batch]
                actions_probs_old_batch = actions_probs_old[indices_batch]
                advantages_batch = advantages[indices_batch]
                state_values_old_batch = state_values_old[indices_batch]
                loss_policy, loss_entropy, loss_actor, loss_critic = agent.backward(
                    states_batch, 
                    actions_batch,
                    actions_probs_old_batch, 
                    advantages_batch, 
                    advantage_mean, 
                    advantage_std,
                    state_values_old_batch,
                    alpha_entropy,
                    lock,
                )
                writer.add_scalar(f'worker_{agent.worker_id}/loss_policy', loss_policy, it)
                writer.add_scalar(f'worker_{agent.worker_id}/loss_entropy', loss_entropy, it)
                writer.add_scalar(f'worker_{agent.worker_id}/loss_actor', loss_actor, it)
                writer.add_scalar(f'worker_{agent.worker_id}/loss_critic', loss_critic, it)


            # Wait for all workers to do a backward pass before performing a gradient update.
            barrier.wait()
            if agent.worker_id == 0:
                torch.nn.utils.clip_grad_norm_(actor_network.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(critic_network.parameters(), max_norm=0.5)
                policy_optimizer.step()
                value_optimizer.step()
                policy_optimizer.zero_grad()
                value_optimizer.zero_grad()

    print(f'\nWorker {agent.worker_id} finished training without early stopping.')
    barrier.abort()     # Break the barrier because the worker finished
