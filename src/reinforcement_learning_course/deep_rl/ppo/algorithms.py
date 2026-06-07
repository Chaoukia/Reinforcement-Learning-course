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
    """A worker agent implementing the PPO (Proximal Policy Optimization) algorithm.

    Each worker maintains its own copy of the environment and collects
    trajectories that are used to update shared actor and critic networks
    in a multi-worker parallel training setup.
    """

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
            worker_id: Integer identifier for this worker.
            n_workers: Total number of parallel workers.
            epsilon: Clipping parameter for the PPO surrogate objective.
            lambd: GAE lambda parameter for advantage estimation.
            gamma: Discount factor. Defaults to 0.99.

        Attributes:
            worker_id: Integer identifier for this worker.
            n_workers: Total number of parallel workers.
            epsilon: Clipping parameter for the PPO surrogate objective.
            lambd: GAE lambda parameter for advantage estimation.
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
        """Create and return the actor and critic neural networks.

        Returns:
            A tuple containing:
                - actor_network: Neural network for the policy.
                - critic_network: Neural network for the value function.

        Raises:
            NotImplementedError: Subclasses must implement this method.
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
                - action_logprob: Log probability of the sampled action.
        """

        with torch.no_grad():
            logits = self.actor_network(torch.from_numpy(state).float())
            dist = Categorical(logits=logits)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            return action.item(), action_logprob.item()

    def action(self, state: np.array) -> int:
        """Select an action deterministically from the policy.

        Used during testing/evaluation to get the best action without sampling.

        Args:
            state: NumPy array representing the current state.

        Returns:
            Integer action sampled from the policy distribution.
        """

        with torch.no_grad():
            logits = self.actor_network(torch.from_numpy(state).float())
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
            env: Gymnasium environment used to reset after episode termination.
            state: Starting state for the rollout.
            t_max: Maximum number of steps before truncating the episode.
            episode: Current episode counter.
            R: Cumulative reward accumulated so far in the current episode.
            rewards_episodes: Deque storing recent episode returns for mean computation.
            reward_mean: Running mean of episode returns, or None if not yet computed.
            writer: TensorBoard SummaryWriter for logging episode returns.

        Returns:
            A tuple containing:
                - states: List of visited states (length t_max + 1).
                - actions: List of actions taken (length t_max).
                - actions_logprobs: List of log probabilities of sampled actions (length t_max).
                - rewards: List of rewards received (length t_max).
                - dones: List of episode termination flags (length t_max).
                - episode: Updated episode counter.
                - R: Updated cumulative reward for the current episode.
                - reward_mean: Updated running mean of episode returns.
        """

        states, actions, actions_logprobs, rewards, dones = [state], [], [], [], []
        t = 0
        while t < t_max:
            action, action_logprob = self.action_explore(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = (terminated or truncated)
            R += reward
            actions.append(action)
            actions_logprobs.append(action_logprob)
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

        return states, actions, actions_logprobs, rewards, dones, episode, R, reward_mean

    def calculate_advantages(
            self,
            shared_advantages: np.array,
            state_values: torch.tensor,
            rewards: list[float],
            dones: list[bool]
        ) -> torch.tensor:
        """Calculate GAE advantages and store them in the shared advantages array.

        Computes Generalized Advantage Estimation (GAE) for each transition,
        writes each advantage into the shared array (indexed by worker ID) so
        that the global mean and standard deviation can be computed across all
        workers, and returns the advantages as a tensor for use in the critic
        update.

        Args:
            shared_advantages: Shared numpy array of shape (n_workers, t_max)
                used to accumulate advantages across all workers.
            state_values: Tensor of state values of shape (t_max + 1, 1)
                produced by the critic network.
            rewards: List of rewards received at each transition.
            dones: List of episode termination flags for each transition.

        Returns:
            Tensor of shape (t_max, 1) containing the computed GAE advantages.
        """

        # Initialize the advantages list
        n_transitions = shared_advantages.shape[1]  # equal to t_max
        advantages_tensor = torch.zeros((n_transitions, 1), dtype=torch.float32)
        advantage = 0
        V_next_state = state_values[-1]
        for t in range(n_transitions-1, -1, -1):
            V_state = state_values[t]
            td = rewards[t] + self.gamma*V_next_state*(1 - dones[t]) - V_state
            advantage = td + (self.gamma*self.lambd)*advantage*(1 - dones[t])
            shared_advantages[self.worker_id, t] = advantage.item()
            advantages_tensor[t, 0] = advantage
            V_next_state = V_state

        return advantages_tensor

    def backward(
            self,
            states_batch: torch.tensor,
            actions_batch: list[int],
            actions_logprobs_old_batch: torch.tensor,
            advantages_batch: torch.tensor,
            advantage_mean,
            advantage_std,
            state_values_old_batch: torch.tensor,
            alpha_entropy: float,
            lock
        ) -> tuple[float, float, float, float]:
        """Compute losses and accumulate gradients for actor and critic networks.

        Computes the clipped PPO surrogate objective for the actor and the
        smooth L1 loss for the critic, then accumulates gradients into the
        shared network parameters under a lock.

        Args:
            states_batch: Tensor of states for the current mini-batch.
            actions_batch: List of integer actions taken in the mini-batch.
            actions_logprobs_old_batch: Tensor of log probabilities of actions
                under the old policy.
            advantages_batch: Tensor of GAE advantage estimates for the mini-batch.
            advantage_mean: Shared value holding the mean of all advantages,
                used for normalization.
            advantage_std: Shared value holding the standard deviation of all
                advantages, used for normalization.
            state_values_old_batch: Tensor of critic values computed before the
                update, used as targets for the critic loss.
            alpha_entropy: Coefficient for the entropy regularization term.
            lock: Threading lock used to serialize gradient accumulation.

        Returns:
            A tuple containing:
                - loss_policy: Scalar loss for the clipped policy gradient term.
                - loss_entropy: Scalar loss for the entropy regularization term.
                - loss_actor: Scalar total actor loss (policy + entropy).
                - loss_critic: Scalar loss for the value function.
        """

        logits = self.actor_network(states_batch)
        dist = Categorical(logits=logits)
        actions_logprobs_batch = dist.log_prob(torch.tensor(actions_batch))
        ratio = torch.exp(actions_logprobs_batch - actions_logprobs_old_batch)
        advantage_normalized_batch = ((advantages_batch - advantage_mean.value)/(advantage_std.value + 1e-8)).squeeze()
        loss_policy = -torch.minimum(
            ratio*advantage_normalized_batch, torch.clip(ratio, 1-self.epsilon, 1+self.epsilon)*advantage_normalized_batch
        ).mean()/self.n_workers
        loss_entropy = -dist.entropy().mean()/self.n_workers
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
        torch.save(self.critic_network.state_dict(), path_critic)

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
    """Reset all entries in the shared advantages array to zero.

    Iterates over every (worker, timestep) index pair and sets the
    corresponding shared advantage value to zero, preparing the array
    for the next training iteration.

    Args:
        shared_advantages: Shared numpy array of shape (n_workers, t_max)
            containing advantages accumulated across all workers.
    """

    for i, j in itertools.product(
        range(shared_advantages.shape[0]),
        range(shared_advantages.shape[1])
        ):
        shared_advantages[i, j] = 0


def share_adam_optimizer(optimizer: torch.optim.Optimizer) -> None:
    """Move Adam optimizer state tensors to shared memory.

    Enables multiple processes to share a single Adam optimizer by
    placing its internal state tensors (step counter, first and second
    moment estimates) into shared memory.

    Args:
        optimizer: Adam optimizer whose state tensors will be moved to
            shared memory.
    """

    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state[param]
            state["step"] = torch.tensor(0.0).share_memory_()
            state["exp_avg"] = torch.zeros_like(param).share_memory_()
            state["exp_avg_sq"] = torch.zeros_like(param).share_memory_()


def advantage_mean_std(
        shared_advantages: np.array,
        advantage_mean,
        advantage_std,
    ) -> None:
    """Compute the mean and standard deviation of all shared advantages.

    Iterates over the shared advantages array to compute the global mean
    and standard deviation across all workers and timesteps, writes the
    results into the provided shared value objects, and then resets the
    shared advantages array to zero.

    Args:
        shared_advantages: Shared numpy array of shape (n_workers, t_max)
            containing advantages accumulated by all workers.
        advantage_mean: Shared value object whose .value attribute will be
            set to the computed mean of all advantages.
        advantage_std: Shared value object whose .value attribute will be
            set to the computed standard deviation of all advantages.
    """

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
        shared_advantages: np.array,
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
    """Run the training loop for a single PPO worker process.

    Executes n_train iterations of trajectory collection followed by
    mini-batch gradient updates. Workers synchronize via a barrier at
    key points: before unrolling, after computing advantages, and after
    each mini-batch backward pass. Only worker 0 performs the actual
    optimizer step and logs advantage statistics. Training stops early
    if the running mean return meets the threshold or if another worker
    triggers early stopping.

    Args:
        shared_advantages: Shared numpy array of shape (n_workers, t_max)
            used to accumulate advantages across workers.
        advantage_mean: Shared value object for the global advantage mean.
        advantage_std: Shared value object for the global advantage standard
            deviation.
        actor_network: Shared actor neural network updated by all workers.
        critic_network: Shared critic neural network updated by all workers.
        policy_optimizer: Optimizer for the actor network.
        value_optimizer: Optimizer for the critic network.
        env: Gymnasium environment for this worker to interact with.
        agent: PPOWorker instance managing policy inference and gradient
            computation.
        t_max: Number of environment steps to collect per iteration.
        n_train: Total number of training iterations to run.
        batch_size: Mini-batch size for the PPO update.
        alpha_entropy: Coefficient for entropy regularization in the actor
            loss.
        epochs: Number of epochs to iterate over collected data per
            training iteration.
        thresh: Early stopping threshold for the running mean return.
        print_iter: Frequency (in episodes) at which worker 0 prints
            progress.
        log_dir: Directory path for TensorBoard logging.
        barrier: Synchronization barrier shared across all worker processes.
        lock: Threading lock for serializing gradient accumulation.
    """

    torch.set_num_threads(1)
    agent.actor_network = actor_network
    agent.critic_network = critic_network
    writer = SummaryWriter(log_dir=log_dir)
    episode = 0
    R = 0
    reward_mean = None
    rewards_episodes = deque(maxlen=100)
    state, _ = env.reset()
    learn_it = 0
    for it in range(n_train):
        try:
            # Synchronize all workers (same weights) before unrolling the policy
            barrier.wait()
            states, actions, actions_logprobs_old, rewards, dones, episode, R, reward_mean = agent.unroll(
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
        states = torch.from_numpy(np.array(states)).float()
        actions_logprobs_old = torch.from_numpy(np.array(actions_logprobs_old)).float()
        with torch.no_grad():
            state_values_old = agent.critic_network(states)

        advantages = agent.calculate_advantages(
            shared_advantages,
            state_values_old,
            rewards,
            dones
        )
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
                actions_logprobs_old_batch = actions_logprobs_old[indices_batch]
                advantages_batch = advantages[indices_batch]
                state_values_old_batch = state_values_old[indices_batch]
                loss_policy, loss_entropy, loss_actor, loss_critic = agent.backward(
                    states_batch,
                    actions_batch,
                    actions_logprobs_old_batch,
                    advantages_batch,
                    advantage_mean,
                    advantage_std,
                    state_values_old_batch,
                    alpha_entropy,
                    lock,
                )
                writer.add_scalar(f'worker_{agent.worker_id}/loss_policy', loss_policy, learn_it)
                writer.add_scalar(f'worker_{agent.worker_id}/loss_entropy', loss_entropy, learn_it)
                writer.add_scalar(f'worker_{agent.worker_id}/loss_actor', loss_actor, learn_it)
                writer.add_scalar(f'worker_{agent.worker_id}/loss_critic', loss_critic, learn_it)
                learn_it += 1

                # Wait for all workers to do a backward pass before performing a gradient update.
                barrier.wait()
                if agent.worker_id == 0:
                    torch.nn.utils.clip_grad_norm_(actor_network.parameters(), max_norm=0.5)
                    torch.nn.utils.clip_grad_norm_(critic_network.parameters(), max_norm=0.5)
                    policy_optimizer.step()
                    value_optimizer.step()
                    policy_optimizer.zero_grad()
                    value_optimizer.zero_grad()

                # Wait for worker 0 to finish updating the parameters before moving to the next batch update.
                barrier.wait()

    print(f'\nWorker {agent.worker_id} finished training without early stopping.')
    barrier.abort()     # Break the barrier because the worker finished
