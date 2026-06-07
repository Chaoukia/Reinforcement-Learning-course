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


class A2CWorker(Agent[np.array, int]):
    """Base class for an Advantage Actor-Critic worker with normalized advantages.

    Implements the core A2C logic including trajectory collection, advantage
    calculation with shared normalization across workers, and gradient updates
    for both actor and critic networks.

    Attributes:
        n_workers: Total number of parallel workers sharing the same networks.
        actor_network: Neural network representing the actor (policy).
        critic_network: Neural network representing the critic (value function).
    """

    def __init__(self,
                 env: Env[np.array, int],
                 worker_id: int,
                 n_workers: int,
                 gamma: float = 0.99
                 ) -> None:
        """Initialize the Actor-Critic agent.

        Args:
            env: Gymnasium environment with numpy array observations and integer actions.
            worker_id: Integer identifier of this worker.
            n_workers: Total number of parallel workers sharing the same networks.
            gamma: Discount factor. Defaults to 0.99.

        Attributes:
            actor_network: Neural network representing the actor (policy).
            critic_network: Neural network representing the critic (value function).
        """

        super().__init__(env, gamma)
        self.worker_id = worker_id
        self.n_workers = n_workers
        self.actor_network, self.critic_network = self.make_networks()

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Create and return the policy and value networks.

        Returns:
            A tuple containing:
                - actor_network: Neural network for the actor (policy).
                - critic_network: Neural network for the critic (value function).

        Raises:
            NotImplementedError: Must be implemented by subclasses.
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

        logits = self.actor_network(torch.from_numpy(state).float())
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
        ) -> tuple[torch.tensor, torch.tensor, torch.tensor, np.array, np.array, float, float, float]:
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
                - actions_logprobs: List of log probabilities of sampled actions (length t_max).
                - entropies: List of policy entropies evaluated at each state (length t_max).
                - rewards: List of rewards received (length t_max).
                - dones: List of episode termination flags (length t_max).
                - episode: Updated episode counter.
                - R: Updated cumulative reward for the current episode.
                - reward_mean: Updated running mean of episode returns.
        """

        states = np.empty((t_max+1, *state.shape), dtype=np.float32)
        states[0] = state
        actions_logprobs = torch.empty(t_max, dtype=torch.float32)
        entropies = torch.empty(t_max, dtype=torch.float32)
        rewards = np.empty(t_max, dtype=np.float32)
        dones = np.empty(t_max, dtype=np.float32)
        for t in range(t_max):
            action, action_logprob, entropy = self.action_explore(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = (terminated or truncated)
            R += reward
            actions_logprobs[t] = action_logprob
            entropies[t] = entropy
            rewards[t] = reward
            dones[t] = done
            state = next_state if not done else env.reset()[0]
            states[t+1] = state
            if done:
                writer.add_scalar(f'worker_{self.worker_id}/return', R, episode)
                if len(rewards_episodes) >= rewards_episodes.maxlen:
                    reward_mean = np.mean(rewards_episodes) if reward_mean is None else reward_mean + (R - rewards_episodes[0])/rewards_episodes.maxlen

                rewards_episodes.append(R)
                R = 0
                episode += 1

        return states, actions_logprobs, entropies, rewards, dones, episode, R, reward_mean

    def calculate_advantages(
            self,
            shared_advantages: np.array,
            state_values: torch.tensor,
            rewards: torch.tensor,
            dones: torch.tensor
        ) -> torch.tensor:
        """Calculate the advantages and store them in the shared advantages array.

        Computes advantages for each transition,
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
            Tensor of shape t_max containing the computed advantages.
        """

        # Initialize the advantages list
        t_max = shared_advantages.shape[1]
        advantages_tensor = torch.empty(t_max, dtype=torch.float32)
        advantage = 0
        V_next_state = state_values[-1]
        for t in range(t_max-1, -1, -1):
            V_state = state_values[t]
            advantage = rewards[t] + self.gamma*V_next_state.detach()*(1 - dones[t]) - V_state
            shared_advantages[self.worker_id, t] = advantage.item()
            advantages_tensor[t] = advantage
            V_next_state = V_state

        return advantages_tensor

    def backward(self,
                 actions_logprobs_batch: torch.tensor,
                 entropy_batch: torch.tensor,
                 advantages_batch: torch.tensor,
                 advantage_mean,
                 advantage_std,
                 alpha_entropy: float,
                 lock
                 ) -> tuple[float, float, float, float]:
        """Update both policy and value networks using a collected trajectory.

        Normalizes advantages using the shared mean and standard deviation computed
        across all workers, then computes and backpropagates policy and critic losses.

        Args:
            states: List of states visited in the trajectory.
            actions_logprobs: Log probabilities of sampled actions.
            entropies: Policy entropies at each step.
            advantages: Per-timestep advantage tensors computed by calculate_advantages.
            advantage_mean: Shared value holding the global advantage mean.
            advantage_std: Shared value holding the global advantage standard deviation.
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

        advantage_normalized_batch = ((advantages_batch.detach() - advantage_mean.value)/(advantage_std.value + 1e-8))
        loss_policy = -(actions_logprobs_batch*advantage_normalized_batch).mean()
        loss_entropy = -entropy_batch.mean()
        loss_actor = loss_policy + alpha_entropy*loss_entropy
        loss_critic = (advantages_batch**2).mean()
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


def train_a2c_worker(
        shared_advantages: np.array,
        advantage_mean,
        advantage_std,
        actor_network: nn.Module,
        critic_network: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        value_optimizer: torch.optim.Optimizer,
        env: Env[np.array, int],
        agent: A2CWorker,
        t_max: int,
        n_train: int,
        batch_size: int,
        alpha_entropy: float,
        thresh: int,
        print_iter: int,
        log_dir: str | Path,
        barrier,
        lock,
    ) -> None:
    """Run the training loop for a single A2C worker process.

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
        agent: A2CWorker instance managing policy inference and gradient
            computation.
        t_max: Number of environment steps to collect per iteration.
        n_train: Total number of training iterations to run.
        batch_size: Mini-batch size for the A2C update.
        alpha_entropy: Coefficient for entropy regularization in the actor
            loss.
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
            states, entropies, actions_logprobs, rewards, dones, episode, R, reward_mean = agent.unroll(
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
        state_values = agent.critic_network(torch.from_numpy(states).float())
        advantages = agent.calculate_advantages(
            shared_advantages,
            state_values,
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
        loss_policy, loss_entropy, loss_actor, loss_critic = agent.backward(
            actions_logprobs,
            entropies,
            advantages,
            advantage_mean,
            advantage_std,
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
