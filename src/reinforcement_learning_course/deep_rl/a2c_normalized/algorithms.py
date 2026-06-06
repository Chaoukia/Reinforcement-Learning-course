import numpy as np
import itertools
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
    """Base class for an Advantage Actor-Critic worker with normalized advantages.

    Implements the core A2C logic including trajectory collection, advantage
    calculation with shared normalization across workers, and gradient updates
    for both actor and critic networks.

    Attributes:
        n_workers: Total number of parallel workers sharing the same networks.
        policy_network: Neural network representing the actor (policy).
        value_network: Neural network representing the critic (value function).
    """

    def __init__(self,
                 env: Env[np.array, int],
                 n_workers: int,
                 gamma: float = 0.99
                 ) -> None:
        """Initialize the Actor-Critic agent.

        Args:
            env: Gymnasium environment with numpy array observations and integer actions.
            n_workers: Total number of parallel workers sharing the same networks.
            gamma: Discount factor. Defaults to 0.99.

        Attributes:
            policy_network: Neural network representing the actor (policy).
            value_network: Neural network representing the critic (value function).
        """

        super().__init__(env, gamma)
        self.n_workers = n_workers
        self.policy_network, self.value_network = self.make_networks()

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Create and return the policy and value networks.

        Returns:
            A tuple containing:
                - policy_network: Neural network for the actor (policy).
                - value_network: Neural network for the critic (value function).

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

    def calculate_advantages(self,
                             shared_advantages: np.array,
                             index: int,
                             states: list[np.array],
                             rewards: list[float],
                             done: bool,
                             ) -> list[torch.tensor]:
        """Calculate advantages and store them in the shared advantages array.

        Computes n-step bootstrapped advantage estimates for each timestep in the
        trajectory and writes them into the shared array so that all workers can
        collectively compute a global mean and standard deviation for normalization.

        Args:
            shared_advantages: Shared array of shape (n_workers, t_max) used to
                accumulate advantages across all workers.
            index: Row index in shared_advantages corresponding to this worker.
            states: List of states visited in the trajectory (length t+1).
            rewards: List of rewards received at each step (length t).
            done: Whether the episode terminated at the end of the trajectory.

        Returns:
            A tuple containing:
                - advantages_list: List of per-timestep advantage tensors.
                - n_steps: Number of bootstrapping steps (len(states) - 1).
        """

        # Initialize the advantages list
        advantages_list = [0 for i in range(len(states)-1)]
        if done:
            R = 0

        else:
            with torch.no_grad():
                R = self.value_network(torch.from_numpy(states[-1]))

        for t in range(len(states)-2, -1, -1):
            R = rewards[t] + self.gamma*R
            advantage = R - self.value_network(torch.from_numpy(states[t]))
            shared_advantages[index, t] = advantage.item()
            advantages_list[t] = advantage

        return advantages_list, len(states)-1

    def backward(self,
                 states: list[np.array],
                 actions_logprobs: list[torch.tensor],
                 entropies: list[torch.tensor],
                 advantages: list[torch.tensor],
                 advantage_mean,
                 advantage_std,
                 alpha_entropy: float,
                 optimizer_policy: optim,
                 optimizer_value: optim
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

        optimizer_policy.zero_grad()
        optimizer_value.zero_grad()
        loss_policy, loss_entropy, loss_critic = 0, 0, 0
        for t in range(len(states)-2, -1, -1):
            advantage_normalized = (advantages[t].detach() - advantage_mean.value)/(advantage_std.value + 1e-8)
            loss_policy += -self.gamma**t*advantage_normalized*actions_logprobs[t]
            loss_entropy += -entropies[t]
            if abs(advantages[t]) <= 1:
                loss_critic += (advantages[t]**2)/2

            else:
                loss_critic += abs(advantages[t])

        loss_policy /= (len(states) - 1)*self.n_workers
        loss_entropy /= (len(states) - 1)*self.n_workers
        loss_actor = loss_policy + alpha_entropy*loss_entropy
        loss_critic /= (len(states) - 1)*self.n_workers
        loss_actor.backward()
        loss_critic.backward()
        # Clip the gradient of the actor
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1)

        return loss_policy.item(), loss_entropy.item(), loss_actor.item(), loss_critic.item()


def reset_shared_advantages(shared_advantages: np.array) -> None:
    """Reset all entries of the shared advantages array to zero.

    Args:
        shared_advantages: Shared array of shape (n_workers, t_max) to be zeroed out.
    """

    for i, j in itertools.product(
        range(shared_advantages.shape[0]),
        range(shared_advantages.shape[1])
        ):
        shared_advantages[i, j] = 0


def share_adam_optimizer(optimizer) -> None:
    """Move Adam optimizer state tensors to shared memory.

    Enables the optimizer state to be shared across processes by placing the
    step counter, first moment, and second moment tensors in shared memory.

    Args:
        optimizer: An Adam optimizer instance whose state will be moved to
            shared memory in-place.
    """

    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state[param]
            state["step"] = torch.tensor(0.0).share_memory_()
            state["exp_avg"] = torch.zeros_like(param).share_memory_()
            state["exp_avg_sq"] = torch.zeros_like(param).share_memory_()


def train_a2c_worker(worker_id,
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
                     thresh,
                     print_iter,
                     log_dir,
                     barrier,
                     lock,
                     ) -> None:
    """Run the training loop for a single A2C worker process.

    Each worker collects trajectories, updates the shared advantages array, waits
    at a barrier for all workers to synchronize, then worker 0 computes the global
    advantage mean and standard deviation before all workers perform gradient updates.
    Training stops early if the rolling mean return exceeds thresh, or after
    n_episodes episodes.

    Args:
        worker_id: Integer identifier for this worker process.
        shared_advantages: Shared array of shape (n_workers, t_max) for accumulating
            per-worker advantage estimates across a synchronization step.
        advantage_mean: Shared value object holding the global advantage mean,
            computed by worker 0 after each barrier synchronization.
        advantage_std: Shared value object holding the global advantage standard
            deviation, computed by worker 0 after each barrier synchronization.
        n_advantages_total: Shared value object accumulating the total number of
            advantage estimates across all workers within a synchronization step.
        policy_network: Shared policy (actor) network used by the agent.
        value_network: Shared value (critic) network used by the agent.
        policy_optimizer: Shared optimizer for the policy network.
        value_optimizer: Shared optimizer for the value network.
        env: Gymnasium environment instance for this worker.
        agent: A2CWorker instance that performs unroll, calculate_advantages, and backward.
        t_max: Maximum number of steps per unroll before bootstrapping.
        n_episodes: Maximum number of episodes to train for.
        alpha_entropy: Coefficient for entropy regularization in the actor loss.
        thresh: Rolling mean return threshold for early stopping.
        print_iter: Frequency (in episodes) at which worker 0 prints training progress.
        log_dir: Directory path for TensorBoard SummaryWriter logs.
        barrier: Multiprocessing barrier used to synchronize workers before and after
            advantage normalization.
        lock: Multiprocessing lock used to safely update shared counters and optimizers.
    """

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
            states, actions_logprobs, rewards, entropies, R, done = agent.unroll(state, t_max)
            reward_episode += R
            advantages, n_advantages = agent.calculate_advantages(
                shared_advantages,
                worker_id,
                states,
                rewards,
                done
            )
            # Update the total number of advantages gathered by the workers.
            with lock:
                n_advantages_total.value += n_advantages

            # The worker must wait for the other workers to update the shared advantages.
            try:
                barrier.wait()
                if worker_id == 0:
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

            except BrokenBarrierError:
                print(f"\nWorker {worker_id}: Barrier was broken, likely due to another worker finishing earlier.")
                return # Stop training because one worker finished early

            # Wait for worker 0 to finish calculating the advantage mean and std.
            barrier.wait()
            loss_policy, loss_entropy, loss_actor, loss_critic = agent.backward(
                states,
                actions_logprobs,
                entropies,
                advantages,
                advantage_mean,
                advantage_std,
                alpha_entropy,
                policy_optimizer,
                value_optimizer,
            )
            state = states[-1]
            writer.add_scalar(f'worker_{worker_id}/loss_policy', loss_policy, it)
            writer.add_scalar(f'worker_{worker_id}/loss_entropy', loss_entropy, it)
            writer.add_scalar(f'worker_{worker_id}/loss_actor', loss_actor, it)
            writer.add_scalar(f'worker_{worker_id}/loss_critic', loss_critic, it)
            if worker_id == 0:
                writer.add_scalar(f'worker_{worker_id}/advantage_mean', advantage_mean.value, it)
                writer.add_scalar(f'worker_{worker_id}/advantage_std', advantage_std.value, it)

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
