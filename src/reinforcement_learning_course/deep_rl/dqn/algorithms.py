import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from reinforcement_learning_course.deep_rl.dqn.utils import Memory, deque, update_epsilon
from reinforcement_learning_course.core import Agent
from gymnasium import Env


ExperienceSample = tuple[torch.tensor, np.array, torch.tensor, torch.tensor, torch.tensor]


class DQN(Agent[np.array, int]):
    """Deep Q-Network (DQN) agent for deep reinforcement learning.
    
    Implements DQN with experience replay and target network for off-policy
    learning. Optionally supports Double Q-learning for reduced overestimation.
    """

    def __init__(self, 
                 env: Env[np.array, int], 
                 max_size: int = 1e5, 
                 gamma: float = 0.99, 
                 double_learning: bool = False
                 ) -> None:
        """Initialize the DQN agent.
        
        Args:
            env: Gymnasium environment with numpy array observations and integer actions.
            max_size: Maximum size of the experience replay buffer. Defaults to 1e5.
            gamma: Discount factor. Defaults to 0.99.
            double_learning: If True, use Double Q-learning to reduce overestimation bias.
              Defaults to False.
        
        Attributes:
            q_network: Main Q-value network.
            q_network_target: Target network for stable learning.
            buffer: Experience replay buffer.
        """

        super().__init__(env, gamma)
        self.max_size = max_size
        self.double_learning = double_learning
        self.q_network, self.q_network_target = self.make_networks()
        self.buffer = Memory(max_size)

    def make_networks(self) -> tuple[nn.Module, nn.Module]:
        """Create the main and target Q-value networks.
        
        Must be implemented by subclasses to specify network architecture.
        
        Returns:
            A tuple containing:
                - q_network: Main network for learning.
                - q_network_target: Target network for stable TD updates.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError
    
    def action_explore(self, state: np.array, epsilon: float) -> int:
        """Select an action using epsilon-greedy exploration.
        
        Args:
            state: NumPy array representing the current state.
            epsilon: Exploration rate in [0, 1] - probability of random action.
        
        Returns:
            Integer action selected by epsilon-greedy policy.
        """
        
        bern = np.random.binomial(1, epsilon)
        if bern == 1: return self.env.action_space.sample()

        return self.action(state)

    def action(self, state: np.array) -> int:
        """Select the action with the highest Q-value.
        
        Args:
            state: NumPy array representing the current state.
        
        Returns:
            Integer action with the maximum Q-value.
        """
        
        with torch.no_grad():
            q_values = self.q_network(torch.from_numpy(state))
            return torch.argmax(q_values).item()
        
    def sample_batch(self, batch_size: int) -> ExperienceSample:
        """Sample a batch of transitions from the replay buffer.
        
        Args:
            batch_size: Number of transitions to sample.
        
        Returns:
            A tuple containing:
                - states_batch: Tensor of shape (batch_size, state_shape).
                - actions_batch: Array of shape (batch_size,).
                - rewards_batch: Tensor of shape (batch_size,).
                - next_states_batch: Tensor of shape (batch_size, state_shape).
                - dones_batch: Tensor of shape (batch_size,) with terminal flags.
        """

        batch = self.buffer.sample(batch_size)
        states_batch = torch.tensor(np.stack(batch.state), dtype=torch.float32)
        actions_batch = np.array(batch.action)
        rewards_batch = torch.tensor(batch.reward, dtype=torch.float32)
        next_states_batch = torch.tensor(np.stack(batch.next_state), dtype=torch.float32)
        dones_batch = torch.tensor(batch.done, dtype=torch.int)
        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch
        
    def update_q_network(self, 
                         batch_size: int, 
                         optimizer: optim, 
                         writer: SummaryWriter, 
                         it: int
                         ) -> None:
        """Update the Q-network using a batch from the replay buffer.
        
        Computes TD loss and performs gradient update.
        
        Args:
            batch_size: Number of transitions to sample for the update.
            optimizer: PyTorch optimizer for the Q-network.
            writer: TensorBoard writer for logging.
            it: Current iteration number for logging.
        """

        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = self.sample_batch(batch_size)
        q_values_states = self.q_network(states_batch)
        q_values = q_values_states[np.arange(batch_size), actions_batch]
        with torch.no_grad(): 
            q_values_next_states = self.q_network_target(next_states_batch)
            # DQN targets
            if not self.double_learning:
                q_targets = rewards_batch + self.gamma*torch.max(q_values_next_states, dim=1).values*dones_batch

            # Double DQN target.
            else:
                actions_max = torch.argmax(q_values_states, dim=1)
                q_targets = rewards_batch + self.gamma*q_values_next_states[np.arange(batch_size), actions_max]*dones_batch

        loss = F.mse_loss(q_values, q_targets)
        writer.add_scalar('loss', loss.item(), it)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def update_q_network_target(self):
        """
        Description
        -------------
        Update the weights of the q network.
        
        Arguments
        -------------
        batch_size : Int, the batch size.
        
        Returns
        --------------
        """

        self.q_network_target.load_state_dict(self.q_network.state_dict())

    def unroll(self, 
               epsilon_start: int, 
               epsilon_stop: int, 
               epsilon: float, 
               decay_rate: float, 
               it: int, 
               n_learn: int, 
               tau: int, 
               batch_size: int, 
               optimizer: optim, 
               writer: SummaryWriter
               ) -> None:
        """
        Description
        --------------
        Unroll an episode.
        
        Arguments
        --------------
        
        Returns
        --------------
        """

        state, _ = self.env.reset()
        done = False
        reward_episode = 0
        while not done:
            action = self.action_explore(state, epsilon)
            it += 1
            epsilon = update_epsilon(epsilon_start, epsilon_stop, decay_rate, it)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            reward_episode += reward
            done = (terminated or truncated)
            self.buffer.add(state, action, reward, next_state, 1 - done)
            state = next_state
            
            # Learning phase
            if it%n_learn == 0: self.update_q_network(batch_size, optimizer, writer, it)

            # Target network update phase
            if it%tau == 0: self.update_q_network_target()

        return reward_episode, epsilon, it
    
    def pretrain(self, n_pretrain: int = 1000, print_iter_pretrain: int = 100) -> None:
        """
        Description
        --------------
        Pre-fill the replay buffer by running a pretraining phase where actions are taken uniformly at random.
        
        Arguments
        --------------
        n_episodes : Int, number of episodes in the pretraining phase.
        
        Returns
        --------------
        """
        
        for episode in range(n_pretrain):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = (terminated or truncated)
                self.buffer.add(state, action, reward, next_state, 1 - done)
                state = next_state
                
            if episode%print_iter_pretrain == 0:
                print('pretrain episode : %d' %episode)
                
    def train(self, n_episodes: int = 1000, 
              n_pretrain: int = 100, 
              epsilon_start: float = 1., 
              epsilon_stop: float = 0.01, 
              decay_rate: float = 1e-3, 
              n_learn: int = 5, 
              tau: int = 50, 
              batch_size: int = 32, 
              lr: float = 1e-4, 
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
        n_pretrain    : Int, number of episodes in the pretraining phase.
        epsilon_start : Float in [0, 1], initial value of epsilon.
        epsilon_stop  : Float in [0, 1], final value of epsilon.
        decay_rate    : Float in [0, 1], decay rate of epsilon.
        n_learn       : Int, number of iterations between two consecutive updates of the main network.
        tau           : Int, number of iterations between two consecutive updates of the target network.
        batch_size    : Int, size of the batch to sample from the replay buffer.
        lr            : Float, learning rate.
        max_tau       : Int, number of iterations between two consecutive updates of the target network.
        thresh        : Float, lower bound on the average of the last 10 training episodes above which early stopping is activated.
        file_save     : String, name of the file containingt the saved network weights.
        print_iter    : Int, number of episodes between two consecutive prints.
        
        Returns
        --------------
        """

        optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        writer = SummaryWriter(log_dir=log_dir)

        # Pretraining.
        self.pretrain(n_pretrain)
        epsilon, it = epsilon_start, 0

        # Training.
        rewards = deque(maxlen=100)
        reward_mean = None
        for episode in range(n_episodes):
            reward_episode, epsilon, it = self.unroll(epsilon_start, epsilon_stop, epsilon, decay_rate, it, n_learn, tau, batch_size, optimizer, writer)
            writer.add_scalar('return', reward_episode, episode)
            if len(rewards) < rewards.maxlen:
                rewards.append(reward_episode)

            else:
                reward_mean = np.mean(rewards) if reward_mean is None else reward_mean + (reward_episode - rewards[0])/rewards.maxlen
                rewards.append(reward_episode)
                if reward_mean >= thresh:
                    print('Early stopping achieved after %d episodes' %episode)
                    return
                
            if episode%print_iter == 0 and reward_mean is not None:
                print('Episode : %d, epsilon : %.3f, return : %.3F' %(episode, epsilon, reward_mean))
            
        print('Training finished without early stopping.')

    def save(self, path: str) -> None:
        """
        Description
        --------------
        Save the weights of the main network.
        
        Parameters
        --------------
        path: String, path to the weights of the main network.
        
        Returns
        --------------
        """
        
        torch.save(self.q_network.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Description
        --------------
        Load the weights of the main network.
        
        Parameters
        --------------
        path: String, path to the weights of the main network.
        
        Returns
        --------------
        """
        
        self.q_network.load_state_dict(torch.load(path))

