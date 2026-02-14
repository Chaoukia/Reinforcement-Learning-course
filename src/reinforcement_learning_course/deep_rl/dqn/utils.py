import numpy as np
import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
    
class Memory:
    """Uniform experience replay memory buffer.
    
    Stores transitions and provides sampling functionality for off-policy
    learning with experience replay.
    """
    
    def __init__(self, max_size):
        """Initialize the replay buffer.
        
        Args:
            max_size: Maximum number of transitions to store in the buffer.
        
        Attributes:
            buffer: Deque container with fixed maximum capacity.
        """
        
        self.buffer = deque(maxlen = max_size)
    
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the replay buffer.
        
        Args:
            state: Observation from the environment.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting observation.
            done: Whether the episode terminated.
        """

        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Randomly sample transitions from the replay buffer.
        
        Args:
            batch_size: Number of transitions to sample.
        
        Returns:
            Named tuple Transition with fields (state, action, reward, next_state, done),
            where each field is iterable over the batch.
        """
        
        transitions = random.sample(self.buffer, batch_size)
        return Transition(*zip(*transitions))
    

def update_epsilon(epsilon_start, epsilon_stop, decay_rate, it):
    """Compute epsilon value with exponential decay schedule.
    
    Args:
        epsilon_start: Initial exploration rate.
        epsilon_stop: Final exploration rate.
        decay_rate: Exponential decay rate.
        it: Current iteration number.
    
    Returns:
        Float, the epsilon value for the current iteration.
    """

    return epsilon_stop + (epsilon_start - epsilon_stop)*np.exp(-decay_rate*it)


