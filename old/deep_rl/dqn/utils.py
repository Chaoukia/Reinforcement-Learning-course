import numpy as np
import torch
import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
    
class Memory:
    
    """
    Class of the uniform experience replay memory.
    """
    
    def __init__(self, max_size):
        """
        Description
        -------------------------
        Constructor of class Memory.
        
        Arguments & Attributes
        -------------------------
        max_size   : Int, maximum size of the replay memory.
        buffer     : collections.deque object of maximum length max_size, the container representing the replay memory.
        """
        
        self.buffer = deque(maxlen = max_size)
    
    def add(self, state, action, reward, next_state, done):
        """
        Description
        -------------
        Add experience to the replay buffer.
        
        Arguments
        -------------
        state      : np.array, state from a gym environment.
        action     : Int, action.
        reward     : Float, reward of the transition.
        next_state : np.array, next state of the transition.
                     
        Returns
        -------------
        """

        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Description
        -------------
        Randomly sample "batch_size" transitions from the replay buffer.
        
        Arguments
        -------------
        batch_size : Int, the number of transitions to sample.
        
        Returns
        -------------
        Named tuple, the sampled batch.
        """
        
        transitions = random.sample(self.buffer, batch_size)
        return Transition(*zip(*transitions))