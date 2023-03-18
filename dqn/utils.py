import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import animation
from collections import deque
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQNetwork(nn.Module):
    
    """
    Class of the neural network estimating the Q values in DQN.
    """
    
    def __init__(self, input_size=8, fc_1_dim=512, fc_2_dim=256, out=4):
        """
        Description
        -------------------------
        Constructor of DQNetwork class.
        
        Arguments & Attributes
        -------------------------
        input_size : Int, dimension of the state space.
        out        : Int, output dimension, equal to the number of possible actions.
        fc_1       : nn.Linear, first fully connected layer.
        fc_2       : nn.Linear, second fully connected layer.
        output     : nn.Linear, output fully connected layer.
        """
        
        super(DQNetwork, self).__init__()
        
        self.fc_1 = nn.Linear(input_size, fc_1_dim)
        self.fc_2 = nn.Linear(fc_1_dim, fc_2_dim)
        self.output = nn.Linear(fc_2_dim, out)
                
    def forward(self, x):
        """
        Description
        ---------------
        The forward pass.
        
        Arguments
        ---------------
        x : torch.tensor of dimension (batch_size, input_size)
        
        Returns
        ---------------
        torch.tensor of dimension (batch_size, out)
        """
        
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        return self.output(x)
    
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
        
        self.buffer.append((torch.from_numpy(state.reshape((1, -1))), [action], torch.tensor([reward], dtype=torch.float32), 
                            torch.from_numpy(next_state.reshape((1, -1))), torch.tensor([done], dtype=torch.int)))
    
    def sample(self, batch_size):
        """
        Description
        -------------
        Randomly sample "batch_size" transitions from the replay buffer.
        
        Arguments
        -------------
        batch_size : Int, the number of tarnsitions to sample.
        
        Returns
        -------------
        Named tuple, the sampled batch.
        """
        
        transitions = random.sample(self.buffer, batch_size)
        return Transition(*zip(*transitions))