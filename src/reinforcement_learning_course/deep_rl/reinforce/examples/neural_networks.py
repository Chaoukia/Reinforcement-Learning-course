import torch
import torch.nn as nn
from collections import OrderedDict


class LunarLanderNetwork(nn.Module):
    """
    LunarLander Deep Q-Network.
    """

    def __init__(self) -> None:
        """
        Description
        -------------------------
        Constructor.

        Parameters
        -------------------------

        Returns
        -------------------------
        """

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(8, 512)), 
                                              ('relu1', nn.ReLU()), 
                                              ('linear2', nn.Linear(512, 256)), 
                                              ('relu2', nn.ReLU()), 
                                              ('linear3', nn.Linear(256, 4))]))
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Description
        -------------------------
        Run a forward propagation.

        Parameters
        -------------------------
        x : torch.tensor, input state

        Returns
        -------------------------
        torch.tensor, output q-values for the state.
        """

        return self.mlp(x)
    
    
class CartPoleNetwork(nn.Module):
    """
    CartPole Deep Q-Network.
    """

    def __init__(self):
        """
        Description
        -------------------------
        Constructor.

        Parameters
        -------------------------

        Returns
        -------------------------
        """

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(4, 512)), 
                                              ('relu1', nn.ReLU()), 
                                              ('linear2', nn.Linear(512, 256)), 
                                              ('relu2', nn.ReLU()), 
                                              ('linear3', nn.Linear(256, 2))]))
        
    def forward(self, x):
        """
        Description
        -------------------------
        Run a forward propagation.

        Parameters
        -------------------------
        x : torch.tensor, input state

        Returns
        -------------------------
        torch.tensor, output q-values for the state.
        """

        return self.mlp(x)
    
    
class MountainCarNetwork(nn.Module):
    """
    MountainCar Deep Q-Network.
    """

    def __init__(self):
        """
        Description
        -------------------------
        Constructor.

        Parameters
        -------------------------

        Returns
        -------------------------
        """

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(2, 512)), 
                                              ('relu1', nn.ReLU()), 
                                              ('linear2', nn.Linear(512, 256)), 
                                              ('relu2', nn.ReLU()), 
                                              ('linear3', nn.Linear(256, 3))]))
        
    def forward(self, x):
        """
        Description
        -------------------------
        Run a forward propagation.

        Parameters
        -------------------------
        x : torch.tensor, input state

        Returns
        -------------------------
        torch.tensor, output q-values for the state.
        """

        return self.mlp(x)
    
    
class AcrobotNetwork(nn.Module):
    """
    Acrobot Deep Q-Network.
    """

    def __init__(self):
        """
        Description
        -------------------------
        Constructor.

        Parameters
        -------------------------

        Returns
        -------------------------
        """

        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(6, 512)), 
                                              ('relu1', nn.ReLU()), 
                                              ('linear2', nn.Linear(512, 256)), 
                                              ('relu2', nn.ReLU()), 
                                              ('linear3', nn.Linear(256, 3))]))
        
    def forward(self, x):
        """
        Description
        -------------------------
        Run a forward propagation.

        Parameters
        -------------------------
        x : torch.tensor, input state

        Returns
        -------------------------
        torch.tensor, output q-values for the state.
        """

        return self.mlp(x)