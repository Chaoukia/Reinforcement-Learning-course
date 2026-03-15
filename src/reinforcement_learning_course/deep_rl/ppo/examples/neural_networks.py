import torch
import torch.nn as nn
from collections import OrderedDict


class LunarLanderPolicyNetwork(nn.Module):
    """
    LunarLander Policy Network.
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
    

class LunarLanderValueNetwork(nn.Module):
    """
    LunarLander Value Network.
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
                                              ('linear3', nn.Linear(256, 1))]))
        
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
    
    
class CartPolePolicyNetwork(nn.Module):
    """
    CartPole Policy Network.
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
    

class CartPoleValueNetwork(nn.Module):
    """
    CartPole Value Network.
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
                                              ('linear3', nn.Linear(256, 1))]))
        
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
    
    
class MountainCarPolicyNetwork(nn.Module):
    """
    MountainCar Policy Network.
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
    

class MountainCarValueNetwork(nn.Module):
    """
    MountainCar Value Network.
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
                                              ('linear3', nn.Linear(256, 1))]))
        
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
    
    
class AcrobotPolicyNetwork(nn.Module):
    """
    Acrobot Policy Network.
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
    

class AcrobotValueNetwork(nn.Module):
    """
    Acrobot Value Network.
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
                                              ('linear3', nn.Linear(256, 1))]))
        
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