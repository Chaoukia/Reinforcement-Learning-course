import torch.nn as nn
from collections import OrderedDict

class LunarLanderNetwork(nn.Module):

    """
    Description
    -------------------------
    Class describing the Q network for a LunarLander agent.
    """

    def __init__(self):
        """
        Description
        -------------------------
        Constructor of class DeepBranches.

        Parameters & Attributes
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
        
    def forward(self, x):
        """
        Description
        -------------------------
        Run a forward propagation through the MLP.

        Parameters & Attributes
        -------------------------

        Returns
        -------------------------
        """

        return self.mlp(x)