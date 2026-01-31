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
                                      ('relu2', nn.ReLU())]))
        self.policy_head = nn.Sequential(OrderedDict([('linear', nn.Linear(256, 4))]))
        self.value_head = nn.Sequential(OrderedDict([('linear', nn.Linear(256, 1))]))
        
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

        features = self.mlp(x)
        actions_logits, value = self.policy_head(features), self.value_head(features)
        return actions_logits, value
    
class CartPoleNetwork(nn.Module):

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
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(4, 512)), 
                                      ('relu1', nn.ReLU()), 
                                      ('linear2', nn.Linear(512, 256)), 
                                      ('relu2', nn.ReLU())]))
        self.policy_head = nn.Sequential(OrderedDict([('linear', nn.Linear(256, 2))]))
        self.value_head = nn.Sequential(OrderedDict([('linear', nn.Linear(256, 1))]))
        
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

        features = self.mlp(x)
        actions_logits, value = self.policy_head(features), self.value_head(features)
        return actions_logits, value
    
class MountainCarNetwork(nn.Module):

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
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(2, 512)), 
                                      ('relu1', nn.ReLU()), 
                                      ('linear2', nn.Linear(512, 256)), 
                                      ('relu2', nn.ReLU())]))
        self.policy_head = nn.Sequential(OrderedDict([('linear', nn.Linear(256, 3))]))
        self.value_head = nn.Sequential(OrderedDict([('linear', nn.Linear(256, 1))]))
        
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

        features = self.mlp(x)
        actions_logits, value = self.policy_head(features), self.value_head(features)
        return actions_logits, value
    
class AcrobotNetwork(nn.Module):

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
        self.mlp = nn.Sequential(OrderedDict([('linear1', nn.Linear(6, 512)), 
                                      ('relu1', nn.ReLU()), 
                                      ('linear2', nn.Linear(512, 256)), 
                                      ('relu2', nn.ReLU())]))
        self.policy_head = nn.Sequential(OrderedDict([('linear', nn.Linear(256, 3))]))
        self.value_head = nn.Sequential(OrderedDict([('linear', nn.Linear(256, 1))]))
        
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

        features = self.mlp(x)
        actions_logits, value = self.policy_head(features), self.value_head(features)
        return actions_logits, value