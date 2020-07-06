"""Loss function module"""

import torch
import torch.nn as nn

class RMSLELoss(nn.Module):
    """
    RMS Log Error function
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        """
        calling function for the loss
        """
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
