import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means
        """
        # super(SimpleNet, self).__init__()
        super().__init__()
        

        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

        ############################################################################
        # Student code begin
        ############################################################################

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, 5),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.Conv2d(10, 20, 5),
            nn.MaxPool2d(3),
            nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(500, 100),
            nn.Linear(100, 15)  
        )       
        
        # self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')
        
        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        model_output = None
        ############################################################################
        # Student code begin
        ############################################################################
        
        o = self.conv_layers(x)
        
        # print("network output 1: ", o.size())
        
        
        model_output = self.fc_layers(o.reshape(-1, 500))
        
        # print("network output 2: ", model_output.size())
        

        ############################################################################
        # Student code end
        ############################################################################

        return model_output
