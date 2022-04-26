import torch
import torch.nn as nn


class SimpleNetFinal(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means
        """
        # super(SimpleNetFinal, self).__init__()
        super().__init__()
        

        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

        ############################################################################
        # Student code begin
        ############################################################################

        self.conv_layers = nn.Sequential(
            # 1x64x64
            nn.Conv2d(1, 8, 5),
            # batch normalization
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # 8x30x30
            nn.Conv2d(8, 16, 3),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # 16x14x14
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # 32x6x6
            #dropout layer
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2),
            nn.ReLU()
            # output is 64x2x2
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, 10)  
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
        
        model_output = self.fc_layers(o.reshape(-1, 256))
        
        ############################################################################
        # Student code end
        ############################################################################

        return model_output
