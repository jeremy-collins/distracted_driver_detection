import torch
import torch.nn as nn


class OptimalNet(nn.Module):
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

        # self.conv_layers = nn.Sequential(
        #     # 1x224x224
        #     # (in, out, kernel_size)
        #     nn.Conv2d(1, 32, 3, padding=1),
        #     nn.MaxPool2d(2),
        #     nn.SiLU(),
            
        #     # 32x112x112
        #     nn.Conv2d(32, 16, 3, padding=1),
        #     # nn.MaxPool2d(2),
        #     nn.SiLU(),
        
        #     # 32x112x112
        #     nn.Conv2d(16, 24, 3, padding=1),
        #     nn.MaxPool2d(2),
        #     nn.SiLU(),
            
        #     # 24x56x56
        #     nn.Conv2d(24, 40, 5, padding=2),
        #     nn.MaxPool2d(2),
        #     nn.SiLU(),
            
        #     # 40x28x28
        #     nn.Conv2d(40, 80, 3, padding=1),
        #     nn.MaxPool2d(2),
        #     nn.SiLU(),
            
        #     # 80x14x14
        #     nn.Conv2d(80, 112, 5, padding=2),
        #     # nn.MaxPool2d(2),
        #     nn.SiLU(),
            
        #     # 112x14x14
        #     nn.Conv2d(112, 192, 5, padding=2),
        #     nn.MaxPool2d(2),
        #     nn.SiLU(),

        #     # 192x7x7
        #     nn.Conv2d(192, 320, 3, padding=0),
        #     nn.MaxPool2d(2),
        #     nn.SiLU(),
            
        #     # 320x6x6
        #     nn.Conv2d(320, 1280, 1, padding=0),
        #     nn.MaxPool2d(2),
        #     nn.SiLU(),
        # )
        
        # self.fc_layers = nn.Sequential(
        #     nn.Linear(40960, 100),
        #     # nn.Linear(256, 100),
        #     nn.Linear(100, 15)  
        # )      
        
        self.conv_layers = nn.Sequential(
            # 1x64x64
            nn.Conv2d(1, 16, 5, padding=2),
            # batch normalization
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.SiLU(),
            
            # 16x32x32
            nn.Conv2d(16, 32, 3, padding=1),
            # batch normalization
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.SiLU(),
            
            # 32x16x16
            nn.Conv2d(32, 64, 3, padding=1),
            # batch normalization
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.SiLU(),
            
            # 64x8x8
            nn.Conv2d(64, 128, 3, padding=1),
            # batch normalization
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.SiLU(),
            
            # 128x4x4
            # dropout layer
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, 3, padding=1),
            # batch normalization
            # nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.SiLU(),
            
            # # 32x6x6
            # #dropout layer
            # nn.Dropout(0.5),
            # nn.Conv2d(40, 80, 3),
            # nn.MaxPool2d(2),
            # nn.ReLU()
            # # output is 64x2x2
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256*2*2, 100),
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
        
        # print(o.shape)
        
        model_output = self.fc_layers(o.reshape(-1, 256*2*2))
        
        ############################################################################
        # Student code end
        ############################################################################

        return model_output
