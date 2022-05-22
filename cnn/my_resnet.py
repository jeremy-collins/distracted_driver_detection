import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet152, efficientnet_b4


class MyResNet18(nn.Module):
    def __init__(self):
        """Initialize network layers.

        Note: Do not forget to freeze the layers of ResNet except the last one
        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Download pretrained resnet using pytorch's API (Hint: see the import statements)
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None

        model = resnet18(pretrained=True)
        # model = efficientnet_b4(pretrained=True)
        # model = resnet152(pretrained=True)

        
        # print(list(model.children()))
        
        self.conv_layers = nn.Sequential(
            *list(model.children())[:-1]
            )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 128),
            # nn.Linear(2048, 128),
            # nn.Linear(1792, 128),


            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(512, 128),

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
        
        # freezing the layers of ResNet except the last one
        # for param in self.conv_layers.parameters():
        #     param.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass with the net, duplicating grayscale channel to 3-channel.

        Args:
            x: tensor of shape (N,C,H,W) representing input batch of images
        Returns:
            y: tensor of shape (N,num_classes) representing the output (raw scores) of the net
                Note: we set num_classes=15
        """
        model_output = None
        # x = x.repeat(1, 3, 1, 1)  # as ResNet accepts 3-channel color images
        
        o = self.conv_layers(x)
        
        # print("network output 1: ", o.size())
        
        model_output = self.fc_layers(o.reshape(-1, 512))
        # model_output = self.fc_layers(o.reshape(-1, 2048))
        # model_output = self.fc_layers(o.reshape(-1, 1792))


        # print("network output 2: ", model_output.size())

        return model_output
