import torch
import torch.nn as nn

# Sourced from HW 2 (ECE661) written previously in the semester

# Resblock definition
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()

        # 2 convolutional layers are used in the "basic" ResBlock
        self.conv1 = nn.Sequential(
            # Convolution -> Batch Norm -> ReLU
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            # Convolution -> Batch Norm
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        # Downsample (if required) and perform ReLU activation
        # This is to ensure that the output and input can add together
        self.downsample = downsample

        self.relu = nn.ReLU(inplace=True) # Perform ReLU
        self.out_channels = out_channels # Save output channels (used above)

    def forward(self, x):
        res = x  # Define the residual connection
        out = self.conv1(x)  # First set of layers
        out = self.conv2(out)  # Second set of layers
        if self.downsample is not None:
            res = self.downsample(x)  # Downsample if selected
        out += res  # Add residual to output of the block
        out = self.relu(out)  # Perform activation
        return out
    
# ResNet 20: 1 conv -> 3 Resnet Layers -> 1 FC Layer
class ResNet20(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_channels = 16

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        # ResNet Layers (Individual layers 2 - 19):
        self.layer_0 = self.make_layer(block, 16, layers[0], stride=1)
        self.layer_1 = self.make_layer(block, 32, layers[1], stride=2)
        self.layer_2 = self.make_layer(block, 64, layers[2], stride=2)

        # Classifer (Indiviudal layer 20)
        # Total individual layers = 20
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Suggested in documentation as a safer alternative
        self.fc1 = nn.Linear(64, num_classes)

    # Makes each layer of the ResNet using the a selected "block"
    # In this case, my ResBlock will be used as input into this function
    def make_layer(self, block, out_channels, blocks, stride=1):
        
        # Downsample if required
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            downsample = None

        # Append ResBlocks to create the ResNet layer
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride=stride, downsample=downsample)
        )
        self.in_channels = out_channels
        # Adds a block for each of the 
        for _ in range(1, blocks):
            layers.append(
                block(self.in_channels, out_channels, stride=1, downsample=None)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)  # Passes first through the convolutional layer
        # ResNet Layers 1 - 3
        output = self.layer_0(output)
        output = self.layer_1(output)
        output = self.layer_2(output)
        output = self.avgpool(output)  # Average Pool
        output = torch.flatten(
            output, 1
        )  # Flatten and then pass the input in the FC layer
        output = self.fc1(output)
        return output
