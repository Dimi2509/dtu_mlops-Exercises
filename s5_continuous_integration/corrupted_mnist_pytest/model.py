import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    """MNIST dataset, create a CNN. Input image is 1x28x28"""

    def __init__(self, k=10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)  
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)  
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)  
        self.fc1 = nn.Linear(in_features=128 * 1 * 1, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=k)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')
        
        x = torch.relu(self.conv1(x))                     # 32x26x26
        x = torch.max_pool2d(x, kernel_size=2, stride=2)  # 32x13x13
        x = torch.relu(self.conv2(x))                     # 64x11x11
        x = torch.max_pool2d(x, kernel_size=2, stride=2)  # 64x5x5
        x = torch.relu(self.conv3(x))                     # 128x3x3
        x = torch.max_pool2d(x, kernel_size=2, stride=2)  # 128x1x1
        # print(f"x.shape before flatten: {x.shape}")
        x = torch.flatten(
            x, start_dim=1
        )  # Inpput is 1x128x1x1 (batch size 1, 128 channels, 1x1 image), flattening from dimension 1 outputs 1x128
        # print(f"x.shape after flatten: {x.shape}")

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = MyAwesomeModel(k=10)
    print(model)
    sample_input = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output = model(sample_input)
    print(f"Output shape: {output.shape}")
