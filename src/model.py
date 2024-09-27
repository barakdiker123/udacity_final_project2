import torch.nn.init as init
from typing import Any, Callable, List, Optional, Type, Union
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel2(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super(MyModel2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find("Linear") != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)


class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.2) -> None:
        super(MyModel, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=3, padding=1),
            # Add batch normalization (BatchNorm2d) here
            # YOUR CODE HERE
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 64, 3, padding=1),  # -> 32x16x16
            # Add batch normalization (BatchNorm2d) here
            # YOUR CODE HERE
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 32x8x8
            # nn.Conv2d(32, 64, 3, padding=1),  # -> 64x8x8
            # Add batch normalization (BatchNorm2d) here
            # YOUR CODE HERE
            nn.MaxPool2d(2, 2),  # -> 64x4x4
            nn.Conv2d(64, 128, 3, padding=1),  # -> 128x4x4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 128x28*28 -> 128*14*14
            nn.Conv2d(128, 256, 3, padding=1),  # -> 256x14x14
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 384, 3, padding=1),  # -> 256x14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 384*14*14-> 384*7*7
            # nn.AdaptiveAvgPool2d((7, 7)),
            # nn.AvgPool2d(7, 7),
            nn.Flatten(),  # -> 1x128x2x2 , 1*128*14*14
            nn.Linear(384 * 7 * 7, 4096),  # 256*7*7
            nn.BatchNorm1d(4096),
            nn.Dropout(dropout),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):

        # Just call the model on x here:
        # YOUR CODE HERE
        return self.model(x)


# define the CNN architecture


class MyModel1(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 2, kernel_size=5)
        self.dense1 = nn.Linear(in_features=56180, out_features=50)
        self.dense1_bn = nn.BatchNorm1d(50)
        self.dense2 = nn.Linear(50, num_classes)

    # self.conv1 = nn.Sequential(
    #    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    #    nn.BatchNorm2d(16),
    #    nn.MaxPool2d(2, 2),
    #    nn.ReLU(),
    #    nn.Dropout2d(0.2),
    # )

    # YOUR CODE HERE
    # Define a CNN architecture. Remember to use the variable num_classes
    # to size appropriately the output of your classifier, and if you use
    # the Dropout layer, use the variable "dropout" to indicate how much
    # to use (like nn.Dropout(p=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        x = x.view(-1, 56180)  # reshape
        x = F.relu(self.dense1_bn(self.dense1(x)))
        x = F.relu(self.dense2(x))
        return F.log_softmax(x, dim=1)
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        # return x


######################################################################################
#                                     TESTS
######################################################################################


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"


def initialize_weights(model):
    # Loop through each layer in the model
    for layer in model.modules():
        # Initialize convolutional layers
        if isinstance(layer, nn.Conv2d):
            # Randomize weights using Kaiming He initialization
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, 0)  # Set biases to zero

        # Initialize fully connected layers
        elif isinstance(layer, nn.Linear):
            # Randomize weights using Xavier initialization
            init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, 0)
    for layer in model.modules():
        # Randomize weights using Xavier initialization
        init.xavier_normal_(layer.weight)
