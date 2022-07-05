import torch
import torch.nn as nn

from utils import *

class SwissRollModel(nn.Module):
    """
    Simple diffusion model used to approximate the swiss roll 
    distribution. It consists on a first layer with 16 hidden units
    which is shared across all timesteps, and a different second layer
    with two unit vectors as output. 
    The reason for having a separate second layer for each timestep 
    is to include the temporal dependency. 
    """
    def __init__(self, input_size=2, T=40):
        super(SwissRollModel, self).__init__()

        self.first_layer = nn.Sequential(nn.Linear(input_size, 16),
                                    nn.ReLU(inplace=True))
        self.second_layers = [nn.Linear(16, input_size) for _ in range(T)]

    def forward(self, x, t):
        x = self.first_layer(x)
        x = self.second_layers[t](x)
        return x

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        for layer in self.second_layers:
            layer = layer.to(*args, **kwargs)
        return self

# Testing code
if __name__ == "__main__":
    simple_model = SwissRollModel()
    input_test = torch.rand(2)
    print("Input shape: ", input_test.size())
    print("Output shape at t = 10", simple_model(input_test, 10).size())
