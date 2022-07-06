import torch
import torch.nn as nn

from torch_rbf import *
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

        self.first_layer = RBF(input_size, 16, basis_func_dict()['gaussian'])
        self.second_layers = nn.ModuleList([nn.Linear(16, input_size) for _ in range(T)])

    def forward(self, x, t):
        x = self.first_layer(x)
        x = self.second_layers[t](x)
        return x

    # def to(self, *args, **kwargs):
    #     self = super().to(*args, **kwargs)
    #     self.first_layer = self.first_layer.to(*args, **kwargs)
    #     for layer in self.second_layers:
    #         layer = layer.to(*args, **kwargs)
    #     return self


# https://github.com/acids-ircam/diffusion_models/blob/main/diffusion_03_waveform.ipynb
import torch.nn.functional as F
class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out

class ConditionalModel(nn.Module):
    def __init__(self, n_steps):
        super(ConditionalModel, self).__init__()
        self.lin1 = ConditionalLinear(2, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = nn.Linear(128, 2)
    
    def forward(self, x, y):
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        return self.lin3(x)

# Testing code
if __name__ == "__main__":
    simple_model = SwissRollModel()
    input_test = torch.rand(2)
    input_test = torch.unsqueeze(input_test, 0)
    print("Input shape: ", input_test.size())
    print("Output shape at t = 10", simple_model(input_test, 10).size())
