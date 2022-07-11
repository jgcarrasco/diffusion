from bs4 import BeautifulSoup
import torch
import torch.nn as nn

from torch_rbf import *
from utils import *

class SwissRollModel(nn.Module):
    """
    Simple model used to approximate the reverse diffusion process on
    a the swiss roll distribution. It consists on a first layer with 16 hidden 
    units which is shared across all timesteps, and a different second layer
    with two unit vectors as output. 
    The reason for having a separate second layer for each timestep 
    is to include the temporal dependency. 
    """
    def __init__(self, input_size=2, T=40):
        super(SwissRollModel, self).__init__()

        self.first_layer = RBF(input_size, 16, basis_func_dict()['gaussian'])
        self.second_layers = nn.ModuleList(
            [nn.Linear(16, input_size) for _ in range(T)]
            )

    def forward(self, x, t):
        x = self.first_layer(x)
        x = self.second_layers[t](x)
        return x

class Diffusion(nn.Module):
    """
    Diffusion model, composed by an analytical forward diffusion process, and 
    a reverse diffusion process which is learned by a NN.
    """
    def __init__(self, model, timesteps=100, beta_1=1e-5, beta_T=1e-2):
        super(Diffusion, self).__init__()
        # Precompute the betas, alphas and useful values that will be used
        self.betas = schedule_variances(beta_1=beta_1, beta_T=beta_T, 
                                        T=timesteps, mode='sigmoid')
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, -1)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        # Model of the reverse diffusion process
        self.model = model
        self.timesteps = timesteps
    
    def forward(self, x, t):
        """
        The forward pass implements the predicted noise of image x_t at 
        timestep t
        """
        return self.model(x, t)

    def sample_q(self, x_0, t):
        z = torch.randn_like(x_0)
        return extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 +\
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * z

    

# Testing code
if __name__ == "__main__":
    simple_model = SwissRollModel()
    input_test = torch.rand(2)
    input_test = torch.unsqueeze(input_test, 0)
    print("Input shape: ", input_test.size())
    print("Output shape at t = 10", simple_model(input_test, 10).size())
