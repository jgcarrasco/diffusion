import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import utils


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
        self.lin3 = ConditionalLinear(128, 4, n_steps)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, y):
        x = self.lin1(x, y)
        x = self.lin2(x, y)
        x = self.lin3(x, y)
        # Split into mu and sigma
        mu, sigma = torch.split(x, 2, dim=-1)
        # Pass sigma through a sigmoid layer in order to restrict it to [0, 1]
        sigma = self.sigmoid(sigma)
        return mu, sigma


class Diffusion:
    def __init__(self, model=None, timesteps=40):
        """
        Constructor for the diffusion class. The variance schedule of the
        forward diffusion process as well as another values derived from it
        will be precomputed and stored.

        Parameters
        ----------
        model : `nn.Module`
            Model used to predict the mean and stdev of the reversed diffusion
            process. It must return two tensors, mu and sigma, of the same
            shape of the input tensor. By default, a `ConditionalModel` is used.
        timesteps : int
            Number of timesteps for the diffusion process.
        """
        # Precompute betas and other values
        self.betas = utils.schedule_variances(timesteps=timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, -1)

        self.timesteps = timesteps
        if model:
            self.model = model
        else:
            self.model = ConditionalModel(n_steps=timesteps)

    
if __name__ == "__main__":
    x = torch.randn((10, 2))
    model = ConditionalModel(n_steps=40)
    mu, sigma = model(x, torch.tensor([39]))
    print("Input shape: ", x.shape)
    print("Output shapes (mu, sigma): ", mu.shape, sigma.shape)

    diffusion = Diffusion()
    plt.plot(diffusion.betas, label="betas")
    plt.plot(diffusion.alphas, label="alphas")
    plt.plot(diffusion.alphas_cumprod, label="alphas_cumprod")
    plt.legend()
    plt.show()