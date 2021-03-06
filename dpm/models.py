import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import utils
from utils import extract


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
    
class RBF(nn.Module):
    """
    RBF layer implemented in PyTorch. This layer differs from a simple Linear
    layer as it uses a radial basis function as an activation function, i.e.
    the output of the layer only depends on the distance of the input and a 
    learned center. 

    More formally, each output of the RBF will have the shape

    o_i(x) = phi(||x - c_i||) 

    where phi is a radial basis activation function (typically a gaussian) and 
    c_i is a learnable parameter for each neuron in the hidden layer. If a 
    gaussian is used as a radial basis function, then:

    o_i(x) = exp(-beta_i * ||x - c_i||)

    where b_i are also learnable parameters. As for the distance, the 
    Euclidean norm will be used.

    TO DO: Implement normalization
    """
    def __init__(self, in_features, out_features):
        """
        Constructor for the RBF layer

        Parameters
        ----------
        in_features: int
            Number of input features
        out_features: int
            Desired number of output features
        """
        super(RBF, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.centers = nn.parameter.Parameter(
            torch.empty((out_features, in_features))
            )
        self.betas = nn.parameter.Parameter(
            torch.ones(out_features)
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.centers, a=math.sqrt(5))

    def forward(self, x):
        """
        Forward pass for the RBF layer

        Parameters
        ----------
        x : `torch.Tensor`
            Input tensor of shape (N, `self.in_features`)
        
        Returns
        -------
        `torch.Tensor` of shape (N, `out_features`)
        """
        size = (x.size(0), self.out_features, self.in_features) # (N, out, in)
        # Broadcast the shape of x to (N, out, in)
        x = x.unsqueeze(1).expand(size)
        # The same for self.centers
        c = self.centers.unsqueeze(0).expand(size)
        distances = (x - c).square().sum(-1).sqrt() # (N, out)
        b = self.betas.unsqueeze(0).expand(distances.size())
        # Apply gaussian activation function
        return torch.exp(-b*distances)


class RBFModel(nn.Module):
    def __init__(self, timesteps=40):
        super(RBFModel, self).__init__()

        self.timesteps = timesteps
        self.rbf = RBF(2, 16)
        self.temporal_layer = ConditionalLinear(16, 4, timesteps)
        
    def forward(self, x, t):
        x = self.rbf(x)
        x = self.temporal_layer(x, t)
        mu, sigma = torch.split(x, 2, dim=-1)
        return mu, sigma
        

class Diffusion(nn.Module):
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
        super(Diffusion, self).__init__()

        # Precompute betas and other values
        self.betas = utils.schedule_variances(timesteps=timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, -1)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1]).float(), 
             self.alphas_cumprod[:-1]], 0)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        # Cumulative covariance from the entire forward trajectory
        self.beta_full_trajectory = 1. - self.alphas.log().sum().exp()

        self.timesteps = timesteps
        if model:
            self.model = model
        else:
            self.model = RBFModel(timesteps=timesteps)
    
    def sample_forward(self, x_0, t):
        """
        Implementation of the forward diffusion process from x_0 to x_t at 
        timestep t, q(x_t | x_0)

        Parameters
        ----------
        x_0 : `torch.Tensor`
            Tensor of shape (n_batch, 2), containing samples drawn from the
            original distribution.
        t : `torch.Tensor`
            Tensor of shape (n_batch, ), indicating from which timestep we
            want to sample, for every observation.

        Returns
        -------
        `torch.Tensor` of shape (n_batch, 2) containing the results of the 
        forward diffusion process.
        """
        z = torch.randn_like(x_0)
        return x_0 * extract(self.sqrt_alphas_cumprod, t, x_0) \
             + z * extract(self.sqrt_one_minus_alphas_cumprod, t, x_0)

    def sample_backward(self, x_t, t):
        """
        Implementation of the learned reverse diffusion process,
        p(x_t-1 | x_t)
        """
        mu, sigma = self.model(x_t, t)
        z = torch.rand_like(x_t)
        return mu + sigma*z
    
    def generate(self, n_samples=100, return_all=False):
        """
        Generates `n_samples` by sampling from a gaussian distribution and
        iteratively applying the learned reverse diffusion process.
        """
        X_t = []
        device = next(self.model.parameters()).device
        # Sample from a gaussian
        x_t = torch.randn((n_samples, 2)).to(device)
        X_t.append(x_t.unsqueeze(0))
        for t in range(self.timesteps-1, -1, -1):
            x_t = self.sample_backward(x_t, torch.tensor([t]).to(device))
            X_t.append(x_t.unsqueeze(0))
        
        X_t = torch.cat(X_t, dim=0)
        
        return X_t if return_all else x_t


    def compute_mu_sigma_posterior(self, x_0, x_t, t):
        """
        Compute the mean and stdev of the gaussian q(x_t-1 | x_t, x_0). 
        This will be used when computing the loss.
        """
        # Compute the mean
        a = (extract(self.sqrt_alphas_cumprod_prev, t, x_0) \
          *  extract(self.betas, t, x_0)) \
          / (1. - extract(self.alphas_cumprod, t, x_0))
        b = (extract(self.sqrt_alphas_cumprod, t, x_0) \
            * (1. - extract(self.alphas_cumprod_prev, t, x_0))) \
            / (1. - extract(self.alphas_cumprod, t, x_0))
        mu = a*x_0 + b*x_t
        # Compute the variance
        sigma2 = (1. - extract(self.alphas_cumprod_prev, t, x_0)) \
            / (1. - extract(self.alphas_cumprod, t, x_0)) * extract(self.betas, t, x_0)
        sigma = torch.sqrt(sigma2)

        return mu, sigma
    
    def compute_loss(self, x_0, t):
        """
        Compute the loss terms for a single batch.
        There is a loss term for each timestep. For each observation of the 
        batch, a timestep is specified.

        Parameters
        ----------
        x_0 : `torch.Tensor`
            Tensor of shape (n_batch, 2) containing the samples
        t : `torch.Tensor`
            Tensor of shape (n_batch,) containing the indices for each sample
        
        Returns
        -------
        Final loss term consisting on the mean of the independent loss terms.
        """
        x_t = self.sample_forward(x_0, t)
        mu, sigma = self.model(x_t, t)
        mu_posterior, sigma_posterior = self.compute_mu_sigma_posterior(x_0, x_t, t)
        # # KL divergence between q(x_t-1 | x_t, x_0) and p(x_t-1 | x_t)
        # # As both are gaussians, it can be solved in closed form
        # KL = (torch.log(sigma) - torch.log(sigma_posterior)
        # + (sigma_posterior**2 + (mu_posterior - mu)**2)/(2*sigma**2) - 0.5)
        # # Conditional entropies H_q(x^1|x^0) and H_q(x^T|x^0)
        # H_startpoint = 0.5*self.betas[0].log() \
        #     + 0.5 * (1 + np.log(2*torch.pi))
        # H_endpoint = 0.5*self.beta_full_trajectory.log() \
        #     + 0.5 * (1 + np.log(2*torch.pi))
        # # Differential entropy H_p(X_T)
        # H_prior = 0.5 * (1. + np.log(2*torch.pi))
        loss = sigma.log() + (sigma_posterior**2 + (mu_posterior-mu)**2) / (2*sigma**2)
        return loss.mean()

            
    #---------------
    # VISUALIZATION
    #---------------
    
    def plot_forward(self, x_0):
        """
        Plots the distribution at timesteps 0, T/2 and T, where T is the total
        number of timesteps.
        """

        x_Td2 = self.sample_forward(x_0, 
                                    torch.tensor([int((self.timesteps-1)/2)]))
        x_T = self.sample_forward(x_0, 
                                  torch.tensor([int(self.timesteps-1)]))

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(3*3, 3))
        ax1.scatter(x_0[:, 0], x_0[:, 1], alpha=0.5, s=5)
        ax1.set_title("t = " + str(0))
        ax2.scatter(x_Td2[:, 0], x_Td2[:, 1], alpha=0.5, s=5)
        ax2.set_title("t = " + str(int(self.timesteps/2)))
        ax3.scatter(x_T[:, 0], x_T[:, 1], alpha=0.5, s=5)
        ax3.set_title("t = " + str(self.timesteps))
        plt.show()
    
if __name__ == "__main__":
    x = torch.randn((10, 2))
    t = torch.randint(1, 10, size=(10,))
    rbf = RBFModel()
    mu, sigma = rbf(x, t)
    print(mu.shape, sigma.shape)
