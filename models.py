import torch
import torch.nn as nn

from torch_rbf import *
from utils import *

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
        x = self.lin1(x, y)
        x = self.lin2(x, y)
        return self.lin3(x)

class SwissRollModel(nn.Module):
    """
    Simple model used to approximate the reverse diffusion process on
    a the swiss roll distribution. It consists on a first layer with 16 hidden 
    units which is shared across all timesteps, and a different second layer
    with two unit vectors as output. 
    The reason for having a separate second layer for each timestep 
    is to include the temporal dependency. 
    """
    def __init__(self, input_size=2, timesteps=100):
        super(SwissRollModel, self).__init__()

        self.first_layer = RBF(input_size, 16, basis_func_dict()['gaussian'])
        self.embedding = nn.Embedding(timesteps, 16, sparse=False)
        self.output_layer = RBF(16, input_size, basis_func_dict()['gaussian'])

        self.embedding.weight.data.uniform_()

    def forward(self, x, t):
        x = self.first_layer(x) # (batch_size, 16)
        cond = self.embedding(t) # (batch_size, 16)
        out = self.output_layer(cond * x) # (batch_size, input_size)
        return out

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
        self.sqrt_alphas = torch.sqrt(self.alphas)
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
        """
        Get x_t given x_0 and a timestep t. This is equivalent to applying
        the forward diffusion distribution q, t times, i.e. q(x_t | x_0)
        """
        z = torch.randn_like(x_0)
        return extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 +\
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * z

    def sample_p(self, x_t, t):
        """
        x_{t-1} = (x_t - beta_t/sqrt(1-alpha_cumprod_t)*z_pred) + sigma*z
        """
        #z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        z = torch.randn_like(x_t)
        z_pred = self.model(x_t, t)
        a = 1./extract(self.sqrt_alphas, t, x_t.shape)
        b = extract(self.betas, t, x_t.shape) / \
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sigma = extract(self.betas, t, x_t.shape).sqrt()
        x_tm1 = a*(x_t - b*z_pred) + sigma*z
        return x_tm1 

    def generate(self, n_samples=100):   
        x_t = torch.randn((n_samples, 2))
        for t in reversed(range(self.timesteps)):
            t = torch.tensor([t])
            x_t = self.sample_p(x_t, t)
        return x_t 

    def compute_loss(self, x_0, t):
        z = torch.randn_like(x_0)
        x_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 +\
              extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * z
        z_pred = self.model(x_t, t)
        # Compute the loss
        return (z - z_pred).square().mean()

    def plot_reverse_process(self, n_samples=1000, n_snapshots=10):
        
        fig, axes = plt.subplots(ncols=n_snapshots, figsize=(3*n_snapshots, 3))

        # list of timesteps where a plot will be shown
        t_snapshots = torch.linspace(0, self.timesteps, n_snapshots, 
                                        dtype=torch.int64)[:-1]

        x_t = torch.randn((n_samples, 2))
        # Plot the first sample
        i = n_snapshots - 1
        x_t_cpu = x_t.detach().numpy()
        axes[i].scatter(x_t_cpu[:, 0], x_t_cpu[:, 1])
        axes[i].set_title("t = " + str(self.timesteps))
        i -= 1
        for t in reversed(range(self.timesteps)):
            t = torch.tensor([t])
            x_t = self.sample_p(x_t, t)
            if torch.isin(t[0], t_snapshots):
                # Plot the sample
                x_t_cpu = x_t.detach().numpy()
                axes[i].scatter(x_t_cpu[:, 0], x_t_cpu[:, 1])
                axes[i].set_title("t = " + str(t[0].item()+1))
                i -= 1
        return fig   

# Testing code
if __name__ == "__main__":
    simple_model = SwissRollModel()
    input_test = torch.rand((2, 2))
    t = torch.tensor([10, 20])
    print("Input shape: ", input_test.size())
    print("Output shape at t = 10", simple_model(input_test, t).size())
