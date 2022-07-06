import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

import torch

from sklearn.datasets import make_swiss_roll

def make_dataset(n_samples=300, noise=0.3):
    """
    Returns a normalized swiss roll dataset in a (n_samples, 2) matrix, where
    each rows corresponds to a single sample, and each column to a component (x or y).
    """
    points = make_swiss_roll(n_samples=n_samples, noise=noise)[0][:, [0,2]]
    # Normalize the data so it lies between [-1 and 1]
    #max_value = np.max(points)
    max_value = 10.0
    return points/max_value

def extract(a, t, x_shape):
    """
    a is a tensor of size (timesteps,)
    t is a tensor of size (n_samples,)
    x_shape is (n_samples, *)

    This function returns an (n_samples,) which contains the values of 
    a indexed by t.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def schedule_variances(beta_1=1e-4, beta_T = 0.02, T=1000, mode='linear'):
    """
    Schedule the variances used in the forward diffusion step.

    Parameters:
        - mode: 'linear' performs a simple linear interpolation between beta_1 and beta_T
                'sigmoid' betas have the shape of a sigmoid function over time (https://github.com/acids-ircam/diffusion_models/blob/main/diffusion_03_waveform.ipynb)
                'fixed' performs the schedule used in this paper (footnote of section 2.4.1, https://arxiv.org/abs/1503.03585)
    Returns:
        - betas: Array with all the beta_t
        - alphas: (1 - beta_t)
        - alphas_bar: Just the cumulative product of the alphas
    All will be arrays of shape (T, )
    """
    if mode == 'linear':
        betas = np.linspace(beta_1, beta_T, T)
    elif mode == 'sigmoid':
        betas = np.linspace(-6, 6, T)
        betas = 1/(1 + np.exp(-betas)) * (beta_T - beta_1) + beta_1
    elif mode == 'fixed':
        betas = np.array([1./(T-t+1) for t in range(1, T+1)])
    else:
        raise Exception("Invalid mode!")
    return torch.tensor(betas, dtype=torch.float32)


# TESTING CODE
if __name__ == "__main__":
    x_0 = torch.randn((10, 2))
    a = torch.tensor([0, 1])
    t = torch.randint(0, 2, size=(10,))
    out = extract(a, t, x_0.shape)
    print("x_0: ", x_0.shape)
    print("a: ", a.shape)
    print("t: ", t.shape)
    print("out: ", out.shape) 
    print(out)