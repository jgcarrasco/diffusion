import matplotlib.pyplot as plt

from sklearn.datasets import make_swiss_roll

import torch

def make_dataset(n_samples=1000, noise=0.3):
    """
    Recreates the swiss roll dataset used in the first diffusion paper
    (https://arxiv.org/abs/1503.03585)

    Parameters
    ----------
    n_samples : int
        number of samples that will contain the dataset
    noise : float
        amount of noise applied to the dataset
    
    Returns
    -------
        `torch.Tensor` of shape (n_samples, 2) where each rows corresponds to
        a single points, and each column corresponds to a coordinate (x or y)  
    """
    X = make_swiss_roll(n_samples=n_samples, noise=noise)[0][:, [0, 2]]
    max_value = 5.0
    return torch.tensor(X/max_value)

def schedule_variances(timesteps=40):
    betas = torch.arange(1, timesteps+1, dtype=torch.float32)
    betas = 1./(timesteps - betas + 1)
    return betas


if __name__ == "__main__":
    betas = schedule_variances()
    plt.plot(betas)
    plt.show()