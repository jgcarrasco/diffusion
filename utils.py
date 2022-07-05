import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

from sklearn.datasets import make_swiss_roll

def make_dataset(n_samples=300):
    """
    Returns a normalized swiss roll dataset in a (n_samples, 2) matrix, where
    each rows corresponds to a single sample, and each column to a component (x or y).
    """
    points = make_swiss_roll(n_samples=n_samples)[0][:, [0,2]]
    # Normalize the data so it lies between [-1 and 1]
    max_value = np.max(points)
    points[:, 0], points[:, 1] = points[:, 0]/max_value, points[:, 1]/max_value
    return points

def schedule_variances(beta_1=1e-4, beta_T = 0.02, T=1000):
    """
    Schedule the variances used in the forward diffusion step.
    In this case, it will simply perform a linear interpolation between
    beta_1 and beta_T.
    Returns:
        - betas: Array with all the beta_t
        - alphas: (1 - beta_t)
        - alphas_bar: Just the cumulative product of the alphas
    All will be arrays of shape (T, )
    """
    ts = np.arange(T)
    betas = np.interp(ts, [ts[0], ts[-1]], [beta_1, beta_T])
    alphas = (1 - betas)
    alphas_bar = np.cumprod(alphas)
    return betas, alphas, alphas_bar