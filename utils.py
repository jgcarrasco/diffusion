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
    
    hline = np.random.uniform(low=[0., -0.01], high=[1., 0.01], 
                               size=(int(n_samples/2), 2))
    vline = np.random.uniform(low=[0.01, 0.], high=[0.01, 1.], 
                               size=(int(n_samples/2), 2))
    points = np.concatenate([hline, vline])

    return points

def extract(a, t, x_shape):
    """
    a is a tensor of size (timesteps,)
    t is a tensor of size (n_samples,)
    x_shape is (n_samples, *)

    This function returns an (n_samples, *) which contains the values of 
    a indexed by t, broadcasted to the format of x_shape. In other words,
    this function is used to be able to select different betas for separate
    timesteps in a parallel fashion, in order to use batches for training.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def extract2(a, t, x_shape):
    out = torch.gather(a, 0, t.to(a.device))
    reshape = [t.shape[0]] + [1] * (len(x_shape) - 1)
    return out.reshape(*reshape)

def schedule_variances(beta_1=1e-5, beta_T = 1e-2, T=1000, mode='sigmoid'):
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
        betas = torch.linspace(beta_1, beta_T, T)
    elif mode == 'sigmoid':
        betas = torch.linspace(-6, 6, T)
        betas = torch.sigmoid(betas) * (beta_T - beta_1) + beta_1
    elif mode == 'fixed':
        betas = torch.tensor([1./(T-t+1) for t in range(1, T+1)])
    else:
        raise Exception("Invalid mode!")
    return betas

def plot_summary(x_seq, n_snapshots=10):
    """
    Plots a summary of the learned reverse process.

    Parameters
    ----------
    x_seq : `torch.Tensor`
        Tensor of shape (N, T, D), where N is the number of samples, T is 
        the total number of timesteps and D is the input shape or tensor of
        shape (E, N, T, D), where the extra dimension E contains information
        about different stages of training.
    
    Returns
    -------
    `n_snapshots` plots containing samples drawn at different steps of the 
    learned reverse process. If x_seq is of dimension (E, N, T, D), plots the 
    same, but at different stages of training.
    """
    assert len(x_seq.size()) == 3 or len(x_seq.size()) == 4, \
        "x_seq must be of shape (N, T, D) or (E, N, T, D)"
    if len(x_seq.size()) == 3:
        timesteps = x_seq.size()[1] - 1
        x_seq = x_seq.cpu().detach().numpy()

        fig, axes = plt.subplots(ncols=n_snapshots, figsize=(28, 3))
        ts = np.linspace(0, timesteps, n_snapshots, dtype=int)

        for t, ax in zip(ts, axes):
            ax.scatter(x_seq[:, t, 0], x_seq[:, t, 1], alpha=0.5, color='C0', 
                    edgecolor='white', s=40)
            ax.set_title("t = " + str(t))
            ax.set_aspect("equal")
    else:
        epochs = x_seq.size()[0]
        timesteps = x_seq.size()[2] - 1
        x_seq = x_seq.cpu().detach().numpy()

        fig, axes = plt.subplots(nrows=epochs, ncols=n_snapshots, 
                                figsize=(n_snapshots*1.5, epochs*1.5))
        ts = np.linspace(0, timesteps, n_snapshots, dtype=int)
        for epoch in range(epochs):
            for t, ax in zip(ts, axes[epoch]):
                ax.scatter(x_seq[epoch, :, t, 0], x_seq[epoch, :, t, 1], 
                           alpha=0.5, color='C0', edgecolor='white', s=40)
                ax.set_title("t = " + str(t))
                ax.set_aspect("equal")
    fig.tight_layout()
    return fig

def plot_trajectory(x, y):
    u = np.diff(x)
    v = np.diff(y)
    pos_x = x[:-1] + u/2
    pos_y = y[:-1] + v/2
    norm = np.sqrt(u**2+v**2) 

    fig, ax = plt.subplots(figsize=(1.6*8, 8))
    ax.scatter(x[0], y[0], marker="o", color='red', s=100, zorder=3, label="Start")
    ax.scatter(x[-1], y[-1], marker="o", color='yellow', s=100, zorder=3, label="End")
    ax.plot(x,y, marker="o", zorder=1)
    ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=2, pivot="mid")
    ax.legend()
    return ax

# TESTING CODE
if __name__ == "__main__":
    print("FIRST TEST")
    print("")
    x_0 = torch.randn((10, 2))
    a = torch.tensor([0, 1])
    t = torch.randint(0, 2, size=(10,))
    out = extract(a, t, x_0.shape)
    out2 = extract2(a, t, x_0.shape)
    print("x_0: ", x_0.shape)
    print("a: ", a.shape)
    print("t: ", t.shape)
    print("out: ", out.shape)
    print("out2: ", out2.shape)
    print("SECOND TEST")
    print("")
    x_0 = torch.randn((10, 2))
    a = torch.tensor([0, 42])
    t = torch.tensor([1])
    out = extract(a, t, x_0.shape)
    out2 = extract2(a, t, x_0.shape)
    print("x_0: ", x_0.shape)
    print("a: ", a.shape)
    print("t: ", t.shape)
    print("out: ", out.shape) 
    print("out2: ", out2.shape)
    print(out)