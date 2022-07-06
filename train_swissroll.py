import os

import torch
import torch.nn as nn

from utils import *
from models import *


# Hyperparameters and constants
n_epochs = 10000

T = 100
# Precompute the variance schedule and values derived from it
betas = schedule_variances(T=T, mode='sigmoid')
# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

device = "cuda" if torch.cuda.is_available() else "cpu"

def forward_q(x_0, t, noise=None):
    if noise is None:
        noise = torch.rand(x_0.shape).to(t.device)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_0.shape)

    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

# Loading the data
X = torch.tensor(make_dataset(n_samples=300), dtype=torch.float32)

# Loading the model and optimizer
#model = SwissRollModel(input_size=X.size()[-1], T=T).to(device)
model = ConditionalModel(T).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criteria = nn.MSELoss().to(device)

L = []

for i in range(n_epochs):
    X = torch.tensor(make_dataset(n_samples=300), dtype=torch.float32).to(device)
    # Sample a random timestep for each sample
    t = torch.randint(0, T, size=(X.shape[0],), dtype=torch.int64).to(device)
    # Sample noise from a gaussian
    z = torch.randn(X.size(), dtype=torch.float32).to(device)
    # Compute x_t
    x_t = forward_q(X, t)
    # Predict the noise for x_t at timestep t
    z_pred = model(x_t, t)
    # Compute the loss and backpropagate
    loss = criteria(z, z_pred)
    optimizer.zero_grad()
    loss.backward()
    # Perform gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    print("Epoch", i, "Loss", loss.item())
    L.append(loss.item())

plt.plot(L)
plt.show()

# Save the model for inference
torch.save(model.state_dict(), os.path.join("saved_models", "test.pth"))