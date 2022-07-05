import os

import torch
import torch.nn as nn

from utils import *
from models import *


# Hyperparameters and constants
n_epochs = 1000

T = 40
# Precompute the variance schedule
betas, alphas, alphas_bar = schedule_variances(beta_1=1e-4, beta_T=0.25, T=T)
betas, alphas, alphas_bar = torch.tensor(betas), torch.tensor(alphas), torch.tensor(alphas_bar)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Loading the data
X = torch.tensor(make_dataset(n_samples=300), dtype=torch.float32).to(device)
betas.to(device), alphas.to(device), alphas_bar.to(device)


# Loading the model and optimizer
model = SwissRollModel(input_size=X.size()[-1], T=T).to(device)
optimizer = torch.optim.Adam(model.parameters())
criteria = nn.MSELoss().to(device)

for i in range(n_epochs):
    for x_0 in X:
        # Sample a random timestep
        t = np.random.randint(0, T)
        # Sample noise from a gaussian
        z = torch.randn(x_0.size(), dtype=torch.float32).to(device)
        # Compute x_t
        x_t = torch.sqrt(alphas_bar[t]) * X + torch.sqrt(1 - alphas_bar[t]) * z
        # Predict the noise for x_t at timestep t
        z_pred = model(x_t, t)
        # Compute the loss and backpropagate
        loss = criteria(z, z_pred)
        optimizer.zero_grad()
        optimizer.step()
        print("Epoch", i, "Loss", loss.item())

# Save the model for inference
torch.save(model.state_dict(), os.path.join("saved_models", "test.pth"))