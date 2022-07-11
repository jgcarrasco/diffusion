import os
from tqdm import tqdm
from numpy import diff

import torch
import torch.nn as nn

from utils import *
from models import *

# Parameters
timesteps = 100
n_epochs = 2000
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the data
X_0 = torch.tensor(make_dataset(n_samples=1000), dtype=torch.float32).to(device)
# Instantiate diffusion model
#model = SwissRollModel(timesteps=timesteps)
model = ConditionalModel(n_steps=timesteps)
diffusion = Diffusion(model, timesteps=100, beta_1=1e-4, beta_T=1e-1).to(device)
opt = torch.optim.Adam(diffusion.parameters(), lr=1e-3)

L = []

for epoch in tqdm(range(n_epochs)):
    # Sample timesteps from a uniform distribution
    ts = torch.randint(0, diffusion.timesteps, (X_0.shape[0],)).to(device)
    loss = diffusion.compute_loss(X_0, ts)
    opt.zero_grad()
    loss.backward()
    # Perform gradient clipping
    torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.)
    opt.step()
    L.append(loss.item())
    
plt.plot(L)
plt.show()

# Save the model for inference
torch.save(diffusion.state_dict(), os.path.join("saved_models", "test.pth"))