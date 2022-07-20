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
batch_size = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the data
X_0 = torch.tensor(make_dataset(n_samples=1000), dtype=torch.float32).to(device)
# Instantiate diffusion model
#model = SwissRollModel(timesteps=timesteps)
model = ConditionalModel(n_steps=timesteps)
diffusion = Diffusion(model, timesteps=100, beta_1=1e-5, beta_T=1e-2).to(device)
opt = torch.optim.Adam(diffusion.parameters(), lr=1e-3)
# Create EMA model
ema = EMA(0.9)
ema.register(diffusion)

L = []
X_seq = []

for epoch in tqdm(range(n_epochs)):
    idxs = torch.randperm(X_0.size()[0])
    for i in range(0, X_0.size()[0], batch_size):
        idx = idxs[i:i+batch_size]
        batch_x_0 = X_0[idx]
        # Sample timesteps from a uniform distribution
        ts = torch.randint(0, diffusion.timesteps, 
                           (batch_x_0.size()[0] // 2 + 1, )).to(device)
        ts = torch.cat([ts, timesteps - ts - 1], dim=0)[:batch_x_0.size()[0]].long()
        loss = diffusion.compute_loss(batch_x_0, ts)
        opt.zero_grad()
        loss.backward()
        # Perform gradient clipping
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.)
        opt.step()
        # Update the exponential moving average
        ema.update(diffusion)
    # Print loss
    if (epoch % 200 == 0):
        print(loss)
        x_seq = diffusion.generate(n_samples=1000, return_all=True)
        X_seq.append(torch.unsqueeze(x_seq, dim=0))
        # fig = plot_summary(x_seq)
        # plt.show()
    L.append(loss.item())

X_seq = torch.cat(X_seq)
fig = plot_summary(X_seq)
plt.show()

# Save the model for inference
ema.ema(diffusion)
torch.save(diffusion, os.path.join("saved_models", "test.pth"))