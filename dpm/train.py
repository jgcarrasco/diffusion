import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from utils import *
from models import *

# Parameters
timesteps = 40
n_epochs = 500
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the data
x_0 = make_dataset().to(device)
# Instantiate the model
diffusion = Diffusion(timesteps=timesteps).to(device)
opt = torch.optim.Adam(diffusion.parameters(), lr=1e-3)

L = []

for epoch in tqdm(range(n_epochs)):
    n_samples = x_0.shape[0]
    t = torch.randint(0, timesteps-1, size=(n_samples,)).to(device)
    
    loss = diffusion.compute_loss(x_0, t)
    opt.zero_grad()
    loss.backward()
    opt.step()

    L.append(loss.item())

plt.plot(L)
plt.show()

gen = diffusion.generate(n_samples=1000)
gen = gen.cpu().detach().numpy()
plt.scatter(gen[:, 0], gen[:, 1])
plt.show()

os.makedirs("saved_models", exist_ok=True)
torch.save(diffusion, os.path.join("saved_models", "test.pth"))