# ddim
Implementation and experiments with Denoising Diffusion Implicit Models (DDIM)

## TO DO
- [ ] Implementing DDIM (faster inference)

## Background

### DDPM

The idea behind Denoising Diffusion Probabilistic Models (DDPM) is to learn to generate images from noise by iteratively denoising the image.

Given a data distribution $x_o \sim q(x_0)$, we define a **forward process** which produces $x_1,...,x_T$ by adding Gaussian noise at each timestep via the distribution $q(x_t|x_{t-1})$. Given sufficiently large $T$, the last latent $x_T$ is almost an isotropic Gaussian distribution.

Therefore, if we had the exact reverse distribution, $q(x_{t-1}|x_t)$ we could be able to sample $x_T \sim \mathcal{N}(0, \mathbb{I})$ and run the process in reverse to get a sample from the original distribution. However, this reverse process depends on the original distribution, to it is untractable, so we approximate it using a neural network.

#### DDIM

