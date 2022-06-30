# Diffusion Probabilistic Models
Implementation and experiments with Diffusion Probabilistic Models

## TO DO
- [ ] Implementing original DPM (http://arxiv.org/abs/1503.03585)
  

## Background

### DPM

![DPM](imgs/DPM.PNG)

### DDPM

The idea behind Denoising Diffusion Probabilistic Models (DDPM) is to learn to generate images from noise by iteratively denoising the image.

Given a data distribution $x_o \sim q(x_0)$, we define a **forward process** which produces $x_1,...,x_T$ by adding Gaussian noise at each timestep via the distribution $q(x_t|x_{t-1})$ (which is a Markovian process). Given sufficiently large $T$, the last latent $x_T$ is almost an isotropic Gaussian distribution.

Therefore, if we had the exact reverse distribution, $q(x_{t-1}|x_t)$ we could be able to sample $x_T \sim \mathcal{N}(0, \mathbb{I})$ and run the process in reverse to get a sample from the original distribution. However, this reverse process depends on the original distribution, to it is untractable, so we approximate it using a neural network.

![Representation of the forward and generation processes in a DDPM](/imgs/ddpm.PNG)

#### DDIM

As $T$ has to be large and the generative process also consists on performing $T steps, it results in a quite considerably slow sampling (20 hours to sample 50k 32x32 images on a Nvidia 2080 Ti).

This is the motivation behind Denoised Diffusion Implicit Models (DDIM). They proposed an alternative non-Markovian noissing process that has the same objective function as DDPM, but allows producing different reverse samplers by changing the variance of the reverse noise. By setting this noise to 0, they provide a way to turn any model $\epsilon_\theta(x_t, t)$ into a deterministic mapping from latents to images, and find that this provides an alternative way to sample with fewer steps. 