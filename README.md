# Lab 4 – Autoencoders, Variational Autoencoders & GANs

This notebook explores **generative deep learning models** through three core architectures:

- **Autoencoders (AE)**
- **Variational Autoencoders (VAE)**
- **Generative Adversarial Networks (GANs)**

All models are implemented in **PyTorch** and trained on the **MNIST** dataset.

---

## Notebook Overview

### 1. Autoencoder (AE)
- Encoder–decoder neural network
- Compresses MNIST images into a **2D latent space**
- Trained using **Mean Squared Error (MSE)**
- Used to visualize learned representations of handwritten digits

### 2. Variational Autoencoder (VAE)
- Probabilistic extension of the autoencoder
- Learns latent parameters **μ (mean)** and **σ (variance)**
- Loss function combines:
  - Reconstruction loss (MSE)
  - **KL Divergence** regularization
- Enables smooth sampling and structured latent space

### 3. Generative Adversarial Network (GAN)
- Two competing networks:
  - **Generator**: produces synthetic MNIST images
  - **Discriminator**: classifies real vs fake images
- Trained via adversarial loss
- Tracks generator and discriminator losses
- Generates novel digit samples after training

---

## Learning Objectives

- Understand **representation learning** using autoencoders
- Learn how **latent distributions** are enforced in VAEs
- Observe adversarial training dynamics in GANs
- Visualize latent spaces and generated samples
- Compare reconstruction-based and adversarial generative models

---

## 🛠 Requirements

Install dependencies before running the notebook:

```bash
pip install torch torchvision matplotlib numpy
```

Optional (to run notebooks):

```bash
pip install jupyter
```

## How to Run

- Clone the repository
- Start Jupyter Notebook:
Open:
```bash
lab4-ae-and-vae-gans.ipynb
```
- Run all cells in order
- CUDA is used automatically if available.

## Outputs & Visualizations

- MNIST reconstructions (AE & VAE)
- 2D latent space plots
- GAN-generated digit samples
- Training loss curves

## Repository Structure

```bash
.
├── lab4-ae-and-vae-gans.ipynb
└── README.md
```

## Notes

- Latent dimension is intentionally small for visualization
- Models are implemented from scratch for educational clarity
- Intended for learning and experimentation

## 📚 References

- Kingma & Welling — Auto-Encoding Variational Bayes
- Goodfellow et al. — Generative Adversarial Networks
- PyTorch Documentation
