"""NICE model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, latent_dim, device):
        """Initialize a VAE.

        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        super(Model, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1, 2),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )

        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.upsample = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2),  # B, 1, 28, 28
            nn.Sigmoid()
        )


    def sample(self, sample_size, mu=None, logvar=None):
        """
        Generate samples from the latent space.

        Args:
            sample_size (int): Number of samples to generate.
            mu (torch.Tensor): Mean of the latent space distribution (default: None, uses prior).
            logvar (torch.Tensor): Log variance of the latent space distribution (default: None, uses prior).

        Returns:
            torch.Tensor: Generated samples of shape (sample_size, latent_dim).
        """
        if mu==None:
            mu = torch.zeros((sample_size,self.latent_dim)).to(self.device)
        if logvar == None:
            logvar = torch.zeros((sample_size,self.latent_dim)).to(self.device)
        
        z = self.z_sample(mu, logvar)
        return z


    def z_sample(self, mu, logvar):
        """
        Sample z using the reparameterization trick.

        Args:
            mu (torch.Tensor): Mean of the posterior distribution.
            logvar (torch.Tensor): Log variance of the posterior distribution.

        Returns:
            torch.Tensor: Sampled latent variables.
        """
        # Reparametrization trick
        std = torch.exp(0.5 * logvar) #standard deviation
        eps = torch.randn_like(std)   #random noise ~ N(0,I) 
        return mu + std * eps

    def loss(self, x, recon, mu, logvar):
        """
        Compute the ELBO loss.

        Args:
            x (torch.Tensor): Original input images.
            recon (torch.Tensor): Reconstructed images.
            mu (torch.Tensor): Mean of the posterior distribution.
            logvar (torch.Tensor): Log variance of the posterior distribution.

        Returns:
            torch.Tensor: Total ELBO loss.
        """
        #Binary cross-entropy reconstruction loss
        recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')

        #KL divegence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_div

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input images.

        Returns:
            tuple: Reconstructed images, mu, logvar
        """
        batch_size = x.size(0)
        
        # Encode
        x      = self.encoder(x).view(batch_size, -1)
        mu     = self.mu(x)
        logvar = self.logvar(x)

        # Sample latent variables
        z = self.sample(mu, logvar)

        # Decode
        z     = self.upsample(z).view(batch_size, 64, 7, 7)
        recon = self.decoder(z)

        return recon, mu, logvar