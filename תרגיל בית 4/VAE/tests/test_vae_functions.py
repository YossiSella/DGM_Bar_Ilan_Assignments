## patch solution to run the test using pytest package- importing packages sys and os.
##TODO in the future will try to use a more elegant and generic solution

# for now the test can be ran usign pytest or directly running the script (using main function)

import sys
import os

# Add the parent directory of the test file to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from VAE import Model

def test_sample():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    latent_dim = 10
    model = Model(latent_dim, device)

    sample_size = 5
    samples = model.sample(sample_size)

    # Check shape
    assert samples.shape == (sample_size, latent_dim), f"Expected shape ({sample_size}, {latent_dim}), got {samples.shape}"

    # Check device
    assert samples.device == device, f"Expected device {device}, got {samples.device}"

def test_z_sample():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    latent_dim = 10
    model = Model(latent_dim, device)

    batch_size = 5
    mu = torch.zeros((batch_size, latent_dim)).to(device)
    logvar = torch.zeros((batch_size, latent_dim)).to(device)

    z = model.z_sample(mu, logvar)

    # Check shape
    assert z.shape == (batch_size, latent_dim), f"Expected shape ({batch_size}, {latent_dim}), got {z.shape}"

    # Check device
    assert z.device == device, f"Expected device {device}, got {z.device}"

def test_loss():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    latent_dim = 10
    model = Model(latent_dim, device)

    batch_size = 5
    x = torch.rand((batch_size, 1, 28, 28)).to(device)
    recon = torch.rand((batch_size, 1, 28, 28)).to(device)
    mu = torch.zeros((batch_size, latent_dim)).to(device)
    logvar = torch.zeros((batch_size, latent_dim)).to(device)

    loss = model.loss(x, recon, mu, logvar)

    # Check loss is scalar
    assert isinstance(loss.item(), float), f"Loss should be a scalar, got {type(loss)}"

def test_forward():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    latent_dim = 10
    model = Model(latent_dim, device)

    batch_size = 5
    x = torch.rand((batch_size, 1, 28, 28)).to(device)
    
    recon, mu, logvar = model(x)

    # Check shapes
    assert recon.shape == x.shape, f"Expected shape {x.shape}, got {recon.shape}"
    assert mu.shape == (batch_size, latent_dim), f"Expected shape ({batch_size}, {latent_dim}), got {mu.shape}"
    assert logvar.shape == (batch_size, latent_dim), f"Expected shape ({batch_size}, {latent_dim}), got {logvar.shape}"
    
    # Check devices
    assert recon.device == device, f"Expected device {device}, got {recon.device}"
    assert mu.device == device, f"Expected device {device}, got {mu.device}"
    assert logvar.device == device, f"Expected device {device}, got {logvar.device}"