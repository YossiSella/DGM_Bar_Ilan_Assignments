## patch solution to run the test using pytest package- importing packages sys and os.
##TODO in the future will try to use a more elegant and generic solution

# for now the test can be ran usign pytest or directly running the script (using main function)

import sys
import os

# Add the parent directory of the test file to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
from nice import NICE

@pytest.fixture
def nice_model():
    in_out_dim = 16  # Example dimensionality
    mid_dim = 8
    hidden = 3
    coupling = 4
    prior = 'logistic'
    coupling_type = 'additive'
    
    if torch.cuda.is_available():
        print(f"Available GPU: {torch.cuda.get_device_name(0)}")
        
    else:
        print("No GPU available.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Initializing NICE model with dimensions:", in_out_dim)
    return NICE(prior, coupling, coupling_type, in_out_dim, mid_dim, hidden, device)

def test_initialization(nice_model):
    print("Testing initialization...")
    assert nice_model is not None
    assert isinstance(nice_model, NICE)
    print("Initialization test passed.")

def test_forward_inverse(nice_model):
    print("Testing forward and inverse transformations...")
    x = torch.rand(10, nice_model.in_out_dim, device=nice_model.device)  # Random input batch
    print("Input tensor:", x)

    # Forward transformation
    z, log_det_J = nice_model.f(x)
    print("Latent space tensor:", z)
    print("Log determinant of Jacobian:", log_det_J)

    # Reverse transformation
    x_reconstructed = nice_model.f_inverse(z)
    print("Reconstructed tensor:", x_reconstructed)

    # Check reconstruction accuracy
    assert torch.allclose(x, x_reconstructed, atol=1e-5), "Reconstructed x does not match original x"
    print("Forward and inverse transformations test passed.")

def test_log_prob(nice_model):
    print("Testing log probability computation...")
    x = torch.rand(10, nice_model.in_out_dim, device=nice_model.device)  # Random input batch
    print("Input tensor:", x)

    # Compute log probability
    log_prob = nice_model.log_prob(x)
    print("Log probabilities:", log_prob)

    assert log_prob.shape == (10,), "Log-probability shape is incorrect"
    assert torch.isfinite(log_prob).all(), "Log-probability contains NaN or Inf"
    print("Log probability computation test passed.")

def test_sampling(nice_model):
    print("Testing sampling...")
    sample_size = 5

    # Generate samples
    samples = nice_model.sample(sample_size)
    print("Generated samples:", samples)

    assert samples.shape == (sample_size, nice_model.in_out_dim), "Sample shape is incorrect"
    assert torch.isfinite(samples).all(), "Samples contain NaN or Inf"
    print("Sampling test passed.")

def test_scaling_layer():
    print("Testing scaling layer...")
    from nice import Scaling

    dim = 16
    scaling_layer = Scaling(dim)
    print("Scaling layer initialized with dimension:", dim)

    x = torch.rand(10, dim)
    print("Input tensor:", x)

    # Forward transformation
    y, log_det_j = scaling_layer(x)
    print("Transformed tensor:", y)
    print("Log determinant of Jacobian:", log_det_j)

    # Reverse transformation
    x_reconstructed, _ = scaling_layer(y, reverse=True)
    print("Reconstructed tensor:", x_reconstructed)

    assert torch.allclose(x, x_reconstructed, atol=1e-5), "Scaling layer reconstruction failed"
    assert torch.isfinite(log_det_j).all(), "Log determinant contains NaN or Inf"
    print("Scaling layer test passed.")
