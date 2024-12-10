## patch solution to run the test using pytest package- importing packages sys and os.
##TODO in the future will try to use a more elegant and generic solution

# for now the test can be ran usign pytest or directly running the script (using main function)

import sys
import os

# Add the parent directory of the test file to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
from nice import AffineCoupling  # Replace with the correct import path


@pytest.fixture
def setup_affine_coupling():
    """Fixture to create an AffineCoupling layer for testing."""
    in_out_dim = 4  # Input/output dimension
    mid_dim = 8  # Number of units in hidden layers
    hidden = 2  # Number of hidden layers
    mask_config = 0  # Mask configuration

    # Initialize AffineCoupling layer
    return AffineCoupling(in_out_dim=in_out_dim, mid_dim=mid_dim, hidden=hidden, mask_config=mask_config)


def test_forward_reverse_invertibility(setup_affine_coupling):
    """Test if forward and reverse transformations are invertible."""
    layer = setup_affine_coupling

    # Create a random input tensor
    x = torch.randn(10, layer.mask.size(0))  # Batch size = 10, feature size = in_out_dim
    log_det_J = torch.zeros(10)  # Initialize log_det_J

    # Forward transformation
    x_out, log_det_J_out = layer(x, log_det_J, reverse=False)

    # Reverse transformation
    x_reconstructed, log_det_J_reconstructed = layer(x_out, log_det_J_out, reverse=True)

    # Assert that the reconstructed input matches the original input
    assert torch.allclose(x, x_reconstructed, atol=1e-5), "Reconstructed input does not match the original input."

    # Assert that the Jacobian determinant remains consistent
    assert torch.allclose(log_det_J, log_det_J_reconstructed, atol=1e-5), \
        "Jacobian determinant does not match after forward and reverse transformations."

def test_jacobian_update(setup_affine_coupling):
    """Test if log_det_J is updated correctly during forward transformation."""
    layer = setup_affine_coupling
    x = torch.randn(10, layer.mask.size(0)) * 10  # Input tensor with variability
    log_det_J = torch.zeros(10, requires_grad=True)  # Ensure gradient tracking

    # Forward transformation
    _, log_det_J_out = layer(x, log_det_J, reverse=False)

    # Debug: Print values for log_det_J and scale
    x1 = x[:, layer.mask]
    y = layer.network(x1)
    dim = y.shape[1] // 2
    scale = torch.sigmoid(y[:, :dim] + 2.0)
    print("Scale values:", scale)
    print("log_det_J before:", log_det_J)
    print("log_det_J after:", log_det_J_out)
    print("Log(scale):", torch.log(scale).sum(dim=1))

    # Ensure log_det_J is updated
    assert not torch.allclose(log_det_J, log_det_J_out), "log_det_J was not updated during forward transformation."

def test_scale_positivity(setup_affine_coupling):
    """Test if the scale values are positive."""
    layer = setup_affine_coupling
    x = torch.randn(10, layer.mask.size(0))  # Input tensor

    # Extract the scale values directly from the network
    x1 = x[:, layer.mask]
    y = layer.network(x1)
    dim = y.shape[1] // 2
    scale = y[:, :dim]
    scale = torch.sigmoid(scale + 2.0)  # Apply the same activation as in the forward method

    # Assert that all scale values are positive
    assert torch.all(scale > 0), "Scale values are not positive."
