import torch
from nice import Scaling

def test_scaling_layer():
    dim = 4  # Input/output dimensionality
    layer = Scaling(dim)

    # Generate a random input tensor
    x = torch.randn(10, dim)  # Batch size = 10
    log_det_J = torch.zeros(10)  # Initialize log determinant

    # Forward transformation
    x_out, log_det_J_out = layer(x, log_det_J, reverse=False)

    # Reverse transformation
    x_reconstructed, log_det_J_reconstructed = layer(x_out, log_det_J_out, reverse=True)

    # Check invertibility
    assert torch.allclose(x, x_reconstructed, atol=1e-5), "Input and reconstructed input do not match."
    assert torch.allclose(log_det_J, log_det_J_reconstructed, atol=1e-5), "Log determinant mismatch after inversion."

    print("Scaling layer test passed.")

test_scaling_layer()