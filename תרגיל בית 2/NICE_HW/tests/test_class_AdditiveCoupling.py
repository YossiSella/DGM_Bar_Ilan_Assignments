import torch
from nice import AdditiveCoupling  # Import the class from your implementation

def test_additive_coupling_forward_and_reverse():
    # Test parameters
    in_out_dim = 4  # Input/output dimensions
    mid_dim = 64    # Hidden layer size
    hidden = 3      # Number of hidden layers
    mask_config = 0  # Mask configuration (even indices)

    # Create Additive Coupling Layer
    coupling_layer = AdditiveCoupling(in_out_dim, mid_dim, hidden, mask_config)

    # Generate random input tensor
    batch_size = 2
    x = torch.randn(batch_size, in_out_dim)  # Random input tensor
    log_det_J = torch.zeros(batch_size)     # Initial Jacobian determinant

    # Forward Transformation
    x_transformed, log_det_J_forward = coupling_layer(x, log_det_J, reverse=False)

    # Reverse Transformation
    x_recovered, log_det_J_reverse = coupling_layer(x_transformed, log_det_J_forward, reverse=True)

    # Assertions
    assert torch.allclose(x, x_recovered, atol=1e-6), "Reverse transformation failed to recover original input."
    assert torch.allclose(log_det_J, log_det_J_forward), "Jacobian determinant changed during forward transformation."
    assert torch.allclose(log_det_J, log_det_J_reverse), "Jacobian determinant changed during reverse transformation."

    # Check that x1 (unchanged part) remains unaltered
    x_even, _ = x[:, ::2], x[:, 1::2]  # Split into even and odd
    x_trans_even, _ = x_transformed[:, ::2], x_transformed[:, 1::2]
    assert torch.allclose(x_even, x_trans_even), "Unchanged part of the input (x1) was altered."

    print("Test passed for Additive Coupling layer!")

# Run the test manually
if __name__ == "__main__":
    test_additive_coupling_forward_and_reverse()