## patch solution to run the test using pytest package- importing packages sys and os.
##TODO in the future will try to use a more elegant and generic solution

# for now the test can be ran usign pytest or directly running the script (using main function)

import sys
import os

# Add the parent directory of the test file to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    mask        = torch.tensor([i % 2 == mask_config for i in range (in_out_dim)])
    x_to_transform, x_no_change = x[:, mask], x[:, ~mask]  # Split into even and odd
    x_out_trans, x_out_unchainged = x_transformed[:, mask], x_transformed[:, ~mask]
    assert torch.allclose(x_no_change, x_out_unchainged), "Unchanged part of the input (x1) was altered."

    print("Test passed for Additive Coupling layer!")

# Run the test manually
if __name__ == "__main__":
    test_additive_coupling_forward_and_reverse()