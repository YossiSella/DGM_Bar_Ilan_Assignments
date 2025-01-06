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
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()  # Get the current device index
        device_name = torch.cuda.get_device_name(device_index)
        print(f"CUDA Device Name: {device_name}, CUDA Device index: {device_index}")
    else:
        print("CUDA is not available on this system.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    latent_dim = 10
    model = Model(latent_dim, device)

    sample_size = 5
    samples = model.sample(sample_size)

    # Check shape
    assert samples.shape == (sample_size, latent_dim), f"Expected shape ({sample_size}, {latent_dim}), got {samples.shape}"

    # Check device
    assert samples.device == device, f"Expected device {device}, got {samples.device}"
