"""NICE model
"""

import torch
import torch.nn as nn
from torch.distributions.transforms import Transform,SigmoidTransform,AffineTransform
from torch.distributions import Uniform, TransformedDistribution
import numpy as np


"""Additive coupling layer.
"""
class AdditiveCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an additive coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AdditiveCoupling, self).__init__()
        # Create a binary mask to split inputs into two parts
        self.mask_config = mask_config
        self.mask        = torch.tensor([i % 2 == mask_config for i in range (in_out_dim)])

        # Define a feed-forward network for the transformation t(x1)
        self.network     = self._build_network(in_out_dim // 2, mid_dim, hidden)

    def _build_network(self, input_dim, mid_dim, hidden):
        """Helper to build the MLP network."""
        layers = [nn.Linear(input_dim, mid_dim), nn.ReLU()]
        for _ in range(hidden - 1):
            layers.extend([nn.Linear(mid_dim, mid_dim), nn.ReLU()])
        layers.append(nn.Linear(mid_dim, input_dim)) #Output size matches x2
       
        return nn.Sequential(*layers)

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        # Split inputs into x1 (transformed) and x2 (unchanged)
        x0 = x[:, self.mask]  # Part determined by the mask
        x1 = x[:, ~self.mask] # Complementary part

        if reverse:
            # Reverse transformation: x0 = x0' - t(x1)
            x0 = x0 - self.network(x1)
        else:
            # Forward transformation: x0' = x0 + t(x1)
            x0 = x0 + self.network(x1)

        # Concatenate x0 and x1 back into the full tensor
        x_out = torch.zeros_like(x)
        x_out[:, self.mask]  = x0
        x_out[:, ~self.mask] = x1

        #log_det_J remains unchanged because the Jacobian determinant is 1
        return x_out, log_det_J

class AffineCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an affine coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AffineCoupling, self).__init__()
        # Create a binary mask to split inputs into two parts
        self.mask_config = mask_config
        self.mask        = torch.tensor([i % 2 == mask_config for i in range(in_out_dim)])

        # Define a network for both scale (s) and translation (t)
        self.network     = self._build_network(in_out_dim // 2, mid_dim, hidden)
        
    def _build_network(self, input_dim, mid_dim, hidden):
        """Helper to build the MLP network."""
        layers = [nn.Linear(input_dim, mid_dim), nn.ReLU()]
        for _ in range(hidden - 1):
            layers.extend([nn.Linear(mid_dim, mid_dim), nn.ReLU()])
        layers.append(nn.Linear(mid_dim, input_dim*2)) # Output is scale + shift 
        
        return nn.Sequential(*layers)


    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        # Split input into x0 (transformed) and x1 (unchanged)
        x0 = x[:, self.mask]
        x1 = x[:, ~self.mask]

        # Compute scale and translation
        scale_shift  = self.network(x1)            # Single network generates scale and shift
        scale, shift = scale_shift.chunk(2, dim=1) # Split output to scale and shift
        scale        = torch.sigmoid(scale + 2.0)  # Apply sigmoid activation to ensure positivity

        if reverse:
            # Reverse transformation: x0  = (x0' - t(x1)) / exp(s(x1))
            x0         = (x0 - shift) / scale
            log_det_J  = log_det_J - torch.sum(torch.log(scale))
        else:
            # Forward transformation: x0' = x0' * exp(s(x1)) + t(x1))
            x0         = x0 * scale + shift
            log_det_J  = log_det_J + torch.sum(torch.log(scale))

        # Combine x1 and x2 back into the full tensor
        x_out               = torch.zeros_like(x)
        x_out[:, self.mask]  = x0
        x_out[:, ~self.mask] = x1

        return x_out, log_det_J

"""Log-scaling layer.
"""
class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.

        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)
        self.eps   = 1e-5

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        scale = torch.exp(self.scale)+ self.eps
        
        if reverse:
            # Reverse the transfornation: x = y / exp(s)
            x = x / scale
        else:
            # Transfornation: x = y * exp(s)
            x = x * scale

        log_det_J = torch.sum(torch.log(scale))

        return x, log_det_J

"""Standard logistic distribution.
"""
logistic = TransformedDistribution(Uniform(0, 1), [SigmoidTransform().inv, AffineTransform(loc=0., scale=1.)])

"""Device compatible logistic distribution
"""
def create_logistic_prior(loc=0.0, scale=1.0, device='cpu'):
    """
    Creates a logistic distribution that is device-compatible.

    Args:
        loc: Location parameter (mean of the distribution).
        scale: Scale parameter (variance related to spread).
        device: The device to place tensors on (CPU or GPU).

    Returns:
        A TransformedDistribution object representing the logistic distribution.
    """
    # Make sure base distribution is on the correct device
    base_dist = Uniform(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device)
    )

    # Apply transformations
    transforms = [SigmoidTransform().inv, AffineTransform(loc=loc, scale=scale)]

    # Create the transformed logistic distribution
    logistic = TransformedDistribution(base_dist, transforms)

    return logistic


"""NICE main model.
"""
class NICE(nn.Module):
    def __init__(self, prior, coupling, coupling_type,
        in_out_dim, mid_dim, hidden,device):
        """Initialize a NICE.

        Args:
            coupling_type: 'additive' or 'adaptive'
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            device: run on cpu or gpu
        """
        super(NICE, self).__init__()
        self.device = device
        if prior == 'gaussian':
            self.prior = torch.distributions.Normal(
                torch.tensor(0.).to(device), torch.tensor(1.).to(device))
        elif prior == 'logistic':
            self.prior = create_logistic_prior(device=self.device)
        else:
            raise ValueError('Prior not implemented.')
        
        self.in_out_dim    = in_out_dim
        self.coupling      = coupling
        self.coupling_type = coupling_type

        # Create coupling layers
        self.coupling_layers = nn.ModuleList()
        for i in range(coupling):
            mask_config = i % 2 # Alternate masks (odd/even)
            if coupling_type == "additive":
                self.coupling_layers.append(AdditiveCoupling(in_out_dim, mid_dim, hidden, mask_config))
            elif coupling_type in ["affine", "adaptive"]:
                self.coupling_layers.append(AffineCoupling(in_out_dim, mid_dim, hidden, mask_config))
            else:
                raise ValueError("Invalid coupling type (or hasn't been implemented yet). Use 'additive' or 'affine/adaptive'.")

        # Add the final scaling layer
        self.scaling_layer = Scaling(in_out_dim)

        # Move model to the correct device after layers are defined
        self.to(device) 


    def f_inverse(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        z = z.to(self.device)                           # Ensure input is on the correct device

        z, _ = self.scaling_layer(z, reverse=True)
        for layer in reversed(self.coupling_layers):
            z, _ = layer(z, 0, reverse=True)
        
        return z

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z and log determinant Jacobian
        """
        x         = x.to(self.device)                           # Ensure input is on the correct device
        log_det_J = torch.zeros(x.size(0), device=self.device)  # Initialize Jacobian determinant
        for layer in self.coupling_layers:
            x, log_det_J = layer(x, log_det_J, reverse=False)
        x, log_det_J_scaling = self.scaling_layer(x, reverse=False)
        
        log_det_J += log_det_J_scaling # Adding the scaling layer los 
        
        return x, log_det_J

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        x            = x.to(self.device) # Ensure input is on the correct device
        z, log_det_J = self.f(x)
        
        dequant_adj_term = torch.log(torch.tensor(256.0, device=self.device)) * self.in_out_dim  # Dequantization adjustment
        log_det_J       -= dequant_adj_term                                                      #log det for rescaling from [0.256] (after dequantization) to [0,1]

        log_ll = torch.sum(self.prior.log_prob(z), dim=1) # Prior probability of z
        log_ll = log_ll.to(self.device)

        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.in_out_dim)).to(self.device)
        
        return self.f_inverse(z)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)