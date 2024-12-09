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
            layers = [nn.linear(input_dim, mid_dim), nn.ReLU()]
            for _ in range(hidden - 1):
                layers.extend([nn.linear(mid_dim, mid_dim), nn.ReLU()])
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
        # Split inputs into x1 (unchanged) and x2 (tansformed)
        x1 = x[:, self.mask]  # Part determined by the mask
        x2 = x[:, ~self.mask] # Complementary part

        if reverse:
            # Reverse transformation: x2 = x2' - t(x1)
            x2 = x2 - self.network(x1)
        else:
            # Forward transformation: x2' = x2 + t(x1)
            x2 = x2 + self.network(x1)

        # Concatenate x1 and x2 back into the full tensor
        x_out = torch.zeros_like(x)
        x_out[:, self.mask]  = x1
        x_out[:, ~self.mask] = x2

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
            layers = [nn.linear(input_dim, mid_dim), nn.ReLU()]
            for _ in range(hidden - 1):
                layers.extend([nn.linear(mid_dim, mid_dim), nn.ReLU()])
            layers.append(nn.linear(mid_dim, input_dim*2)) # Output is scale + shift 
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
        # Split input into x1 (unchanged) and x2 (transformed)
        x1 = x[:, self.mask]
        x2 = x[:, ~self.mask]

        # Compute scale and translation
        scale    = self.scale_net(x1)
        tanslate = self.translate_net(x1)



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
        self.eps = 1e-5

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        scale = torch.exp(self.scale)+ self.eps
        #TODO fill in

"""Standard logistic distribution.
"""
logistic = TransformedDistribution(Uniform(0, 1), [SigmoidTransform().inv, AffineTransform(loc=0., scale=1.)])

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
            self.prior = logistic
        else:
            raise ValueError('Prior not implemented.')
        self.in_out_dim = in_out_dim
        self.coupling = coupling
        self.coupling_type = coupling_type

        #TODO fill in

    def f_inverse(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        #TODO fill in

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z and log determinant Jacobian
        """
        #TODO fill in

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)
        log_det_J -= np.log(256)*self.in_out_dim #log det for rescaling from [0.256] (after dequantization) to [0,1]
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.in_out_dim)).to(self.device)
        #TODO

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)