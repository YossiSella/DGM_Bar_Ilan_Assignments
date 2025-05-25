"""DDPM model for MNIST
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from UNet import Unet
from tqdm import trange

class DDPM(nn.Module):
    def __init__(self, timesteps=1000, guidance=False,device='cpu'):
        """
        DDPM model for conditional MNIST generation
        :param timesteps: Number of timesteps in the generation
        :param guidance: Run with or without classifier free guidance.
        :param device: cpu or gpu
        """
        super(DDPM, self).__init__()
        self.device = device
        self.guidance = guidance
        self.timesteps = timesteps
        self.in_channels = 1
        self.image_size = 28
        self.model=Unet(self.timesteps,64)
        self.betas = self._cosine_variance_schedule(self.timesteps).to(device)
        self.alphas = 1-self.betas
        self.alpha_bars = torch.cumprod(self.alphas,dim=-1)
        self.loss = nn.MSELoss()

    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas


    def sample(self):
        '''
        Generates 100 MNIST samples conditioned on digits [0-9]*10.

        :return: samples 100 images conditioned on y =torch.tensor([0,1,2,3,4,5,6,7,8,9]*10).to(self.device)
        '''
        y = torch.tensor([0,1,2,3,4,5,6,7,8,9]*10).to(self.device)
        x = torch.randn(100,1,28,28).to(self.device)
        
        for t in trange(self.timesteps-1,-1,-1, desc='Sampling', leave=True):
            t_batch = torch.full((x.size(0),), t, device= self.device, dtype= torch.long)

            # Predict noise at timestep t
            epsilon_pred = self.model(x, t_batch, y)

            # Compute the coefficients for updating x_t
            alpha     = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta      = self.betas[t]

            # Update step for DDPM reverse diffusion
            x = (1 / torch.sqrt(alpha)) * (x - ((beta / torch.sqrt(1 - alpha_bar)) * epsilon_pred))

            # Add noise for all timesteps except the last (at t=0 the image is clean)
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x    += sigma * noise

        return x.to('cpu')
    

    def ddim_sample(self, num_steps=50, eta=0.0):
        '''
        Generates 100 MNIST samples conditioned on digits [0-9]*10 using DDIM sampling (based on the DDIM paper).

        :param num_steps: Number of steps for DDIM sampling.
        :param eta:       Parameter controlling the amount of noise added.
        
        :return: samples 100 images conditioned on y =torch.tensor([0,1,2,3,4,5,6,7,8,9]*10).to(self.device)
        '''
        y = torch.tensor([0,1,2,3,4,5,6,7,8,9]*10).to(self.device)
        x = torch.randn(100,1,28,28).to(self.device)

        # choose `num_steps` timesteps spaced across the full range
        ddim_steps = torch.linspace(0, self.timesteps - 1, steps=num_steps)
        ddim_steps = torch.round(ddim_steps).long().clamp(0, self.timesteps - 1).flip(0).to(self.device)


        for i in trange(len(ddim_steps) - 1, desc=f"DDIM Sampling ({num_steps} steps)", leave=True):
            t = int(ddim_steps[i].item())
            # Get the previous timestep, or 0 if this is the last step
            t_prev = int(ddim_steps[i + 1].item()) if i < len(ddim_steps) - 1 else 0 

            t_batch = torch.full((x.size(0),), t, device=self.device, dtype=torch.long)

            # Predict noise at timestep t   
            epsilon_pred = self.model(x,t_batch,y)

            # Compute the coefficients for updating x_t
            alpha_bar_t = self.alpha_bars[t]
            alpha_bar_prev = self.alpha_bars[t_prev]

            # Estimate x_0 from x_t and predicted noise
            x_0 = (x - torch.sqrt(1 - alpha_bar_t) * epsilon_pred) / torch.sqrt(alpha_bar_t)
            x_0 = x_0.clamp(-1., 1.)

            # Compute the direction to x_t-1
            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
            noise = torch.randn_like(x_0) if t > 0 else 0.

            eps_coeff = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma**2, min=1e-6))

            x = torch.sqrt(alpha_bar_prev) * x_0 + eps_coeff * epsilon_pred + sigma * noise

            # print("ε_pred std:", epsilon_pred.std().item(), "x std:", x.std().item())


        return x.to('cpu')

    def forward(self, x, epsilon, t, y):
        '''
        Given a clean image x, random noise epsilon and time t, sample x_t and return the noise estimation given x_t using a UNet.
        
        :param x:       Clean MNIST images [Batch size, 1, 28, 28]. 
        :param epislon: i.i.d normal noise with shape of x.
        :param t:       Time from 1 to time step t. Shape [batch_size]
        :param y:       Labels. Shape [batch_size]

        :return:        estimated_epsilon
        '''
        # Retrieve alpha_bar at timestep t, reshaped for broadcasting
        alpha_bar_t = self.alpha_bars[t].reshape(-1, 1, 1, 1)

        # Create noisy images x_t according to DDPM formulation
        x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * epsilon

        # Estimate epsilin from noisy images using UNet
        estimated_epsilon = self.model(x_t, t, y)

        return estimated_epsilon
    

