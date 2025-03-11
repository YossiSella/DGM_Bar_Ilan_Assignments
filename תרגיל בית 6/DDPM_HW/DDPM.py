"""DDPM model for MNIST
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from UNet import Unet
class DDPM(nn.Module):
    def __init__(self, timestepstimesteps=1000, guidance=False,device='cpu'):
        """
        DDPM model for conditional MNIST generation
        :param timesteps: Number of timesteps in the generation
        :param guidance: Run with or without classifier free guidance.
        :param device: cpu or gpu
        """
        super(DDPM, self).__init__()
        self.device = device
        self.guidance = guidance
        self.timesteps = 1000
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

        :return: samples 100 images conditioned on y =torch.tensor([0,1,2,3,4,5,6,7,8,9]*10).to(self.device)
        '''
        y = torch.tensor([0,1,2,3,4,5,6,7,8,9]*10).to(self.device)
        x = torch.randn(100,1,28,28).to(self.device)
        #TODO

        return x.to('cpu')

    def forward(self, x,epsilon,t,y):
        '''
        Given a clean image x, random noise epsilon and time t, sample x_t and return the noise estimation given x_t
        :param x: Clean MNIST images
        :param epislon: i.i.d normal noise size of x
        :param t: time from 1 to time step
        :param y: labels
        :return: estimated_epsilon
        '''
        #TODO



