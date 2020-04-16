import os
import numpy as np
import torch
import torch.nn as nn
import pdb
import sys


class gen(nn.Module):
    def __init__(self, input_channels=1, latent_dim=100, out_channels=2, lr=1e-3):
        
        '''
        input_channels: number of channels in the input image
        latent_dim: dimension of the latent noise space
        out_channels: number of channels in the optical flow (will be 2, 1 for each axis)
        '''
        super(gen, self).__init__()

        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.lr = lr
        
        ## TODO: finalise the input image dim, and accordingly modify the network including the latent dimensions

        self.encoder = nn.Sequential(
                                    nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(8),
                                    
                                    nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(8),
                                    
                                    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(16),
                                    
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    
                                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(16),
                                    
                                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
                                    nn.ReLU(),

                                    nn.MaxPool2d(kernel_size=3, stride=2),
                                    
                                    )

        encoded_dim = 220*76*32

        self.latent_code = nn.Sequential(
                                        nn.Linear(in_features=encoded_dim, out_features=encoded_dim//8, bias=True),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(encoded_dim//8),
                                        
                                        nn.Linear(in_features=encoded_dim//8, out_features=encoded_dim//64, bias=True),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(encoded_dim//64),
                                        
                                        nn.Linear(in_features=encoded_dim//64, out_features=encoded_dim//512, bias=True),
                                        nn.ReLU(),

                                        nn.Linear(in_features=encoded_dim, out_features=2*latent_dim, bias=True),
                                        )

        self.latent_code_decoder = nn.Sequential(
                                                nn.Linear(in_features=latent_dim, out_features=encoded_dim, bias=True),
                                                nn.ReLU(),
                                                )

        self.decoder = nn.Sequential(
                                    nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
                                    nn.ReLU(),
                                    
                                    nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    
                                    nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    
                                    nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2),
                                    nn.ReLU(),
                                    
                                    nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=1),
                                    nn.ReLU(),

                                    nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=1),
                                    nn.ReLU(),

                                    nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=1),
                                    nn.tanh(),
                                    )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def sample_noise(self, mean, logvar):
        tau = torch.randn_like(logvar).to(logvar.device)
        std = torch.exp(0.5*logvar)

        return (mean + tau*std)


    def forward(self, x):
        '''
        x: (batch_size, 2*channels, height, width)
            Each sample in the batch is a preprocessed pair of 2 consecutive images of dim (channels, height, width)
        '''
        
        encoded_vec = self.encoder(x)
        
        noise_params = self.latent_code(encoded_vec)
        mean, logvar = noise_params[:,0:self.latent_dim], noise_params[:,self.latent_dim:]
        sample = self.sample_noise(mean, logvar)
        
        decoded_latent_code = self.latent_code_decoder(sample),
        optical_flow = self.decoder(decoded_latent_code)
        
        return optical_flow, mean, logvar







