import os
import numpy as np
import torch
import torch.nn as nn
import pdb
import sys

class gen(nn.Module):
    def __init__(self, input_channels= 2, latent_dim=108, out_channels=2, lr=1e-3):
        
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
                                    nn.Conv2d(in_channels=input_channels, out_channels=4, kernel_size=3, stride=1),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(4),
                                    
                                    nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2),
                                    nn.MaxPool2d(kernel_size=3, stride=2),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(8),
                                    
                                    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
                                    nn.MaxPool2d(kernel_size=3, stride=2),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(16),
                                    )

        encoded_dim = 16 * 27 * 9

        self.latent_code = nn.Sequential(
                                        nn.Linear(in_features=encoded_dim, out_features=encoded_dim//9, bias=True),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(encoded_dim//9),
                                        
                                        nn.Linear(in_features=encoded_dim//9, out_features=2*latent_dim, bias=True),
                                        )

        self.latent_code_decoder = nn.Sequential(
                                                nn.Linear(in_features=latent_dim, out_features=encoded_dim, bias=True),
                                                nn.ReLU(),
                                                )

        self.decoder = nn.Sequential(
                                    nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(16),
                                    nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(8),

                                    nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(8),
                                    nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=2),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(4),

                                    nn.ConvTranspose2d(in_channels=4, out_channels = 4, kernel_size=2, stride=2),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(4),
                                    nn.ConvTranspose2d(in_channels=4, out_channels = 2, kernel_size=3, stride=1),
                                    )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    def init_weights(self,):
        for param in self.parameters():
            if(len(param.shape)>1):
                torch.nn.init.xavier_normal_(param, gain=10.0)
            else:
                constant_(param, 0)
     
    def sample_noise(self, mean, logvar):
        tau = torch.randn_like(logvar).to(logvar.device)
        std = torch.exp(0.5*logvar)

        return (mean + tau*std)


    def forward(self, x):
        '''
        x: (batch_size, 2*channels, height, width)
            Each sample in the batch is a preprocessed pair of 2 consecutive images of dim (channels, height, width)
        '''
        height = x.shape[2]
        width = x.shape[3]

        encoded_vec = self.encoder(x)
        encoded_vec = encoded_vec.reshape(encoded_vec.shape[0],-1)
        noise_params = self.latent_code(encoded_vec)
        mean, logvar = noise_params[:,0:self.latent_dim], noise_params[:,self.latent_dim:]
        sample = self.sample_noise(mean, logvar)
        
        decoded_latent_code = self.latent_code_decoder(sample)
        decoded_latent_code = decoded_latent_code.reshape(decoded_latent_code.shape[0],16, 9, 27)

        optical_flow = self.decoder(decoded_latent_code)

        # optical_flow = 5 * torch.tanh(optical_flow)

        return optical_flow, mean, logvar







