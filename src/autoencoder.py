import os
import numpy as np
import torch
import torch.nn as nn
import pdb
import sys

class auto_enc(nn.Module):
    def __init__(self, input_channels= 2, latent_dim=108, out_channels=2, lr=1e-3):
        
        '''
        input_channels: number of channels in the input image
        latent_dim: dimension of the latent noise space
        out_channels: number of channels in the optical flow (will be 2, 1 for each axis)
        '''
        super().__init__()

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
                                        
                                        nn.Linear(in_features=encoded_dim//9, out_features=latent_dim, bias=True),
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
        latent_code = self.latent_code(encoded_vec)
        
        decoded_latent_code = self.latent_code_decoder(latent_code)
        decoded_latent_code = decoded_latent_code.reshape(decoded_latent_code.shape[0],16, 9, 27)

        optical_flow = self.decoder(decoded_latent_code)


        return optical_flow







