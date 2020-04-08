import os
import numpy as np
import torch
import torch.nn as nn
import pdb
import sys


class disc(nn.Module):

    def __init__(self, input_channels=3, lr=1e-3):

        super(disc, self).__init__()

        self.model = nn.Sequential(
                                    nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
                                    )
        
        self.linear = nn.Linear(in_features=512, out_features=1, bias=True)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    
    def forward(self, x):
        '''
        x: (batch_size, channels, height, width)
            Each sample in the batch is one image
        '''

        out = self.model(x)
        out = torch.reshape(-1,1)
        prob = nn.Sigmoid(self.linear(out))

        return out
