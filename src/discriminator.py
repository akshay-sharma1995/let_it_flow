import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import sys


class disc(nn.Module):

    def __init__(self, input_channels=1, lr=1e-3):

        super(disc, self).__init__()

        self.conv_model = nn.Sequential(
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
        self.linear = nn.Sequential(
                                        nn.Linear(in_features=encoded_dim, out_features=encoded_dim//9, bias=True),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(encoded_dim//9),
                                        
                                        nn.Linear(in_features=encoded_dim//9, out_features=encoded_dim//36, bias=True),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(encoded_dim//36),

                                        nn.Linear(in_features=encoded_dim//36, out_features=encoded_dim//108, bias=True),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(encoded_dim//108),
                                        
                                        nn.Linear(in_features=encoded_dim//108, out_features=encoded_dim//(8*27*9), bias=True),
                                        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    
    def forward(self, x):
        '''
        x: (batch_size, channels, height, width)
            Each sample in the batch is one image
        '''
        batch_size = x.shape[0]
        out = self.conv_model(x)
        out = out.reshape(batch_size,-1)
        prob = torch.sigmoid(self.linear(out))

        return out
