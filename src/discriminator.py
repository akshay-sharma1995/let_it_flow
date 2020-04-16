import os
import numpy as np
import torch
import torch.nn as nn
import pdb
import sys


class disc(nn.Module):

    def __init__(self, input_channels=1, lr=1e-3):

        super(disc, self).__init__()

        self.conv_model = nn.Sequential(
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
                                    nn.BatchNorm2d(32),

                                    nn.MaxPool2d(kernel_size=3, stride=2),

                                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(32),

                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64),

                                    nn.MaxPool2d(kernel_size=2, stride=2),

                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64),

                                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(128),

                                    nn.MaxPool2d(kernel_size=2, stride=2), # 51 x 16 x 128
                                    )
        
        linear_input = 51*16*128
        self.linear = nn.Sequential(
                                    nn.Linear(in_features=linear_input, out_features=linear_input//4, bias=True), #104448 -> 26112
                                    nn.ReLU(),
                                    nn.BatchNorm1d(linear_input//4),
                                    nn.Linear(in_features=linear_input//4, out_features=linear_input//16, bias=True), #26112 -> 6528
                                    nn.ReLU(),
                                    nn.BatchNorm1d(linear_input//16),
                                    nn.Linear(in_features=linear_input//16, out_features=linear_input//64, bias=True), #6528 -> 1632
                                    nn.ReLU(),
                                    nn.BatchNorm1d(linear_input//64),
                                    nn.Linear(in_features=linear_input//64, out_features=1, bias=True), #1632 -> 1
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
        prob = nn.Sigmoid(self.linear(out))

        return out
