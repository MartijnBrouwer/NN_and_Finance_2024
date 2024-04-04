# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:01:02 2024

@author: joche
"""
import torch
import torch.nn as nn

class AutoencoderFNN(nn.Module):
    def __init__(self, input_size, hidden_size,latent_space):
        super(AutoencoderFNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, latent_space),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_space, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, input_size),
            
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    
class AutoencoderV1(nn.Module):
    def __init__(self, input_size, hidden_size,latent_space):
        super(AutoencoderV1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    
# Convolution autoencoders Network
# class CAE(nn.Module):
#     def __init__(self):
#         super(CAE, self).__init__()
        
#         self.encoder = nn.Sequential(
#                     nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
#                     nn.ReLU(),
#                     nn.MaxPool1d(kernel_size=2, stride=2),
#                     nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
#                     nn.ReLU(),
#                     nn.MaxPool1d(kernel_size=2, stride=2),
#                 )
        
#         # Decoder layers
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=5, stride=2, padding=1, output_padding=1),
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)
#         # encoded = encoded.unsqueeze(1)
#         decoded = self.decoder(encoded)
#         return decoded


import torch.nn as nn

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        
        self.encoder = nn.Sequential(
                    nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    # nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
                    # nn.ReLU(),
                    # nn.MaxPool1d(kernel_size=2, stride=2),
                )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            # nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=5, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


print(CAE())