import torch.nn as nn
from spikingjelly.activation_based import functional, surrogate, neuron, layer
from .base import BaseJellyNet
import torch
import torch.nn as nn

import snntorch as snn
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic  # Assuming icecream is used in your code
from snntorch import utils
import random
import wandb



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model (copied from your code)
class CSNN_bis_reg(nn.Module):
    def __init__(self, image_size, n_input, n_hidden = 512, n_output = 11, threshold=.5, beta=.95, n_steps=5):
        super(CSNN_bis_reg, self).__init__()
        ic("Input", image_size)
        self.n_output = n_output
        if image_size == 224:
            self.in_linear = 93312
        elif image_size == 128:
            #self.in_linear = 800
            self.in_linear = 28800
        elif image_size == 64:
            self.in_linear = 6272
        else:
            print('linear input to be defined')

        self.lm_spN = 1e-8
        self.hidden_size = n_hidden
        self.output_size = n_output
        self.input_size = n_input
        self.image_size = image_size
        self.threshold = threshold
        self.beta = beta
        self.n_step = n_steps

        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=1), #
            nn.MaxPool2d(2),
            snn.Leaky(beta=self.beta, threshold=self.threshold, init_hidden=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=1),#128 #256
            # nn.BatchNorm1d(hidden_size), #nn.InstanceNorm(Normalization normalise sur des dimensions
            nn.MaxPool2d(2),
            snn.Leaky(beta=self.beta, threshold=self.threshold, init_hidden=True),

           
            nn.Flatten(),

            nn.Linear(self.in_linear, self.n_output),#186624 #230400:  124x124
            # nn.Conv2d(hidden_size, output_size, kernel_size=3, padding=1),
            snn.Leaky(init_hidden=True, beta=self.beta, threshold=self.threshold, output=True),
        ).to(device)

    def forward(self, x):
        utils.reset(self.layers)
        spike_rec = []
        mem_rec = []
        tot_spikes = 0
        reg_loss = 0
        # ic(x.shape)
        # x = x.view(x.size(0), -1)
        # ic(x.shape)

        for n in range(self.n_step):
            # x = x.view(x.shape[0], -1)#,data.shape[1] #flatten

            spikes = self.layers[:3](x)
            tot_spikes += spikes.sum()

            spikes_2 = self.layers[3:6](spikes)
            tot_spikes += spikes_2.sum()
            #ic(spikes_2.shape)

            spikes_3, mem = self.layers[6:9](spikes_2)
            tot_spikes += spikes_3.sum()

            # Activity regularization: encourage sparse firing
            reg_loss += self.lm_spN * (tot_spikes)

            # Rate regularization: penalize high firing rates
            if n > 0:
                reg_loss += self.lm_spN * torch.mean(torch.abs(tot_spikes - total_spikes_prev))

            total_spikes_prev = tot_spikes  # .detach()

            spike_rec.append(spikes_3)
            mem_rec.append(mem)
        # ic(tot_spikes)
        spike_trains = torch.stack(spike_rec).sum(0) #.squeeze()
        if spike_trains.dim() == 1:
             spike_trains = spike_trains.unsqueeze(0)

        #ic(spike_trains.shape)
        #spike_trains = torch.stack(spike_rec).sum(0).squeeze()
        #ic(spike_trains.shape)
        reg_loss_norm = reg_loss / self.n_step
        return spike_trains, reg_loss_norm  # tot_spikes, torch.stack(mem_rec), spike_trains


"""class CSNN(BaseJellyNet):
    def __init__(self, in_channels: int = 1, out_channels: int = 11, n_steps=5, encoding_method='direct'):
        super().__init__(in_channels, out_channels, n_steps, encoding_method)

        self.net = nn.Sequential(
            layer.Conv2d(in_channels, 64, kernel_size=3,
                         padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            )

        self.net2 = nn.Sequential(
        
            layer.Conv2d(64, 128, kernel_size=3,
                         padding=1, bias=False),
            layer.BatchNorm2d(128),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),)
            
        self.net3 = nn.Sequential(layer.MaxPool2d(2, 2),
            layer.Conv2d(128, 256, kernel_size=3,
                         padding=1, bias=False),
            layer.BatchNorm2d(256),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),)
        

        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(256 * 16 *16, 16 * 4 * 4, bias=False), #256 * 16 *16
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            

            layer.Linear(16 * 4 * 4, out_channels, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

        functional.set_step_mode(self, step_mode='s')

    def forward(self, x):
        x_sum = 0
        
        x = x.unsqueeze(0).repeat(self.n_steps, 1, 1, 1, 1)
        
        x = self.encode_input(x)
        x = self.net(x)
        x_sum += x.sum(dim = (0,1,2,3,4))
 
        x2 = self.net2(x)
        x_sum += x2.sum(dim = (0,1,2,3,4))
        
        x3= self.net3(x2)
        x_sum += x3.sum(dim = (0,1,2,3,4))
        
        x4 = self.classifier(x3)
        x_sum += x4.sum(dim = (0,1,2))
        
        x_out = x4.sum(0)
        

        return x_out.squeeze(1), x_sum
"""

"""class CSNN(BaseJellyNet):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, n_steps=5, encoding_method='direct'):
        super().__init__(in_channels, out_channels, n_steps, encoding_method)

        self.net = nn.Sequential(
            layer.Conv2d(in_channels, 64, kernel_size=3,
                         padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),
        
            layer.Conv2d(64, 128, kernel_size=3,
                         padding=1, bias=False),
            layer.BatchNorm2d(128),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),
            
            layer.Conv2d(128, 256, kernel_size=3,
                         padding=1, bias=False),
            layer.BatchNorm2d(256),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),)
        

        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(256 * 16 *16, 16 * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            

            layer.Linear(16 * 4 * 4, out_channels, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        
        
        x = x.unsqueeze(0).repeat(self.n_steps, 1, 1, 1, 1)
        x = self.encode_input(x)
        x = self.net(x)
        x = self.classifier(x)
        x = x.sum(0)

        return x.squeeze(1)"""
