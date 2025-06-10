import torch.nn as nn
import torch
import torch.nn.functional as F
from icecream import ic
import snntorch  as snn
from snntorch import utils
from icecream import ic


class CSNN_bis_reg(nn.Module):
    def __init__(self, image_size, n_input, n_hidden = 512, n_output = 1, threshold=.5, beta=.95, n_steps=5):
        super(CSNN_bis_reg, self).__init__()
        ic(image_size)
        if image_size == 64:
            self.in_linear = 16384
        elif image_size == 224:
            self.in_linear = 93312
        elif image_size == 128:
            self.in_linear = 28800
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
            # nn.BatchNorm1d(hidden_size),
            nn.MaxPool2d(2),
            snn.Leaky(beta=self.beta, threshold=self.threshold, init_hidden=True),
            nn.Flatten(),

            nn.Linear(self.in_linear, 1),#186624 #230400:  124x124
            # nn.Conv2d(hidden_size, output_size, kernel_size=3, padding=1),
            snn.Leaky(init_hidden=True, beta=self.beta, threshold=self.threshold, output=True),
        ) #.to(device)

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
        spike_trains = torch.stack(spike_rec).sum(0).squeeze()
        # ic(spike_trains.shape)
        reg_loss_norm = reg_loss / self.n_step
        return spike_trains, reg_loss_norm  # tot_spikes, torch.stack(mem_rec), spike_trains


