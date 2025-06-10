import torch.nn as nn
from spikingjelly.activation_based import functional, surrogate, neuron, layer
from .base import BaseJellyNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from icecream import ic  # Assuming icecream is used in your code
from snntorch import utils
import random
import wandb
import time
import math



class deep_SNN_Base(nn.Module):
    def __init__(self, n_input, n_hidden=1000, n_hidden_RNN =100, n_output=11, n_steps=5, beta=0.95, threshold = .5,cross_last = False):
        super().__init__()
        # two_Rlayers = False,
        self.n_input = n_input #224*224
        self.n_hidden = n_hidden
        self.n_hidden_RNN = n_hidden_RNN
        d_model = n_hidden
        self.n_output = n_output
        self.n_steps = n_steps
        self.beta = beta
        self.threshold = threshold
        self.cross_last = cross_last

        learn_threshold = False
        learn_beta = False


         # Define the actual SNN layers here based on parameters
        # This structure is assumed based on your original forward pass logic
        self.layers = nn.Sequential(

            nn.Linear(self.n_input, self.n_hidden),
            snn.Leaky(beta=self.beta, threshold= self.threshold, init_hidden=True, output=False, learn_threshold=learn_threshold, learn_beta=learn_beta), #,learn_threshold=learn_threshold, learn_beta=learn_beta

            nn.Linear(self.n_hidden, self.n_hidden),
            snn.Leaky(beta=self.beta, threshold= self.threshold, init_hidden=True, output=False, learn_threshold=learn_threshold, learn_beta=learn_beta), #,learn_threshold=learn_threshold, learn_beta=learn_beta),

            nn.Dropout(0.3),
            nn.Linear(self.n_hidden, self.n_hidden),
            snn.Leaky(beta=self.beta, threshold= self.threshold, init_hidden=True, output = False, learn_threshold=learn_threshold, learn_beta=learn_beta), #, learn_threshold=learn_threshold, output = True),

            nn.Dropout(0.3),
            nn.Linear(self.n_hidden, self.n_hidden),
            snn.Leaky(beta=self.beta, threshold= self.threshold, init_hidden=True, output = False, learn_threshold=learn_threshold, learn_beta=learn_beta), #, learn_threshold=learn_threshold, output = True),

            nn.Dropout(0.3),
            nn.Linear(self.n_hidden, self.n_output),
            snn.Leaky(beta=self.beta, threshold= self.threshold, init_hidden=True, output = True, learn_threshold=learn_threshold, learn_beta=learn_beta), #, learn_threshold=learn_threshold, output = True),

        )
        print(f"SNN_Base_Mock initialized with input size: {n_input}")


        
        def reset_hidden_states(self):
          """Resets hidden states of Leaky layers"""
          for layer in [self.Leaky_1, self.Leaky_2, self.Leaky_3]:
              # Check if the layer has a reset_hidden_states method
              if hasattr(layer, 'reset_hidden_states'):
                  layer.reset_hidden_states()
              # If not, check for reset_state (older versions)
              elif hasattr(layer, 'reset_state'):
                  layer.reset_state()

        def forward(self, data):
          raise NotImplementedError("Subclasses must implement forward method")



class ViT_deepSNN(deep_SNN_Base):
    def __init__(self, n_input, image_size=224, patch_size=16, in_channels=1, embed_dim=64, n_hidden_RNN=100,
                 n_hidden=1000, n_output=11, n_steps=5, beta=.8, threshold=.5, cross_last=True):
        patch_overlap = 0.5
        self.total_spikes_possible = 64 * 5 * 1000 * 6
        # --- Patching & Embedding Params ---
        self.img_size = image_size
        self.patch_size = patch_size
        self.n_output = n_output
        self.in_channels = in_channels
        self.embed_dim = embed_dim  # Dimension of each patch embedding
        # num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.stride = int(patch_size * (1 - patch_overlap))

        #num_patches_h = math.floor((image_size - patch_size) / self.stride) + 1
        num_patches_h = (image_size - patch_size) // self.stride + 1
        #num_patches_w = math.floor((image_size - patch_size) / self.stride) + 1
        num_patches_w = (image_size - patch_size) // self.stride + 1
        num_patches = num_patches_h * num_patches_w
        self.num_patches = num_patches

        # Calculate the effective input size for the SNN part
        # It's the total dimension after flattening all patch embeddings
        snn_input_size = num_patches * embed_dim
        print(f"Image size: {image_size} x {image_size}, Patch size: {patch_size} x {patch_size}= {patch_size * patch_size}, ")
        print(f"Input channels: {in_channels}, Patch embedding dimension: {embed_dim}")
        print(f"Overlap: {patch_overlap * 100}%, Stride: {self.stride}")
        print(f"Number of patches: {num_patches_h} x {num_patches_w} = {num_patches}")
        print(f"Calculated SNN input size: {snn_input_size}")

        # --- Initialize Base SNN ---
        # Pass the *correct* input size derived from patches
        super().__init__(n_input=snn_input_size, n_hidden=n_hidden, n_output=self.n_output,
                         n_steps=n_steps, n_hidden_RNN=n_hidden_RNN, beta=beta, threshold=threshold, cross_last=cross_last)

        # --- Patch Embedding Layer ---
        # This layer does both patching and linear projection in one step.
        # Kernel size and stride = patch size ensures non-overlapping patches.

        #self.patch_embed = nn.Conv2d(in_channels, embed_dim,kernel_size=patch_size, stride=self.stride)

        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        # --- Positional Embedding ---
        # Learnable parameters for position information. Add 1 for potential CLS token if used later.
        # Shape: (1, num_patches, embed_dim) -> allows broadcasting across batch
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.lm_spN = 0.001 #058  # -0.0058#nn.Parameter(torch.zeros(1)) #-0.0058 #round(torch.tensor(-0.0058), 4) # nn.Parameter(torch.zeros(1)) #tensor(-0.0058))

        # Optional: Initialize pos_embed more sophisticatedly (e.g., truncated normal)
        # nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        # data shape: (batch, channel, height, width)
        # add noise to image
        data = x# torch.clamp(x + 0.1 * torch.randn_like(x), 0, 1)
        B, C, H, W = data.shape
        batch_size = data.shape[0]

        # 1. Unfold image into patches with possible overlap
        patches = data.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)
        #print(f"patched image shape unfold 0: {patches.shape}")
        # patches shape: (batch, channels, num_patches_h, num_patches_w, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        #print(f"patched image shape 1 contiguous: {patches.shape}")
        # patches shape: (batch, channels, num_patches, patch_size, patch_size)
        
        #num_patches_h, num_patches_w = patches.size(2), patches.size(3)
        #patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        
        #patches = patches.permute(0, 2, 1, 3, 4).contiguous()   # (B, N_patches, C, P, P)
        #patches = patches.view(batch_size, -1, self.in_channels * self.patch_size * self.patch_size)  # (B, N_patches, patch_dim)
        patches = patches.permute(0, 2, 1, 3, 4)  # (B, N_patches, C, P, P)
        patches = patches.flatten(2)  # (B, N_patches, patch_dim)
        #print(f"patched image shape: {patches.shape}")

        # 2. Apply linear projection
        x = self.patch_embed(patches)  # (B, N_patches, embed_dim)
        #print(f"X shape: {x.shape}")
        # 3. Add positional encoding
        x = x + self.pos_embed  # assumes pos_embed shape is (1, N_patches, embed_dim)

        # 4. Flatten patches for SNN input
        # Shape: (batch, num_patches * embed_dim)
        # e.g., (32, 64 * 64) = (32, 4096)
        x_flat = x.flatten(1)

        # --- SNN Processing Loop ---
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []
        memout_rec = []  # Record membrane potential (if needed)
        spkout_rec = []  # Record output spikes

        # Reset hidden states of leaky neurons at the beginning of the sequence
        # Ensure your snn.Leaky layers have init_hidden=True for this to work
        utils.reset(self.layers)  # Reset layers defined in SNN_Base (or defined here)
        #mem_out_final = self.Leaky_out.init_leaky()
        reg_loss = 0
        total_spikes = 0
        total_spikes_prev = 0
        # Present the same processed input (flattened patches) at each time step
        for step in range(self.n_steps):
            # Pass the flattened patch data through the SNN layers

            # For RNN
            Leaky_out_final, mem_out_final = self.layers(x_flat)
            """if isinstance(spk_out, tuple):
                spk_out = spk_out[0]"""
            
            total_spikes += Leaky_out_final.mean()
            # ic(Leaky_out_final.shape)
            mem_out_final = mem_out_final.detach()
            spkout_rec.append(Leaky_out_final)
            memout_rec.append(mem_out_final)


            # Activity regularization: encourage sparse firing
            reg_loss += self.lm_spN * (total_spikes)

            # Rate regularization: penalize high firing rates
            if step > 0:
                reg_loss += self.lm_spN * torch.mean(torch.abs(total_spikes - total_spikes_prev))

            total_spikes_prev = total_spikes  # .detach()
            # Pruning
            #inactive = total_spikes < (0.0001 * self.total_spikes_possible)
            #self.fcR.weight.data[inactive] = 0
            #self.fcR.bias.data[inactive] = 0
        # Stack results across time steps
        # Shape: (n_steps, batch, n_output), e.g., (5, 32, 1)
        mem_rec = torch.stack(memout_rec)  # Contains membrane potential of output layer

        memtot_aggregated = mem_rec.sum(dim=0)  # Shape: (batch, n_output), e.g., (32, 1)

        spk_rec = torch.stack(spkout_rec)

        reg_loss_norm = reg_loss / self.n_steps
        tot_spikes_norm = total_spikes / self.n_steps
        # ic(reg_loss_norm, tot_spikes_norm)
        #ic(spk_rec.shape)
        return spk_rec.sum(0) / self.n_steps, reg_loss_norm  # memtot_aggregated.clone(), ,self.lm_spN, tot_spikes_norm



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
