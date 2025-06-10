import torch.nn as nn
import torch
from spikingjelly.activation_based import functional, surrogate, neuron, layer
from .base import BaseJellyNet
import math
import torch.nn.functional as F
import numpy as np
import snntorch as snn
from snntorch import utils


class SpikingMultiHeadAttention(nn.Module):
    """
    Spiking Multi-Head Attention or Cross Attention module
    """
    def __init__(self, d_model, num_heads, dropout=0.1, tau_m=2.0, v_threshold=1.0, n_hidden_RNN = 100, cross_last = True):
        super().__init__()
        self.d_model = d_model
        self.n_hidden_RNN = n_hidden_RNN
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(self.n_hidden_RNN, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        if cross_last:
          self.output_proj = nn.Linear(d_model, 1)
        else:
          self.output_proj = nn.Linear(d_model, d_model)


        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        # input Q, K, V should be spikes 
       
        # Adjust key shape before projection:
        q = self.q_proj(query) #try with linear layer before attention
        k = self.k_proj(key)
        v = self.v_proj(value)

        q_spike = q
        k_spike = k
        v_spike = v

        # Reshape for multi-head attention
        q_spike_heads = q.view(batch_size, -1, self.d_model).transpose(1, 2)
        k_spike_heads = k.view(batch_size, -1, self.d_model).transpose(1, 2)
        v_spike_heads = v.view(batch_size, -1, self.d_model).transpose(1, 2)

        """q_spike_heads = q_spike.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_spike_heads = k_spike.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_spike_heads = v_spike.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        """
       
        # Scaled dot-product attention with spikes
        attn_scores = torch.matmul(q_spike_heads, k_spike_heads.transpose(-2, -1)) / self.scale
       
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1) #attn_scores #
       
        # Apply attention to value spikes
        attn_output = torch.matmul(attn_probs, v_spike_heads)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
       
        # Project back to output space
        attn_output = self.output_proj(attn_output)
     
        return attn_output   # Return the output spike for this timestep





# RSNN VIT last archi
class RSNN_VIT_Base_2(nn.Module):
    def __init__(self, n_input=224*224, n_hidden=1000, n_hidden_RNN =100, n_output=1, n_steps=5, beta=0.8, threshold = .5,cross_last = False):
        super().__init__()
        # two_Rlayers = False,
        self.n_input = n_input
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


        self.crossATT = SpikingMultiHeadAttention(d_model = d_model, num_heads=1, dropout=0.1, tau_m=2.0, v_threshold=1.0, n_hidden_RNN= self.n_hidden_RNN ,cross_last = self.cross_last) #n_timesteps= n_steps

        # SNN block
        self.layers = nn.Sequential(

            nn.Linear(self.n_input, self.n_hidden),
            snn.Leaky(beta=self.beta, threshold= self.threshold, init_hidden=True, output=False,learn_threshold=learn_threshold), #,learn_threshold=learn_threshold, learn_beta=learn_beta

            nn.Linear(self.n_hidden, self.n_hidden),
            snn.Leaky(beta=self.beta, threshold= self.threshold, init_hidden=True, output=False,learn_threshold=learn_threshold), #,learn_threshold=learn_threshold, learn_beta=learn_beta),

            nn.Dropout(0.2),
            nn.Linear(self.n_hidden, self.n_hidden),
            snn.Leaky(beta=self.beta, threshold= self.threshold, init_hidden=True, output = False,learn_threshold=learn_threshold), #, learn_threshold=learn_threshold, output = True),

        )
        print(f"RSNN VIT initialized with input size: {self.n_input}, hidden size: {self.n_hidden} and output: {self.n_output}")


        ### RNN block
        self.fcR = nn.Linear(self.n_hidden, self.n_hidden_RNN)
        self.Rleaky_1 = snn.RLeaky(beta=0.95, linear_features=self.n_hidden_RNN, output = True) #, spike_grad=snn.surrogate.sigmoid(),learn_beta= learn_beta, learn_threshold=learn_threshold)
        self.Leaky_out = snn.Leaky(beta=self.beta, threshold= self.threshold, init_hidden=True, output = True) #,learn_threshold=learn_threshold, learn_beta=learn_beta)


        def reset_hidden_states(self):
          for layer in [self.Leaky_1, self.Leaky_2, self.Leaky_3]:
              #  if the layer has a reset_hidden_states method
              if hasattr(layer, 'reset_hidden_states'):
                  layer.reset_hidden_states()
              # If layer has reset_state 
              elif hasattr(layer, 'reset_state'):
                  layer.reset_state()

        def forward(self, data):
          raise NotImplementedError("Subclasses must implement forward method")



class RSNN_ViT_2(RSNN_VIT_Base_2): 
   def __init__(self, img_size=128, patch_size=16, in_channels=1, embed_dim=64,
                n_hidden=1000, n_output=1, n_steps=5, beta=.8, threshold=.5, cross_last=True):

       patch_overlap=0.5

       # --- Patching & Embedding Params
       self.img_size = img_size
       self.patch_size = patch_size
       self.in_channels = in_channels
       self.embed_dim = embed_dim # patch embedding dimension

       # --- Compute number of Patches 
       if patch_overlap > 0:        
            self.stride = int(patch_size * (1 - patch_overlap))
            num_patches_h = math.floor((img_size - patch_size) / self.stride) + 1
            num_patches_w = math.floor((img_size - patch_size) / self.stride) + 1
            num_patches = num_patches_h * num_patches_w
            self.num_patches = num_patches
       else:
            num_patches = (img_size // patch_size) * (img_size // patch_size)


       # input size for the SNN. Why?? total dimension after flattening all patch embeddings is numpatch * embed dim
       snn_input_size = num_patches * embed_dim
       print(f"Overlap: {patch_overlap*100}%, Stride: {self.stride}")
       print(f"Number of patches: {num_patches_h} x {num_patches_w} = {num_patches}")
       print(f"Calculated SNN input size: {snn_input_size}")

       # --- Initialize Base SNN ---
       # it gives the input size derived from patches calculation
       super().__init__(n_input=snn_input_size, n_hidden=n_hidden, n_output=n_output,
                        n_steps=n_steps, beta=beta, threshold=threshold, cross_last=cross_last)

       # --- Patch Embedding Layer ---
       # patching and linear projection is done here in one step.
       self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                    kernel_size=patch_size, stride=self.stride)
       #Try with nn.Linear!!


       # --- Positional Embedding ---
       # Learnable parameters for position information. # 1 because we want to keep the same position information across each batch
       self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) #(1, 64, 64) 
       # Other posible type of Init: nn.init.trunc_normal_(self.pos_embed, std=.02)

       
       self.ln_attn_2 = nn.LayerNorm(self.n_hidden_RNN)

   def forward(self, data):
        # data shape (64, 1, 128, 128)
        batch_size = data.shape[0]
        #self.ln_attn_1 = nn.LayerNorm(batch_size)

        # 1- Do the patching and Linear projection
        #(batch, embed_dim, num_patches_h, num_patches_w)# (64, 64, 16, 16)
        x = self.patch_embed(data)

        #2- Flatten spatial dimensions and permute
        # Shape: (batch, embed_dim, num_patches) -> (batch, num_patches, embed_dim) #(64, x, y) => (64, y, x)
        x = x.flatten(2).transpose(1, 2)

        #3- Add Positional Embedding #(batch, num_patches, embed_dim) + (1, num_patches, embed_dim)
        x = x + self.pos_embed

        #4- flatten patches
        # (batch, num_patches * embed_dim) # (64, 64 * 64) = (64, 4096)
        x_flat = x.flatten(1)

        #-- Record SNN spikes and membrane potential during simulation ---
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []
        memout_rec = [] # output membrane potential
        spkout_rec = [] # output spikes

        # To reset hidden states of leaky neurons - be sure snn.Leaky layers have init_hidden=True for this to work
        utils.reset(self.layers) 
        Rleaky_1, mem_rleaky = self.Rleaky_1.init_rleaky()
        mem_out_final= self.Leaky_out.init_leaky()


        # Input should be the flattened patches 
        for step in range(self.n_steps):

            # Branch for RNN
            spk_out = self.layers[:2](x_flat)
            if isinstance(spk_out, tuple):
                spk_out = spk_out[0]
            spk1_rec.append(spk_out)



            #spk_out_parallel_1, mem_11 = self.layers[:2](spk_patch) #mem_out_parallel
            #spk_out_parallel_1 = self.layers[:2](spk_patch) #mem_out_parallel
            #spk_out_parallel_2, mem_12 = self.layers[3:4](spk_out_parallel_1) #mem_out_parallel
            #spk_out_parallel_2 = self.layers[3:4](spk_out_parallel_1) #mem_out_parallel
            #spk_out_parallel_3, mem_13 = self.layers[5:6](spk_out_parallel_2) #mem_out_parallel

            # Main Branch
            spk_out_parallel_3 = self.layers[:7](x_flat) #mem_out_parallel

            if isinstance(spk_out_parallel_3, tuple):
                spk_out_parallel_3 = spk_out_parallel_3[0]
            spk2_rec.append(spk_out_parallel_3)

            # RNN block
            RLinear_1 = self.fcR(spk_out)
            #RLinear_1 = self.ln_attn_2(RLinear_1)
            Rleaky_1, mem_rleaky  = self.Rleaky_1(RLinear_1, Rleaky_1, mem_rleaky)
       
            if isinstance(Rleaky_1, tuple):
                Rleaky_1 = Rleaky_1[0]

            spk3_rec.append(Rleaky_1)

            # Attention block
            query = Rleaky_1
            key = spk_out_parallel_3
            value = spk_out_parallel_3
            crossATT_ouput = self.crossATT(query, key, value) # output = 1 linear when cross LAst = True
            crossATT_ouput = crossATT_ouput.squeeze(1)
            #print(crossATT_ouput.shape)
            #crossATT_ouput = self.ln_attn_1(crossATT_ouput.transpose(0, 1))  # normalize here

           
            # Last Leaky layer
            Leaky_out_final, mem_out_final = self.Leaky_out(crossATT_ouput)
            mem_out_final = mem_out_final.detach()
            spkout_rec.append(Leaky_out_final)
            memout_rec.append(mem_out_final)



      
        mem_rec = torch.stack(memout_rec)        
        spk1s = torch.stack(spk1_rec).sum().item()
        spk2s = torch.stack(spk2_rec).sum().item()
        spk3s = torch.stack(spk3_rec).sum().item()
        spk4s = torch.stack(spkout_rec).sum().item() 

        # Summing spikes across time 
        spk_total = spk1s + spk2s + spk3s + spk4s
   
        spike_train = torch.stack(spkout_rec).sum(dim=0).squeeze() #spkout_rec.sum(dim=0) # (batch, n_output) : (32, 1)
        #print(spike_train.shape)
        memtot_aggregated = mem_rec.sum(dim=0) # (batch, n_output) : (32, 1)
   
     
        return spike_train, spk_total #memtot_aggregated.clone(),




class ViT_RSNN_NoConv(RSNN_VIT_Base_2):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=64, n_hidden_RNN=100,
                 n_hidden=1000, n_output=1, n_steps=5, beta=.8, threshold=.5, cross_last=True):
        patch_overlap = 0.5
        # --- Patching & Embedding Params ---
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim  # Dimension of each patch embedding
        # num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.stride = int(patch_size * (1 - patch_overlap))

        num_patches_h = math.floor((self.img_size - patch_size) / self.stride) + 1
        num_patches_w = math.floor((self.img_size - patch_size) / self.stride) + 1
        num_patches = num_patches_h * num_patches_w
        self.num_patches = num_patches

        # Calculate the effective input size for the SNN part
        # It's the total dimension after flattening all patch embeddings
        snn_input_size = num_patches * embed_dim
        print(f"Overlap: {patch_overlap * 100}%, Stride: {self.stride}")
        print(f"Number of patches: {num_patches_h} x {num_patches_w} = {num_patches}")
        print(f"Calculated SNN input size: {snn_input_size}")

        # --- Initialize Base SNN ---
        # Pass the *correct* input size derived from patches
        super().__init__(n_input=snn_input_size, n_hidden=n_hidden, n_output=n_output,
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
        self.lm_spN = 0.0058  # -0.0058#nn.Parameter(torch.zeros(1)) #-0.0058 #round(torch.tensor(-0.0058), 4) # nn.Parameter(torch.zeros(1)) #tensor(-0.0058))

        # Optional: Initialize pos_embed more sophisticatedly (e.g., truncated normal)
        # nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, data):
        # data shape: (batch, channel, height, width)
        B, C, H, W = data.shape

        # 1. Unfold image into patches with possible overlap
        patches = data.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)
        num_patches_h, num_patches_w = patches.size(2), patches.size(3)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)  # (B, N_patches, C, P, P)
        patches = patches.flatten(2)  # (B, N_patches, patch_dim)

        # 2. Apply linear projection
        x = self.patch_embed(patches)  # (B, N_patches, embed_dim)

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
        Rleaky_1, mem_rleaky = self.Rleaky_1.init_rleaky()
        mem_out_final = self.Leaky_out.init_leaky()
        reg_loss = 0
        total_spikes = 0
        total_spikes_prev = 0
        # Present the same processed input (flattened patches) at each time step
        for step in range(self.n_steps):
            # Pass the flattened patch data through the SNN layers

            # For RNN
            spk_out = self.layers[:2](x_flat)
            if isinstance(spk_out, tuple):
                spk_out = spk_out[0]
            total_spikes += spk_out.mean()
            # spk1_rec.append(spk_out)


            # Parallel
            spk_out_parallel_3 = self.layers[:7](x_flat)  # mem_out_parallel
            total_spikes += spk_out_parallel_3.mean()

            if isinstance(spk_out_parallel_3, tuple):
                spk_out_parallel_3 = spk_out_parallel_3[0]

            # RNN
            RLinear_1 = self.fcR(spk_out)
            Rleaky_1, mem_rleaky = self.Rleaky_1(RLinear_1, Rleaky_1, mem_rleaky)
            total_spikes += Rleaky_1.mean()

            if isinstance(Rleaky_1, tuple):
                Rleaky_1 = Rleaky_1[0]

            # spk3_rec.append(Rleaky_1)

            # Attention
            query = Rleaky_1
            key = spk_out_parallel_3
            value = spk_out_parallel_3
            crossATT_ouput = self.crossATT(query, key, value)  # output = 1 linear when cross LAst = True
            crossATT_ouput = crossATT_ouput.squeeze(1)
            # ic(crossATT_ouput.shape)
            # Last Leaky
            Leaky_out_final, mem_out_final = self.Leaky_out(crossATT_ouput)
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

        # Stack results across time steps
        # Shape: (n_steps, batch, n_output), e.g., (5, 32, 1)
        mem_rec = torch.stack(memout_rec)  # Contains membrane potential of output layer

        memtot_aggregated = mem_rec.sum(dim=0)  # Shape: (batch, n_output), e.g., (32, 1)

        spk_rec = torch.stack(spkout_rec)

        reg_loss_norm = reg_loss / self.n_steps
        tot_spikes_norm = total_spikes / self.n_steps
        # ic(reg_loss_norm, tot_spikes_norm)
        return spk_rec.sum(0).squeeze() / self.n_steps, reg_loss_norm  # memtot_aggregated.clone(), ,self.lm_spN, tot_spikes_norm


"""
class CSNN(BaseJellyNet):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, n_steps=5, encoding_method='direct'):
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
            layer.Linear(256 * 16 *16, 16 * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            

            layer.Linear(16 * 4 * 4, out_channels, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

        functional.set_step_mode(self, step_mode='m')

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