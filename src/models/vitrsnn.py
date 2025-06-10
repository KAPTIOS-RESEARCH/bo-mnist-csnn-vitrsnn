import torch.nn.functional as F
import torch
from torch import nn
from spikingjelly.activation_based import layer, neuron, surrogate
from .base import BaseJellyNet
import snntorch as snn
from snntorch import utils
import math
from icecream import ic

class SpikingMultiHeadAttention(nn.Module):
    """
    Spiking Multi-Head Attention or Cross Attention module
    """
    def __init__(self, n_output, d_model, num_heads, dropout=0.1, tau_m=2.0, v_threshold=1.0, n_hidden_RNN = 100, cross_last = True):
        super().__init__()
        self.d_model = d_model
        self.n_hidden_RNN = n_hidden_RNN
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.n_output = n_output

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(self.n_hidden_RNN, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        if cross_last:
          self.output_proj = nn.Linear(d_model, self.n_output)
        else:
          self.output_proj = nn.Linear(d_model, d_model)


        # Membrane potential for spiking neurons
        """self.membrane_q = MembranePotential(tau_m, v_threshold)
        self.membrane_k = MembranePotential(tau_m, v_threshold)
        self.membrane_v = MembranePotential(tau_m, v_threshold)
        self.membrane_out = MembranePotential(tau_m, v_threshold)"""

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        # Project inputs to Q, K, V
        """q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Generate spikes for Q, K, V using membrane potential
        _, q_spike = self.membrane_q(q, torch.zeros_like(q))  # Assume initial membrane potential is 0
        _, k_spike = self.membrane_k(k, torch.zeros_like(k))
        _, v_spike = self.membrane_v(v, torch.zeros_like(v))"""

        #ic(key.shape)
         # Adjust key shape before projection:
        #key = key.unsqueeze(-1)  # Add a dimension to match k_proj's expected shape
        #OR #key = key.repeat(1, self.head_dim)  # Repeat values to match k_proj's expected shape
        #ic(key.shape)
        q = self.q_proj(query) #try with linear layer before attention
        k = self.k_proj(key)
        v = self.v_proj(value)

        q_spike = q
        k_spike = k
        v_spike = v

        # Reshape for multi-head attention
        #ic("before",q.shape)
        # Reshape for multi-head attention
        q_spike_heads = q.view(batch_size, -1, self.d_model).transpose(1, 2)
        k_spike_heads = k.view(batch_size, -1, self.d_model).transpose(1, 2)
        v_spike_heads = v.view(batch_size, -1, self.d_model).transpose(1, 2)

        """q_spike_heads = q_spike.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_spike_heads = k_spike.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_spike_heads = v_spike.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        """
        #ic("q spike",q_spike_heads.shape)
        # Scaled dot-product attention with spikes
        attn_scores = torch.matmul(q_spike_heads, k_spike_heads.transpose(-2, -1)) / self.scale
        #ic("before before",attn_scores.shape)
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1) #attn_scores #
        #attn_probs = self.dropout(attn_probs)

        # Apply attention to value spikes
        attn_output = torch.matmul(attn_probs, v_spike_heads)
        #ic("before",attn_output.shape)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        #ic("after",attn_output.shape)

        # Project back to output space
        attn_output = self.output_proj(attn_output)
        #ic("after, after",attn_output.shape)
        #ic(self.output_proj.out_features)
        # Apply membrane dynamics to output to get output spike
        #out_spike = snn.Leaky(attn_output, output = False)
        #_, out_spike = self.membrane_out(attn_output, torch.zeros_like(attn_output))

        return attn_output #out_spike  # Return the output spike for this timestep



class AttentionMerge(nn.Module):
        def __init__(self, embed_dim, num_heads=1):
            super().__init__()
            self.query = nn.Parameter(torch.randn(1, 1, embed_dim))  # Learnable query
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        def forward(self, x):  # x: (B, num_patches, embed_dim)

            B = x.size(0)
            # ic(B)
            # ic(x.shape) #64 64 64
            q = self.query.expand(B, -1, -1)  # (B, 1, embed_dim)
            # ic(q.shape)
            attn_output, _ = self.attn(q, x, x)  # Output: (B, 1, embed_dim)
            return attn_output.squeeze(1)



class RSNN_VIT_Base_2(nn.Module):
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


        self.crossATT = SpikingMultiHeadAttention(n_output= self.n_output,d_model = d_model, num_heads=1, dropout=0.1, tau_m=2.0, v_threshold=1.0, n_hidden_RNN= self.n_hidden_RNN ,cross_last = self.cross_last) #n_timesteps= n_steps

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

        )
        print(f"SNN_Base_Mock initialized with input size: {n_input}")


        ### RNN block
        self.fcR = nn.Linear(self.n_hidden, self.n_hidden_RNN)
        self.Rleaky_1 = snn.RLeaky(beta=0.95, linear_features=self.n_hidden_RNN, output = True, learn_threshold=learn_threshold, learn_beta=learn_beta) #, spike_grad=snn.surrogate.sigmoid(),learn_beta= learn_beta)
        #self.fcR2 = nn.Linear(self.n_hidden_RNN, self.n_hidden)
        #self.Rleaky_2 = snn.Leaky(beta=self.beta, threshold= self.threshold, init_hidden=True, output = False) #,learn_threshold=learn_threshold, learn_beta=learn_beta)

        self.Leaky_out = snn.Leaky(beta=self.beta, threshold= self.threshold, init_hidden=True, output = True, learn_threshold=learn_threshold, learn_beta=learn_beta) #,learn_threshold=learn_threshold, learn_beta=learn_beta)


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



class ViT_RSNN_NoConv_MultiOutput(RSNN_VIT_Base_2):
    def __init__(self, image_size=224, patch_size=16, in_channels=1, embed_dim=64, n_hidden_RNN=100,
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
            spk_out_parallel_3 = self.layers[:7](x_flat) #7 # mem_out_parallel
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

