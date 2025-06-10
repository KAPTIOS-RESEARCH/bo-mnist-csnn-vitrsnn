import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import OrganAMNIST
import snntorch as snn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from icecream import ic  # Assuming icecream is used in your code
from snntorch import utils
import random
import wandb
import time

image_size = 224 #224 #64 #64 #128 #128
n_steps = 5
batch_size = 64
n_output = 11

### SimpleSNN patching
wandb_rec = True
random_num = random.randint(0, 500)
model_name = "ViT_deep_SNN"
message ="VIT_deep_SNN on OrganAMNIST seed 3" #L2 reg  #prunning #"add_transformation_noNoise"
n_hidden = 1000
n_input = image_size * image_size # 128 * 128 #224 * 224 # 64 x64
n_hidden_RNN = 100
beta = .95
threshold = 0.5
#two_Rlayers = False
lr_k = 1e-5 #1e-4 # 1e-4
n_epoch = 50 #10 #70#00 #00
count = 6
cross_last = True
lm_spk_num = 0.01 #058  #1e-4 #10 #1e-4 #2e-5 for RSNN

config = {"image_size": image_size,
            "threshold": threshold,
            "beta": beta,
            "n_steps": n_steps,
            "n_input": n_input,
            "n_hidden": n_hidden,
            "n_hidden_RNN": n_hidden_RNN,
            #"two_Rlayers": two_Rlayers,
            "batch_size": batch_size,
            "lr": lr_k,
            "n_epoch": n_epoch,
            "loss": "nn.MSELoss()",# + LM * N_spikes
            "cross_last": cross_last,
            "lm_spikes_number": "training", #~-0.0058 , lm_spN = 1e-4
            "random_num": random_num,
           "message": message,
        }

if wandb_rec == True:
    import wandb
    wandb.login()  #14e015ed9d214a1eb3dae07b9be4c082a85ab2b4  #wandb login
    run = wandb.init(
        settings=wandb.Settings(init_timeout=320),
        project='Model Comparison on OrganA',
        name = f'{model_name}_{random_num} - Img_size={image_size}',
        job_type='train, val and test',
        config=config
    )


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")




# Define the model (copied from your code)
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
            nn.Linear(self.n_hidden, self.n_hidden),
            snn.Leaky(beta=self.beta, threshold= self.threshold, init_hidden=True, output = False, learn_threshold=learn_threshold, learn_beta=learn_beta), #, learn_threshold=learn_threshold, output = True),

            nn.Dropout(0.3),
            nn.Linear(self.n_hidden, self.n_hidden),
            snn.Leaky(beta=self.beta, threshold= self.threshold, init_hidden=True, output = False, learn_threshold=learn_threshold, learn_beta=learn_beta), #, learn_threshold=learn_threshold, output = True),

            nn.Dropout(0.3),
            nn.Linear(self.n_hidden, self.n_hidden),
            snn.Leaky(beta=self.beta, threshold= self.threshold, init_hidden=True, output = False, learn_threshold=learn_threshold, learn_beta=learn_beta), #, learn_threshold=learn_threshold, output = True),

            nn.Dropout(0.3),
            nn.Linear(self.n_hidden, self.n_hidden),
            snn.Leaky(beta=self.beta, threshold= self.threshold, init_hidden=True, output=False, learn_threshold=learn_threshold, learn_beta=learn_beta), #,learn_threshold=learn_threshold, learn_beta=learn_beta

            nn.Dropout(0.3),
            nn.Linear(self.n_hidden, self.n_hidden),
            snn.Leaky(beta=self.beta, threshold= self.threshold, init_hidden=True, output=False, learn_threshold=learn_threshold, learn_beta=learn_beta), #,learn_threshold=learn_threshold, learn_beta=learn_beta),

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



class ViT_SNN(deep_SNN_Base):
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





# Data loading and preprocessing
def load_organ_mnist(batch_size=64, num_workers=2):
    # Data transformations
    ic(image_size)
    transform = transforms.Compose([
        #transforms.Resize((128, 128)),  # Resize to match model input size
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5], std=[0.5])])
        transforms.Normalize(mean=[0], std=[1])])


    # Load datasets
    train_dataset = OrganAMNIST(split='train', transform=transform, download=True, size = image_size)
    val_dataset = OrganAMNIST(split='val', transform=transform, download=True, size = image_size)
    test_dataset = OrganAMNIST(split='test', transform=transform, download=True, size = image_size)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last=True)

    return train_loader, val_loader, test_loader

# Training function
def train(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    import time
    cumul_time = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        torch.cuda.synchronize()
        start_time_training = time.perf_counter() 
    
        inputs, targets = inputs.to(device), targets.to(device).long()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs, reg_loss = model(inputs)
        loss = criterion(outputs, targets.squeeze()) + reg_loss

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        end_time_training = time.perf_counter() 
        training_time = end_time_training - start_time_training
        cumul_time += training_time

        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets.squeeze()).sum().item()


        # Print statistics every 50 batches
        if batch_idx % 50 == 49:
            print(f'Epoch: {epoch}, Batch: {batch_idx+1}, Loss: {running_loss/50:.3f}, Acc: {100.*correct/total:.3f}%')
            running_loss = 0.0

    # Return epoch statistics
    epoch_acc = 100. * correct / total
    mean_training_time = cumul_time / len(train_loader)
    

    total_loss = running_loss/len(train_loader)
    
    return epoch_acc, total_loss, mean_training_time #kenza

# Evaluation function
def evaluate(model, data_loader, criterion, device, compute_auroc=True):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    all_targets = []
    all_outputs = []
    cumul_inf_time = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            torch.cuda.synchronize()
            start_time_inference = time.perf_counter() 
    
            inputs, targets = inputs.to(device), targets.to(device).long()
            outputs, reg_loss = model(inputs)
            loss = criterion(outputs, targets.squeeze()) + reg_loss

            torch.cuda.synchronize()
            end_time_inference = time.perf_counter() 
            inference_time = end_time_inference - start_time_inference  
            cumul_inf_time += inference_time

            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets.squeeze()).sum().item()

            if compute_auroc:
                # Store predictions and targets for AUROC calculation
                outputs_cpu = F.softmax(outputs, dim=1).cpu().numpy()
                targets_cpu = targets.cpu().numpy()

                all_outputs.append(outputs_cpu)
                all_targets.append(targets_cpu)
            

    mean_inf_time = cumul_inf_time / len(data_loader)
    # Calculate accuracy and average loss
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(data_loader)

    # Calculate AUROC if requested
    auroc = None
    if compute_auroc and len(all_targets) > 0:
        all_targets = np.concatenate(all_targets).reshape(-1)
        all_outputs = np.concatenate(all_outputs, axis=0)

        # Compute one-vs-rest AUROC for each class and average
        auroc_per_class = []
        for i in range(model.n_output):
            # One-hot encode the targets for this class
            y_true = (all_targets == i).astype(int)
            y_score = all_outputs[:, i]

            try:
                auroc_class = roc_auc_score(y_true, y_score)
                auroc_per_class.append(auroc_class)
            except ValueError:
                # This can happen if a class doesn't appear in the test set
                pass

        if auroc_per_class:
            auroc = np.mean(auroc_per_class)

        
    return accuracy, avg_loss, auroc*100 if auroc is not None else None, mean_inf_time  # Convert to percentage

# Main training and evaluation function
def train_and_evaluate_model(model, train_loader, val_loader, test_loader, num_epochs=10, lr=0.001):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Lists to store training/validation metrics
    train_accuracies = []
    val_accuracies = []
    val_losses = []
    hist_loss = []
    val_aurocs = []
    training_time_epoch = []
    inference_time_epoch = []
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # Train for one epoch
        train_acc, train_loss,mean_training_time = train(model, train_loader, optimizer, criterion, epoch+1, device)
        training_time_epoch.append(mean_training_time)

        train_accuracies.append(train_acc)
        hist_loss.append(train_loss)
        print(f"Training Accuracy: {train_acc:.2f}%")
        if wandb_rec == True:
            wandb.log({"train accuracy": train_acc,
                    #"train_auroc": train_auroc.cpu().numpy(),
                    "train_loss": train_loss,  # loss.item()
                    "training_time": mean_training_time
                    }, step=epoch)
        else:
            pass

        # Evaluate on validation set
        val_acc, val_loss, val_auroc,mean_inf_time = evaluate(model, val_loader, criterion, device)
        inference_time_epoch.append(mean_inf_time)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        val_aurocs.append(val_auroc)

        if wandb_rec == True:
            wandb.log({"val_accuracy": val_acc,
                   "val_auroc": val_auroc,
                   "inference time": mean_inf_time
                   }, step=epoch)
        else:
            pass


        print(f"Validation Accuracy: {val_acc:.2f}%, Loss: {val_loss:.4f}, Auroc: {val_auroc:.4f}")

    # Final evaluation on test set
    
    test_acc, test_loss, test_auroc,mean_inf_time = evaluate(model, test_loader, criterion, device, compute_auroc=True)

    if wandb_rec == True:
        wandb.finish()
    print(f"\nFinal Test Results:")
    print(f"Accuracy: {test_acc:.2f}%")
    print(f"Loss: {test_loss:.4f}")
    print(f"AUROC: {test_auroc:.4f}")

    # Plot training and validation accuracy
    plt.figure(figsize=(20, 6))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 3, 2)
    plt.plot(range(1, num_epochs+1), hist_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')

    plt.subplot(1, 3, 3)
    #plt.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), val_aurocs, label='Validation AUROC')
    plt.xlabel('Epochs')
    plt.ylabel('AUROC (%)')
    plt.legend()
    plt.title('Validation AUROC') #Training and


    plt.tight_layout()
    plt.savefig(f'results_{model_name}_{image_size}_{random_num}_{n_epoch}.png')
    plt.show()

    return model, train_accuracies, val_accuracies, val_aurocs, hist_loss # test_acc, test_loss, test_auroc

if __name__ == "__main__":
    seed = 888
    torch.manual_seed(seed)#162 #42

    # Hyperparameters
    BATCH_SIZE = batch_size
    NUM_EPOCHS = n_epoch
    LEARNING_RATE = 0.0001
    IMAGE_SIZE = image_size
    N_INPUT = 1  # Single channel grayscale images

    # Load data
    train_loader, val_loader, test_loader = load_organ_mnist(batch_size=BATCH_SIZE)

    # Initialize the model
    model = ViT_SNN(
        image_size=image_size,
        n_input=n_input,
        n_hidden=n_hidden,
        n_output=n_output,
        threshold=threshold,
        beta=beta, #0.95,
        n_steps=n_steps
    ).to(device)

    # Train and evaluate the model
    model, train_accuracies, val_accuracies, val_auroc, hist_loss = train_and_evaluate_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        num_epochs=n_epoch,
        lr=0.0001
    )

    # Save the trained model
    torch.save(model.state_dict(), f'{model_name}_organmnist_{image_size}_{n_epoch}_{seed}.pth')
    print("Model saved successfully!")


