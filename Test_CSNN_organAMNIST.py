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

image_size = 224#224 #128
n_steps = 5
batch_size = 64

### SimpleSNN patching
wandb_rec = False
random_num = random.randint(0, 500)
message ="CSNN on OrganAMNIST" #L2 reg  #prunning #"add_transformation_noNoise"
n_hidden = 1000
model_name = "CSNN"
n_input = image_size * image_size # 128 * 128 #224 * 224 # 64 x64
n_hidden_RNN = 100
beta = .95
threshold = 0.5
#two_Rlayers = False
lr_k = 1e-5 # 1e-4
n_epoch = 50 #70#00 #00
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

print(config)
if wandb_rec == True:
    import wandb
    wandb.login()  #14e015ed9d214a1eb3dae07b9be4c082a85ab2b4  #wandb login
    run = wandb.init(
        settings=wandb.Settings(init_timeout=320),
        project='Model Comparison on OrganA',
        name = f'CSNN_{random_num} - Img_size={image_size}',
        job_type='train, val and test',
        config=config
    )

ic(random_num)


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")




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
    """nn.Conv2d(32, 64, kernel_size=5, padding=1), #
            nn.MaxPool2d(2),
            snn.Leaky(beta=self.beta, threshold=self.threshold, init_hidden=True),

    """
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



# Data loading and preprocessing
def load_organ_mnist(image_size, batch_size=64, num_workers=2):
    # Data transformations
    transform = transforms.Compose([
        #transforms.Resize((128, 128)),  # Resize to match model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])])
#        transforms.Normalize(mean=[0.5], std=[0.5])
#   ])

    # Load datasets
    train_dataset = OrganAMNIST(split='train', transform=transform, download=True, size = image_size)
    val_dataset = OrganAMNIST(split='val', transform=transform, download=True,size = image_size)
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
    torch.cuda.synchronize() #to synchronize the GPU before starting the timer
    cumul_time = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
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
        cumul_time += end_time_training - start_time_training

        ic(outputs.shape, targets.shape)

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

    #torch.cuda.synchronize()  #to synchronize the GPU before stopping the timer
    #end_time_training = time.perf_counter() 
    #training_time = end_time_training - start_time_training
       
    total_loss = running_loss/len(train_loader)
    
    return epoch_acc, total_loss,mean_training_time #kenza

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

    # Calculate accuracy and average loss
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(data_loader)

    mean_inf_time = cumul_inf_time / len(data_loader)
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

    return accuracy, avg_loss, auroc*100, mean_inf_time

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
        print("Training time:  ", mean_training_time)
        print(f"Training Accuracy: {train_acc:.2f}%")
        if wandb_rec == True:
            wandb.log({"train accuracy": train_acc,
                    #"train_auroc": train_auroc.cpu().numpy(),
                    "train_loss": train_loss,  # loss.item()
                    "training_time": mean_training_time}, step=epoch)
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
        print("Inference time:  ", mean_inf_time)
        print(f"Validation Accuracy: {val_acc:.2f}%, Loss: {val_loss:.4f}, Auroc: {val_auroc:.4f}")

     
    # Final evaluation on test set
    test_acc, test_loss, test_auroc,mean_inf_time_2 = evaluate(model, test_loader, criterion, device, compute_auroc=True)

    mean_epoch_inf_time = sum(inference_time_epoch)/ len(inference_time_epoch)
    mean_epoch_train_time = sum(training_time_epoch)/ len(training_time_epoch)
    if wandb_rec == True:
        wandb.finish()
    print(f"\nFinal Test Results:")
    print(f"Accuracy: {test_acc:.2f}%")
    print(f"Loss: {test_loss:.4f}")
    print(f"AUROC: {test_auroc:.4f}")
    print(f"Average inf time:{mean_epoch_inf_time:.4f}")
    print(f"Average train time:{mean_epoch_train_time:.4f}")

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

# Run the main function
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Hyperparameters
    BATCH_SIZE = 64
    NUM_EPOCHS = n_epoch 
    LEARNING_RATE = 0.001
    IMAGE_SIZE = image_size #128
    N_INPUT = 1  # Single channel grayscale images

    # Load data
    train_loader, val_loader, test_loader = load_organ_mnist(batch_size=BATCH_SIZE, image_size = image_size)

    # Initialize the model
    model = CSNN_bis_reg(
        image_size=image_size,
        n_input=N_INPUT,
        n_hidden=512,
        n_output=11,
        threshold=0.5,
        beta=0.95,
        n_steps=5
    ).to(device)

    # Train and evaluate the model
    model, train_accuracies, val_accuracies, val_auroc, hist_loss = train_and_evaluate_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE
    )

    # Save the trained model
    torch.save(model.state_dict(), f'{model_name}_organmnist_{image_size}_{n_epoch}.pth')
    print("Model saved successfully!")