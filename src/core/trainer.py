import wandb, logging
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from src.utils.config import instanciate_module
from src.optimisation.early_stopping import EarlyStopping
import matplotlib.pyplot as plt 
import random
import time

def plot_training(lr,dataset_name, acc_hist_train, acc_hist_test, loss_hist): #auroc_hist_train, auroc_hist_test,total_time
  #Plot Acc Ksâce
  #lr = 1e-4
  random_num = random.randint(1, 500)

  fig = plt.figure(facecolor="w")
  plt.plot(acc_hist_train, label = "train accuracy")
  plt.plot(acc_hist_test, label = "test accuracy")
  plt.title(f"Test and training set Accuracy for {dataset_name} images")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.show()
  plt.savefig(f"./data/SNN5_Accuracy_{random_num}.png", format='png')
    

  """# Plot AUROC
  fig = plt.figure(facecolor="w")
  plt.plot(auroc_hist_train, label = "train AUROC")
  plt.plot(auroc_hist_test, label = "test AUROC")
  plt.title(f"Test and training set AUROC for {dataset_name} images")
  plt.xlabel("Epoch")
  plt.ylabel("AUROC")
  plt.legend()
  plt.show()"""

  #Plot Loss
  fig = plt.figure(facecolor="w")
  plt.plot(loss_hist, label = f"lr = {lr},") #, simulation time = {total_time}
  plt.title("Train set Loss")
  plt.xlabel("Iteration")
  plt.ylabel(f"Loss for Original {dataset_name} images")
  plt.legend()
  
  #plt.savefig(f"./data/CSNN_ViTRSNN_Loss_{random_num}.png", format='png')
  plt.savefig(f"./data/SNN5_Loss_{random_num}.png", format='png')
    
  plt.show()


class BaseTrainer(object):

    def __init__(self, model: nn.Module, parameters: dict, device: str):
        self.model = model
        self.parameters = parameters
        self.device = device
        self.early_stop = None #EarlyStopping(patience=parameters['early_stopping_patience'], enable_wandb=parameters['track'])
        
        # OPTIMIZER
        self.optimizer = Adam(
            self.model.parameters(), 
            lr=parameters['lr'],
            weight_decay=parameters['weight_decay']
        )
   
        #self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size= parameters['step_size_optim'], gamma=parameters['gamma_scheduler'])
        
        # LR SCHEDULER
        self.lr_scheduler = None
        """lr_scheduler_type = parameters['lr_scheduler'] if 'lr_scheduler' in parameters.keys() else 'none'

        if lr_scheduler_type == 'cosine':
            self.lr_scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=100)
        elif lr_scheduler_type == 'plateau':
            self.lr_scheduler = ReduceLROnPlateau(optimizer=self.optimizer, mode='min', factor=0.1)
        elif lr_scheduler_type == 'exponential':
            self.lr_scheduler = ExponentialLR(optimizer=self.optimizer, gamma=parameters['gamma_scheduler'])
        """
        
        # LOSS FUNCTION
        self.criterion = instanciate_module(parameters['loss']['module_name'],
                                   parameters['loss']['class_name'], 
                                   parameters['loss']['parameters'])
        
    def train(self, dl: DataLoader):
        raise NotImplementedError
    
    def test(self, dl: DataLoader):
        raise NotImplementedError
    
    def fit(self, train_dl, test_dl, log_dir: str):
        num_epochs = self.parameters['num_epochs']
        train_acc_hist = []
        test_acc_hist = []
        train_loss_hist = []
        start_time = time.time()
        for epoch in range(num_epochs):
            print(f"Epoch:{epoch}/{num_epochs}")
            train_loss, train_preds, train_targets, train_acc, _, _, training_time = self.train(train_dl)
            test_loss, test_preds, test_targets, test_acc, mean_inf_time = self.test(test_dl)
            train_acc_hist.append(train_acc)
            train_loss_hist.append(train_loss)
            test_acc_hist.append(test_acc)
            # COMPUTE METRICS HERE
            
            if self.parameters['track']:
                wandb.log({
                    f"Train/{self.parameters['loss']['class_name']}": train_loss,
                    #f"Test/{self.parameters['loss']['class_name']}": test_loss,
                    ## ADD METRIC WANDB PLOT HERE
                    f"train accuracy": train_acc,
                    f"val_accuracy": test_acc,
                    f"_step_": epoch,
                    f"training_time": training_time,
                    f"inference time": mean_inf_time,
                    f"train_loss": train_loss,
                })
                
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(test_loss)

            if self.early_stop is not None:
                self.early_stop(self.model, test_loss, log_dir, epoch)

                logging.info(f"Epoch {epoch + 1} / {num_epochs} - {self.parameters['loss']['class_name']}: {train_loss:.4f} - Train/Test: {train_acc} | {test_acc}")
                #| {test_loss:.4f} Train/Test
                if self.early_stop.stop:
                    logging.info(
                        f"Val loss did not improve for {self.early_stop.patience} epochs.")
                    logging.info('Training stopped by early stopping mecanism.')
                    break
         
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n✅ Training completed in {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

        dataset_name = "OrganAMNIST"
        #print(len(train_acc_hist))
        plot_training(self.parameters['lr'],dataset_name, train_acc_hist, test_acc_hist, train_loss_hist) #total_time
        
        if self.parameters['track']:
            wandb.finish()