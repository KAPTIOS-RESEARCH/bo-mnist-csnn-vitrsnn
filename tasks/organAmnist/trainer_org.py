import torch
from torch import nn
from tqdm import tqdm
from src.core.trainer import BaseTrainer
from spikingjelly.activation_based import functional
from torchmetrics.classification import BinaryAccuracy


class OrgTrainer(BaseTrainer):


    def __init__(self, model: nn.Module, parameters: dict, device: str):
        super(OrgTrainer, self).__init__(model, parameters, device)
        if not self.criterion:
            self.criterion = nn.CrossEntropyLoss()   
        
        self.accuracy_metric = BinaryAccuracy().to(device)
        self.LM_spikes = 0.0001 #1e-4

    def train(self, train_loader):
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        train_acc_hist = []
        train_loss_hist = []
        acc = 0
        torch.cuda.synchronize()
        import time
        from icecream import ic
        cumul_inf_time = 0
        
        with tqdm(train_loader, leave=False, desc="Running training phase") as pbar:
            for batch_idx, (inputs, targets) in enumerate(train_loader): #for sample in train_loader:
                #ic(inputs.shape, targets.shape) 
                start_time_inference = time.perf_counter() 
    

                #data, targets = sample
                data, targets = inputs.to(self.device), targets.to(self.device)#.long()
                self.optimizer.zero_grad()
                outputs, sum_spikes  = self.model(data)
                #ic(outputs.shape, targets.shape) 
                
                #print("number of spikes:", int(sum_spikes))
                l2_lambda = 1e-5
                l2_loss = sum((param ** 2).sum() for param in self.model.parameters() if param.requires_grad)
                loss = self.criterion(outputs, targets.squeeze()) + sum_spikes #* self.LM_spikes #+ l2_lambda * l2_loss
                torch.cuda.synchronize()
                end_time_inference = time.perf_counter() 
                inference_time = end_time_inference - start_time_inference  
                cumul_inf_time += inference_time


                #prediction = torch.sigmoid(outputs)
                #acc += self.accuracy_metric(prediction, targets).item() * 100


                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                all_preds.append(outputs)
                all_targets.append(targets.cpu())
                #functional.reset_net(self.model)
                pbar.update(1)
                #64 sould be [64, 11]) and target 64, 1] and 
                #ic(outputs.shape, targets.shape)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets.squeeze()).sum().item()




                """if compute_auroc:
                # Store predictions and targets for AUROC calculation
                outputs_cpu = F.softmax(outputs, dim=1).cpu().numpy()
                targets_cpu = targets.cpu().numpy()

                all_outputs.append(outputs_cpu)
                all_targets.append(targets_cpu)"""


        mean_inf_time = cumul_inf_time / len(train_loader)
        # Calculate accuracy and average loss
        train_acc =  torch.tensor(100. * (correct / total)) #100. *
        #train_acc = torch.tensor(acc/len(train_loader))
        train_loss /= len(train_loader)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        print("train acc", train_acc, "-- loss:",train_loss) 
        print("training time", mean_inf_time) 
        return train_loss, all_preds, all_targets, train_acc, train_acc_hist, train_loss_hist, mean_inf_time

    def test(self, val_loader):
        self.model.eval()
        test_loss = 0.0
        all_preds = []
        all_targets = []
        test_acc_hist = []
        acc_test =0
        total = 0
        correct = 0
        import time
        cumul_inf_time = 0
        with torch.no_grad():
            with tqdm(val_loader, leave=False, desc="Running testing phase") as pbar:
                for idx, sample in enumerate(val_loader):
                    torch.cuda.synchronize()
                    start_time_inference = time.perf_counter()

                    data, targets = sample
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs, sum_spikes  = self.model(data)
                    loss = self.criterion(outputs, targets.squeeze()) + sum_spikes #* self.LM_spikes

                    torch.cuda.synchronize()
                    end_time_inference = time.perf_counter() 
                    inference_time = end_time_inference - start_time_inference  
                    cumul_inf_time += inference_time


                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets.squeeze()).sum().item()

                    #prediction = torch.sigmoid(outputs)
                    #acc_test += self.accuracy_metric(prediction, targets).item() * 100


                    
                    all_preds.append(outputs)
                    all_targets.append(targets.cpu())
                    #functional.reset_net(self.model)
                    pbar.update(1)

        all_targets = torch.cat(all_targets)
        all_preds = torch.cat(all_preds)
        test_loss /= len(val_loader)
        
        #test_acc = torch.tensor(acc_test/len(val_loader))
        # Calculate accuracy and average loss
        test_acc = torch.tensor(100. * correct / total)
        test_acc_hist.append(test_acc)
        mean_inf_time = cumul_inf_time / len(val_loader)
        print("-- test acc: ", test_acc)
        print("-- inference time: ", mean_inf_time)
        
        return test_loss, all_preds, all_targets, test_acc,mean_inf_time
#test_loss, test_preds, test_targets, test_acc, _


