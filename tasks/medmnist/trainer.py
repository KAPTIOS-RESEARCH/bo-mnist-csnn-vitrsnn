import torch
from torch import nn
from tqdm import tqdm
from src.core.trainer import BaseTrainer
from spikingjelly.activation_based import functional


class medmnistTrainer(BaseTrainer):

    def __init__(self, model: nn.Module, parameters: dict, device: str):
        super(medmnistTrainer, self).__init__(model, parameters, device)
        if not self.criterion:
            self.criterion = nn.MSELoss()

    def train(self, train_loader):
        self.model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, leave=False, desc="Running training phase") as pbar:
            for sample in train_loader:
                data, targets = sample
                data, targets = data.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                functional.reset_net(self.model)
                pbar.update(1)

        train_loss /= len(train_loader)
        return train_loss

    def test(self, val_loader):
        self.model.eval()
        test_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            with tqdm(val_loader, leave=False, desc="Running testing phase") as pbar:
                for idx, sample in enumerate(val_loader):
                    data, targets = sample
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                    test_loss += loss.item()
                    
                    all_preds.append(outputs)
                    all_targets.append(targets.cpu())
                    functional.reset_net(self.model)
                    pbar.update(1)

        all_targets = torch.cat(all_targets)
        all_preds = torch.cat(all_preds)
        test_loss /= len(val_loader)
        return test_loss, all_preds, all_targets
