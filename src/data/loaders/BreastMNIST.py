import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler, Subset
from src.data.sets.BreastMNIST import KSpaceBreastMNIST

def get_balanced_loader(dataset, batch_size=64, num_workers=0):

    # Get class counts (frequency)
    labels = np.array([label for _, label in dataset])  
    if labels.ndim > 1: 
        labels = labels.flatten()
    labels = torch.tensor(labels, dtype=torch.int64)  #ensure integer
    print("Labels shape:", labels.shape)
    print("Unique labels:", torch.unique(labels))
    class_counts = torch.bincount(labels)
  
    # Get class weights (inverse of class frequency)
    class_weights = 1.0 / class_counts.float()

    # Create weights for each images
    sampler_weights = [class_weights[label] for _, label in dataset]

    # Create sampler
    sampler = WeightedRandomSampler(weights=sampler_weights, num_samples=len(dataset), replacement=True)

    # Create DataLoader
    balanced_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    return balanced_loader

class BreastMNISTDataloader(object):
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 image_size: int = 128,
                 debug: bool = True):

        super(BreastMNISTDataloader, self).__init__()
        self.data_dir = data_dir
        self.debug = debug
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def train(self):
        train_kspace = KSpaceBreastMNIST(self.data_dir, self.image_size, split = "train")

        if self.debug:
            train_kspace = Subset(train_kspace, range(self.batch_size *2))
        
        return get_balanced_loader(train_kspace, batch_size=self.batch_size)

    def val(self):
        val_kspace = KSpaceBreastMNIST(self.data_dir, self.image_size, split = "val")
        if self.debug:
            val_kspace = Subset(val_kspace, range(self.batch_size *2))
        
        return get_balanced_loader(val_kspace, batch_size=self.batch_size)
    
    def test(self):
        test_kspace = KSpaceBreastMNIST(self.data_dir, self.image_size, split = "test")
        if self.debug:
            test_kspace = Subset(test_kspace, range(self.batch_size *2))
        
        return get_balanced_loader(test_kspace, batch_size=self.batch_size)
    
