from medmnist import OrganAMNIST
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from icecream import ic
import torch, cv2, os, logging

image_size = 128

Transforms_base = transforms.Compose([
            #transforms.Resize((image_size, image_size)),
            #transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0], std=[1])])

class BaseOrganAMNIST(Dataset):
    def __init__(self, data_dir, image_size: int = 128, split = 'train', transform = False):
        os.makedirs(data_dir, exist_ok = True)
        self.image_size = image_size
        if transform:
            self.transform = transform
        else:
            self.transform =transforms.Compose([
                #transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0], std=[1])]
                #transforms.Normalize(mean=[0.5], std=[0.5])]
                #transforms.Normalize((0,), (1,))]
            )
        
        logging.info(self.image_size)
        self.b_mnist = OrganAMNIST(split=split,root=data_dir, download=True, transform=self.transform, size=self.image_size, as_rgb = False) 

    def __len__(self):
        return len(self.b_mnist)

    def __getitem__(self, idx):
        image, label = self.b_mnist[idx]
        return image, label
        



"""class OrganA_MNIST(BaseOrganAMNIST):
    def __init__(self, data_dir: str, image_size: int = 128, split = 'train', transform=None):
        super().__init__(data_dir, image_size, split, transform)
        
    def __getitem__(self, idx):
        image, label = self.b_mnist[idx]
        image_np = image.squeeze().numpy()
        image_tensor = torch.tensor(image_np, dtype=torch.float32)

        image_tensor = image_tensor.unsqueeze(0)
        label = label.squeeze(0).astype('float32')

        #ic(label.shape)
        return image_tensor, label#.squeeze()"""

        