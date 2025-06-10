from medmnist import BreastMNIST
from torch.utils.data import Dataset
import torch, cv2, os
import numpy as np
import torchvision.transforms as transforms

class BaseBreastMNIST(Dataset):
    def __init__(self, data_dir, image_size: int = 128, split = 'train', transform = None):
        os.makedirs(data_dir, exist_ok = True)
        self.image_size = image_size
        if transform:
            self.transform = transform
        else:
            self.transform =transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))]
            )
        self.b_mnist = BreastMNIST(split=split,root=data_dir, download=True, transform=self.transform, size=self.image_size)

    def __len__(self):
        return len(self.b_mnist)

    def __getitem__(self):
        pass


class OriginalBreastMNIST(BaseBreastMNIST):
    def __init__(self, data_dir, image_size: int = 128, split = 'train', transform=None):
        super().__init__(data_dir, image_size, split, transform)

    def __getitem__(self, idx):
        image, label = self.b_mnist[idx]
        image_np = image.squeeze().numpy()
        image_tensor = torch.tensor(image_np, dtype=torch.float32)

        image_tensor = image_tensor.unsqueeze(0)
        label = label.squeeze(0).astype('float32')
        #print("image tensor shape ",image_tensor.shape, label.shape)
        return image_tensor, label

class KSpaceBreastMNIST(BaseBreastMNIST):
    def __init__(self, data_dir, image_size: int = 128, split = 'train', transform=None):
        super().__init__(data_dir, image_size, split, transform)


    def __getitem__(self, idx):
        image, label = self.b_mnist[idx]
        image_np = image.squeeze().numpy()
        kspace_image = np.fft.fftshift(np.fft.fft2(image_np))
        kspace_abs = np.abs(kspace_image)
        kspace_tensor = cv2.resize(kspace_abs, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        kspace_tensor = torch.tensor(kspace_tensor, dtype=torch.float32)
        kspace_tensor = kspace_tensor.unsqueeze(0)
        label = label.squeeze(0).astype('float32')
        
        return kspace_tensor, label 

