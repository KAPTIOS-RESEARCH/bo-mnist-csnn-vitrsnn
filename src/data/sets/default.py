import torch, torchvision

class DefaultDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        transform = None,
    ):
        self.root = root
        self.samples = self._load_samples()
        
        default_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        
        if not transform:
            self.transform = default_transform()
        else:
            self.transform = transform

    def _load_samples(self):
        return []
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]