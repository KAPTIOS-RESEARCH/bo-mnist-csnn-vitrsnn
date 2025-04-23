import torch

def get_available_device():
    """
    Select the best available device for PyTorch computation.

    Priority order:
    - CUDA (GPU) if available
    - MPS (Apple Silicon) if available
    - CPU if no GPU or MPS is available
    
    Returns:
        device (torch.device): The selected PyTorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
