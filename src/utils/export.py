import torch, os, logging
from .quantization import fuse_layers

def save_to_onnx(path: str, model: torch.nn.Module, tensor_x: torch.Tensor):
    """Exports a PyTorch model to ONNX format.

    Args:
        path (str): The directory were the .onnx file will be saved
        model (torch.nn.Module): The model to export
        tensor_x (torch.Tensor): A data sample used to train the model
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok = True)
    
    model = fuse_layers(model)
    onnx_program = torch.onnx.export(
        model, 
        (tensor_x,), 
        dynamo=True,
        export_params=True, 
        opset_version=18,          
        input_names = ['input'],                          
        output_names = ['output'],
    )
    onnx_program.save(path)
    logging.info(f'Model successfully exported to ONNX format in {path}')