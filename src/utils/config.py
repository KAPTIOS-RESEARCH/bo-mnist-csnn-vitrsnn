import torch, random, importlib
import numpy as np
import yaml, logging

def validate_config_file(params: dict):
    """Validate the config .yaml file
    Args:
        params (dict): Load yaml parameter file
    """
    file_keys = list(params.keys())
    base_message = "validate_config_file (AssertionError)]: "
    param_groups = ["model", "trainer", "dataloader", "track", "name", "seed"]

    if type(params) is not dict:
        raise AssertionError(base_message + "params should be a dictionary")

    for mdx, p_group in enumerate(param_groups):
        if mdx > 2:
            if p_group not in file_keys:
                params[p_group] = None
            continue
        if p_group not in file_keys:
            raise AssertionError(
                base_message + 'Parameter file should contain keyword "' + p_group + '"!')

        required_keywords = ['class_name', 'module_name', 'parameters']
        for i in required_keywords:
            if i not in params[p_group].keys():
                raise AssertionError(
                    base_message
                    + f'{p_group} in parameter file should contain keyword "'
                    + p_group
                    + f'": {i}!'
                )
            
    assert 'seed' in file_keys, \
        base_message + 'Config file should contain seed'

    assert 'name' in file_keys, \
        base_message + 'Config file should contain name'
    
    return params



def validate_export_config_file(params: dict):
    """Validate the export config .yaml file
    Args:
        params (dict): Load yaml parameter file
    """
    file_keys = list(params.keys())
    base_message = "validate_config_file (AssertionError)]: "
    param_groups = ["model", "quantization_dataset"]
    param_keys = ["export_path", "model_path"]

    if type(params) is not dict:
        raise AssertionError(base_message + "params should be a dictionary")

    for mdx, p_group in enumerate(param_groups):
        if mdx > 2:
            if p_group not in file_keys:
                params[p_group] = None
            continue
        if p_group not in file_keys:
            raise AssertionError(
                base_message + 'Parameter file should contain keyword "' + p_group + '"!')

        required_keywords = ['class_name', 'module_name', 'parameters']
        for i in required_keywords:
            if i not in params[p_group].keys():
                raise AssertionError(
                    base_message
                    + f'{p_group} in parameter file should contain keyword "'
                    + p_group
                    + f'": {i}!'
                )
    
    for key in param_keys:
        assert key in file_keys, \
            base_message + f'Config file should contain {key}'
    
    return params

def load_config_file(path: str):
    """Load a yaml config file

    Args:
        path (str): Path to the yaml file.

    Returns:
        dict: The loaded parameter dictionary
    """
    try:
        stream_file = open(path, "r")
        parameters = yaml.load(stream_file, Loader=yaml.FullLoader)
        validate_config_file(parameters)
        logging.info("Success: Loaded parameter file at: {}".format(path))
    except AssertionError as e:
        logging.error(e)
        exit()
    return parameters

def load_export_config_file(path: str):
    """Load a yaml config file

    Args:
        path (str): Path to the yaml file.

    Returns:
        dict: The loaded parameter dictionary
    """
    try:
        stream_file = open(path, "r")
        parameters = yaml.load(stream_file, Loader=yaml.FullLoader)
        validate_export_config_file(parameters)
        logging.info("Success: Loaded export config file at: {}".format(path))
    except AssertionError as e:
        logging.error(e)
        exit()
    return parameters


def set_seed(seed):
    """Set the seed for experiment reproduction

    Args:
        seed (int): The seed to set
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def instanciate_module(module_name: str, class_name: str, params: dict = None):
    """Instantiate a module class

    Args:
        module_name (str): The name of the module holding the class
        class_name (str): The name of the class to instanciate
        params (dict): The parameters dictionary to feed to the class

    Raises:
        AttributeError: Wrong parameters
        ModuleNotFoundError: Module does not exist
    """
    try:
        module_ = importlib.import_module(module_name)
        if not hasattr(module_, class_name):
            raise AttributeError(
                f"Class '{class_name}' not found in module '{module_name}'")

        class_ = getattr(module_, class_name)
        if params:
            mod = class_(**params)
        else:
            mod = class_()
        return mod
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Module '{module_name}' not found")
    except AttributeError as e:
        raise e
    except Exception as e:
        raise Exception(f"An error occurred: {e}")
