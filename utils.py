import torch
import numpy as np
import warnings
from datetime import datetime


def get_formatted_time():
    now = datetime.now()
    month = str(now.month).zfill(2)  # Pad single-digit months with leading zero
    day = str(now.day).zfill(2)      # Pad single-digit days with leading zero
    hour = str(now.hour).zfill(2)    # Pad single-digit hours with leading zero
    minute = str(now.minute).zfill(2) # Pad single-digit minutes with leading zero

    formatted_time = f"{month}{day}{hour}{minute}"
    return formatted_time

def save_model(model: torch.nn.Module,
               model_file_path: str):
    """Save a PyTorch model.

    Args:
        model (torch.nn.Module): A PyTorch model.
        model_file_path (str): Target model file path.
    """
    
    model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.save(model, model_file_path)


def get_module(model: torch.nn.Module, module_name: str):
    """Get a module from a PyTorch model.
    
    Example:
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        >>> get_module(model, 'layer1.0')
        BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    Args:
        model (torch.nn.Module): A PyTorch model.
        module_name (str): Module name.

    Returns:
        torch.nn.Module: Corrsponding module.
    """
    for name, module in model.named_modules():
        if name == module_name:
            return module

    return None


def get_super_module(model: torch.nn.Module, module_name: str):
    """Get the super module of a module in a PyTorch model.
    
    Example:
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        >>> get_super_module(model, 'layer1.0.conv1')
        BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    Args:
        model (torch.nn.Module): A PyTorch model.
        module_name (str): Module name.

    Returns:
        torch.nn.Module: Super module of module :attr:`module_name`.
    """
    super_module_name = '.'.join(module_name.split('.')[0:-1])
    return get_module(model, super_module_name)


def set_module(model: torch.nn.Module, module_name: str, module: torch.nn.Module):
    """Set module in a PyTorch model.
    
    Example:
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        >>> set_module(model, 'layer1.0', torch.nn.Conv2d(64, 64, 3))
        >>> model
        ResNet(
            (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            (layer1): Sequential(
            --> (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (1): BasicBlock(
                    ...
                )
                ...
            )
            ...
        )

    Args:
        model (torch.nn.Module): A PyTorch model.
        module_name (str): Module name.
        module (torch.nn.Module): Target module which will be set into :attr:`model`.
    """
    super_module = get_super_module(model, module_name)
    setattr(super_module, module_name.split('.')[-1], module)

def freeze_module(model: torch.nn.Module, module_name: any):
    """Freeze module in a PyTorch model.
    """
    if isinstance(module_name, str):
        module = get_module(model, module_name)
        for param in module.parameters():
            param.requires_grad = False
    elif isinstance(module_name, list):
        for name in module_name:
            freeze_module(model, name)