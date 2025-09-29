import torch
import logging


def setup_device(gpu_index, memory_fraction=1.0):
    """Set up and return the appropriate device with memory limit."""
    # Set the device
    torch.cuda.set_device(gpu_index)
    device = torch.device(f"cuda:{gpu_index}")
    
    # Limit GPU memory usage
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=gpu_index)
    
    # Verify the device was set correctly
    current_device = torch.cuda.current_device()
    if current_device != gpu_index:
        logging.warning(f"Failed to set GPU device to {gpu_index}. Current device is {current_device}")
    
    logging.info(f"GPU current device: {current_device}")
    logging.info(f"Using device: {device}, GPU {gpu_index}")
    logging.info(f"Memory fraction limit set to: {memory_fraction*100}%")
    return device