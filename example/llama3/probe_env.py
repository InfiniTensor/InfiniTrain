import torch
import sys
import os
import subprocess

def get_cuda_version():
    try:
        return torch.version.cuda
    except:
        return "N/A"

def get_flash_attn_version():
    try:
        import flash_attn
        return flash_attn.__version__
    except ImportError:
        return "Not Installed"

def get_gpu_info():
    if not torch.cuda.is_available():
        return "No GPU Available"
    
    info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info.append(f"Device {i}: {props.name}, Compute {props.major}.{props.minor}, Memory {props.total_memory / 1024**3:.2f} GB")
    return "\n".join(info)

def get_libcudart_version():
    try:
        # Check specific path or ldconfig?
        # Just return dummy for now or implement proper check
        return "Unknown"
    except:
        return "Error"

print("InfiniTrain Environment Probe")
print("-" * 30)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA (PyTorch): {get_cuda_version()}")
print(f"FlashAttention (Python): {get_flash_attn_version()}")
print("-" * 30)
print("GPU Info:")
print(get_gpu_info())
print("-" * 30)

# Write to env_snapshot.json
import json
snapshot = {
    "pytorch": torch.__version__,
    "cuda": get_cuda_version(),
    "flash_attn": get_flash_attn_version(),
    "gpu_info": get_gpu_info(),
    "env_vars": dict(os.environ)
}

with open("env_snapshot.json", "w") as f:
    json.dump(snapshot, f, indent=4)

print("Snapshot saved to env_snapshot.json")
