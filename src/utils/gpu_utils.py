"""GPU detection and configuration utilities for multi-vendor support.

Supports:
- NVIDIA GPUs (CUDA)
- Intel Arc GPUs (XPU)
- AMD GPUs (ROCm)
- CPU fallback
"""

import logging
import os
import subprocess
import sys
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def detect_gpu_vendor() -> str:
    """
    Detect the primary GPU vendor.
    
    Returns:
        str: One of 'nvidia', 'intel', 'amd', or 'cpu'
    """
    try:
        # First, try PyTorch detection (most reliable if already installed)
        if torch.cuda.is_available():
            return "nvidia"

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "intel"

        if hasattr(torch, "hip") and torch.hip.is_available():
            return "amd"

        # Fallback: check via lspci (works even without PyTorch GPU support)
        result = subprocess.run(
            ["lspci", "-nn"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        stdout = result.stdout.upper()
        
        if "NVIDIA" in stdout:
            return "nvidia"
        elif "AMD" in stdout and "RADEON" in stdout:
            return "amd"
        elif "INTEL" in stdout and ("ARC" in stdout or "XE" in stdout):
            return "intel"

    except Exception as e:
        logger.warning(f"GPU detection failed: {e}")

    return "cpu"


def get_device() -> torch.device:
    """
    Get the appropriate torch device based on available hardware.
    
    Returns:
        torch.device: The device to use for computation
    """
    vendor = detect_gpu_vendor()

    if vendor == "nvidia" and torch.cuda.is_available():
        return torch.device("cuda")
    elif vendor == "intel" and hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif vendor == "amd" and hasattr(torch, "hip"):
        return torch.device("hip")
    else:
        logger.info("No GPU detected, using CPU")
        return torch.device("cpu")


def verify_intel_xpu() -> bool:
    """
    Verify Intel XPU installation is working correctly.
    Runs the official Intel sanity test.
    
    Returns:
        bool: True if XPU is properly configured
    """
    try:
        import intel_extension_for_pytorch as ipex
        
        print("\n🔍 Intel XPU Verification:")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   IPEX version: {ipex.__version__}")
        print(f"   XPU available: {torch.xpu.is_available()}")
        print(f"   XPU device count: {torch.xpu.device_count()}")
        
        if torch.xpu.device_count() > 0:
            print("\n   Detected XPU devices:")
            for i in range(torch.xpu.device_count()):
                props = torch.xpu.get_device_properties(i)
                print(f"   [{i}]: {props}")
            return True
        else:
            print("   ⚠️  No XPU devices found")
            return False
            
    except ImportError:
        print("   ❌ intel_extension_for_pytorch not installed")
        return False
    except Exception as e:
        print(f"   ❌ XPU verification failed: {e}")
        return False


def display_gpu_info():
    """Display comprehensive GPU information."""
    vendor = detect_gpu_vendor()
    device = get_device()

    print("\n🔍 GPU Information:")
    print(f"   Vendor: {vendor.upper()}")
    print(f"   Device: {device}")

    if vendor == "nvidia":
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")

    elif vendor == "intel":
        verify_intel_xpu()

    elif vendor == "amd":
        if hasattr(torch, "hip"):
            print(f"   ROCm/HIP Available: {torch.hip.is_available()}")
            if torch.hip.is_available():
                print(f"   ROCm Version: {torch.version.hip}")
                print(f"   GPU Count: {torch.cuda.device_count()}")  # AMD uses CUDA API
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    print(f"   GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    else:
        print("   Using CPU for computation")


def detect_and_configure_gpu() -> Tuple[str, torch.device]:
    """
    Auto-detect and configure GPU settings.
    Sets environment variables for other tools.
    
    Returns:
        Tuple[str, torch.device]: (vendor, device)
    """
    vendor = detect_gpu_vendor()
    device = get_device()

    if vendor == "nvidia":
        os.environ["GPU_TYPE"] = "nvidia"
        os.environ["TORCH_DEVICE"] = "cuda"
    elif vendor == "intel":
        os.environ["GPU_TYPE"] = "intel-arc"
        os.environ["TORCH_DEVICE"] = "xpu"
    elif vendor == "amd":
        os.environ["GPU_TYPE"] = "amd-mi50"
        os.environ["TORCH_DEVICE"] = "hip"
    else:
        os.environ["GPU_TYPE"] = "cpu"
        os.environ["TORCH_DEVICE"] = "cpu"

    print(f"✓ Configured for {vendor.upper()} GPU")
    return vendor, device


def setup_torch_device(device: torch.device = None) -> torch.device:
    """
    Setup PyTorch to use the specified device.
    Applies vendor-specific optimizations.
    
    Args:
        device: Specific device to use, or None for auto-detection
        
    Returns:
        torch.device: The configured device
    """
    if device is None:
        device = get_device()
    
    vendor = detect_gpu_vendor()
    
    # Intel XPU specific optimizations
    if vendor == "intel" and device.type == "xpu":
        try:
            import intel_extension_for_pytorch as ipex
            # Enable XPU optimizations
            torch.xpu.set_device(0)  # Use first XPU device
            logger.info("Intel XPU optimizations enabled")
        except ImportError:
            logger.warning("intel_extension_for_pytorch not available")
    
    # NVIDIA CUDA specific optimizations
    elif vendor == "nvidia" and device.type == "cuda":
        # Enable TF32 for faster training on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("CUDA optimizations enabled")
    
    return device


# Convenience function for scripts
def get_optimal_device() -> torch.device:
    """
    Get and configure the optimal device for the current system.
    This is the main function to use in training scripts.
    
    Returns:
        torch.device: Configured device ready for use
    
    Example:
        >>> device = get_optimal_device()
        >>> model = model.to(device)
    """
    vendor, device = detect_and_configure_gpu()
    return setup_torch_device(device)


if __name__ == "__main__":
    # Run diagnostics when executed directly
    print("=" * 60)
    print("GPU DETECTION AND CONFIGURATION")
    print("=" * 60)
    
    display_gpu_info()
    
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    
    vendor, device = detect_and_configure_gpu()
    optimal_device = get_optimal_device()
    
    print(f"\nConfigured device: {optimal_device}")
    print(f"Environment variables set:")
    print(f"  GPU_TYPE={os.environ.get('GPU_TYPE')}")
    print(f"  TORCH_DEVICE={os.environ.get('TORCH_DEVICE')}")
    
    # Run vendor-specific verification
    if vendor == "intel":
        print("\n" + "=" * 60)
        print("INTEL XPU SANITY TEST")
        print("=" * 60)
        verify_intel_xpu()
