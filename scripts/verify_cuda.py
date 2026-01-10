import torch
import sys

def verify_setup():
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available.")
        return

    print(f"CUDA Version: {torch.version.cuda}")
    device_count = torch.cuda.device_count()
    print(f"GPU Count: {device_count}")
    
    # Check for RTX 5090 and Compute Capability
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU 0 Name: {device_name}")
    
    major, minor = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: {major}.{minor}")
    
    # Strict Blackwell Check (sm_120 assumed as 12.0)
    if major < 12:
        print("❌ CRITICAL ERROR: Detected Architecture is older than Blackwell (sm_120).")
        print(f"   Required: 12.0+ | Found: {major}.{minor}")
        raise RuntimeError("Incompatible Architecture. Training Aborted.")
    
    if "5090" not in device_name and "Blackwell" not in device_name:
        print("⚠️  Warning: Detected GPU does not appear to be an RTX 5090.")

    # Tensor Operation Test
    try:
        print("🧪 Running Tenosr Op on CUDA...")
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x)
        print("✅ Matrix Multiplication Successful.")
    except Exception as e:
        print(f"❌ Tensor Operation Failed: {e}")

if __name__ == "__main__":
    verify_setup()
