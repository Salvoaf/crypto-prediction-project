"""
GPU Compatibility Check Script

This script performs a comprehensive check of GPU availability and compatibility
for deep learning tasks. It verifies:
- CUDA availability and version
- GPU hardware specifications
- cuDNN availability and version
- Memory availability
- Basic GPU functionality
- Driver status
"""

import torch
import sys
import platform
import subprocess
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class GPUInfo:
    """Class to store GPU information."""
    name: str
    total_memory: float
    compute_capability: str
    is_available: bool
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None

class GPUChecker:
    """Class to handle GPU compatibility checks."""
    
    def __init__(self):
        """Initialize the GPU checker."""
        self.system_info = self._get_system_info()
        self.gpu_info = {}
        self.check_results = {}
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        return {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'os': platform.system(),
            'os_version': platform.version(),
            'machine': platform.machine()
        }
    
    def _get_nvidia_driver_version(self) -> Optional[str]:
        """Get NVIDIA driver version using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except Exception as e:
            logging.warning(f"Could not get NVIDIA driver version: {e}")
            return None
    
    def _get_gpu_info(self) -> Dict[int, GPUInfo]:
        """Get detailed information about available GPUs."""
        gpu_info = {}
        
        if not torch.cuda.is_available():
            logging.warning("CUDA is not available")
            return gpu_info
        
        n_gpus = torch.cuda.device_count()
        driver_version = self._get_nvidia_driver_version()
        
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            gpu_info[i] = GPUInfo(
                name=torch.cuda.get_device_name(i),
                total_memory=props.total_memory / 1024**3,  # Convert to GB
                compute_capability=f"{props.major}.{props.minor}",
                is_available=True,
                driver_version=driver_version,
                cuda_version=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            )
        
        return gpu_info
    
    def _test_gpu_operation(self) -> bool:
        """Test GPU with a simple matrix multiplication operation."""
        try:
            # Create large matrices
            x = torch.rand(2000, 2000).cuda()
            y = torch.rand(2000, 2000).cuda()
            
            # Perform matrix multiplication
            z = torch.matmul(x, y)
            
            # Check if result is valid
            if torch.isnan(z).any() or torch.isinf(z).any():
                return False
            
            return True
        except Exception as e:
            logging.error(f"GPU test failed: {e}")
            return False
    
    def _check_memory_availability(self) -> Dict[str, Any]:
        """Check available GPU memory."""
        if not torch.cuda.is_available():
            return {'available': False}
        
        try:
            # Get current memory usage
            current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            max_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            free_memory = max_memory - current_memory
            
            return {
                'available': True,
                'total_memory': max_memory,
                'used_memory': current_memory,
                'free_memory': free_memory
            }
        except Exception as e:
            logging.error(f"Error checking memory: {e}")
            return {'available': False}
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all GPU compatibility checks."""
        self.check_results = {
            'system_info': self.system_info,
            'cuda_available': torch.cuda.is_available(),
            'gpu_info': self._get_gpu_info(),
            'memory_info': self._check_memory_availability(),
            'gpu_test_passed': self._test_gpu_operation() if torch.cuda.is_available() else False
        }
        return self.check_results
    
    def print_results(self) -> None:
        """Print check results in a formatted way."""
        print("\n=== GPU Compatibility Check Results ===")
        print("\nSystem Information:")
        print(f"Python Version: {self.system_info['python_version']}")
        print(f"PyTorch Version: {self.system_info['pytorch_version']}")
        print(f"OS: {self.system_info['os']} {self.system_info['os_version']}")
        
        print("\nCUDA Status:")
        print(f"CUDA Available: {self.check_results['cuda_available']}")
        
        if self.check_results['cuda_available']:
            print("\nGPU Information:")
            for gpu_id, info in self.check_results['gpu_info'].items():
                print(f"\nGPU {gpu_id}:")
                print(f"  Name: {info.name}")
                print(f"  Total Memory: {info.total_memory:.2f} GB")
                print(f"  Compute Capability: {info.compute_capability}")
                print(f"  Driver Version: {info.driver_version}")
                print(f"  CUDA Version: {info.cuda_version}")
                print(f"  cuDNN Version: {info.cudnn_version}")
            
            print("\nMemory Status:")
            mem_info = self.check_results['memory_info']
            if mem_info['available']:
                print(f"  Total Memory: {mem_info['total_memory']:.2f} GB")
                print(f"  Used Memory: {mem_info['used_memory']:.2f} GB")
                print(f"  Free Memory: {mem_info['free_memory']:.2f} GB")
            
            print("\nGPU Test:")
            print(f"  Basic Operation Test: {'Passed' if self.check_results['gpu_test_passed'] else 'Failed'}")
        else:
            print("\nPossible reasons for CUDA not being available:")
            print("1. No NVIDIA GPU found")
            print("2. NVIDIA drivers not installed")
            print("3. CUDA toolkit not installed")
            print("4. PyTorch not built with CUDA support")
    
    def save_results(self, filepath: str = "gpu_check_results.json") -> None:
        """Save check results to a JSON file."""
        try:
            # Convert GPUInfo objects to dictionaries
            results = self.check_results.copy()
            results['gpu_info'] = {
                str(k): v.__dict__ for k, v in results['gpu_info'].items()
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Results saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")

def main():
    """Main function to run GPU checks."""
    try:
        checker = GPUChecker()
        checker.run_checks()
        checker.print_results()
        checker.save_results()
    except Exception as e:
        logging.error(f"Error during GPU check: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 