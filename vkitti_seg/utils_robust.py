"""
Robust utility functions for RFF training scripts.
Provides device management, error handling, and common operations.
"""
import logging
import sys
import torch
import traceback
from pathlib import Path
from typing import Optional, Union, Dict, Any
import subprocess
import tempfile
import os


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup robust logging with proper formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:  # Avoid duplicate handlers
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device with proper error handling."""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        # Test CUDA functionality
        try:
            test_tensor = torch.tensor([1.0], device=device)
            _ = test_tensor + 1  # Simple operation test
            return device
        except Exception as e:
            logging.warning(f"CUDA available but not working: {e}. Falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")


def safe_load_model(model_path: Union[str, Path], device: torch.device, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Safely load a model with comprehensive error handling."""
    if logger is None:
        logger = setup_logging("safe_load_model")
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Load with proper device mapping
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        logger.info(f"Successfully loaded model from {model_path}")
        return state_dict
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def validate_file_exists(file_path: Union[str, Path], description: str = "file") -> Path:
    """Validate file exists and return Path object."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Required {description} not found: {file_path}")
    return file_path


def safe_tensor_to_device(tensor: torch.Tensor, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Safely move tensor to device with error handling."""
    try:
        result = tensor.to(device)
        if dtype is not None:
            result = result.to(dtype)
        return result
    except Exception as e:
        logging.error(f"Failed to move tensor to device {device}: {e}")
        raise


def create_output_directory(output_dir: Union[str, Path]) -> Path:
    """Create output directory with proper error handling."""
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    except Exception as e:
        raise RuntimeError(f"Failed to create output directory {output_dir}: {e}")


class RobustSubprocessRunner:
    """Robust subprocess runner with proper logging and error handling."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logging("subprocess_runner")
    
    def run_command(self, command: list, timeout: int = 7200, log_output: bool = True, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
        """Run command with robust error handling and logging.

        Args:
            command: Shell command arguments to execute.
            timeout: Max seconds to wait before timing out.
            log_output: Stream and log stdout in real-time when True.
            env: Optional environment overrides to pass to the subprocess.
        """
        self.logger.info(f"Executing command: {' '.join(command)}")

        # Compose environment for subprocess with safe defaults
        env_map = os.environ.copy()
        if env:
            env_map.update(env)
        # Fix common MKL + libgomp conflicts by defaulting to GNU threading unless already set
        if "MKL_THREADING_LAYER" not in env_map:
            env_map["MKL_THREADING_LAYER"] = "GNU"
            self.logger.info("CMD ENV: MKL_THREADING_LAYER=GNU (auto)")
        
        try:
            if log_output:
                # Stream output in real-time
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    env=env_map
                )
                
                output_lines = []
                for line in iter(process.stdout.readline, ''):
                    line = line.rstrip()
                    if line:
                        self.logger.info(f"CMD: {line}")
                        output_lines.append(line)
                
                process.wait(timeout=timeout)
                result = subprocess.CompletedProcess(
                    command, process.returncode, '\n'.join(output_lines)
                )
            else:
                result = subprocess.run(
                    command, 
                    capture_output=True, 
                    text=True, 
                    timeout=timeout,
                    check=False,
                    env=env_map
                )
            
            if result.returncode != 0:
                self.logger.error(f"Command failed with return code {result.returncode}")
                if hasattr(result, 'stderr') and result.stderr:
                    self.logger.error(f"Error output: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
            
            self.logger.info("Command completed successfully")
            return result
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out after {timeout} seconds")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error running command: {e}")
            raise


def safe_string_replace(content: str, old: str, new: str, expected_count: int = 1) -> str:
    """Safely replace strings with validation."""
    count = content.count(old)
    if count == 0:
        raise ValueError(f"Pattern not found: '{old}'")
    if count != expected_count:
        raise ValueError(f"Expected {expected_count} occurrences of '{old}', found {count}")
    
    return content.replace(old, new)


def validate_rff_setup(steering_model_path: Optional[str], beta: float, logger: logging.Logger) -> bool:
    """Validate RFF configuration parameters."""
    if steering_model_path is None:
        logger.error("RFF enabled but no steering_model_path provided")
        return False
    
    if not Path(steering_model_path).exists():
        logger.error(f"Steering model file not found: {steering_model_path}")
        return False
    
    if not (0.0 <= beta <= 1.0):
        logger.error(f"Beta must be between 0 and 1, got: {beta}")
        return False
    
    return True


class GPUMemoryManager:
    """Manage GPU memory efficiently."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logging("gpu_memory")
        self.device = get_device()
    
    def clear_cache(self):
        """Clear GPU cache if using CUDA."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            self.logger.debug("GPU cache cleared")
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            return {"allocated_gb": allocated, "reserved_gb": reserved}
        return {"allocated_gb": 0.0, "reserved_gb": 0.0}
    
    def log_memory_usage(self, prefix: str = ""):
        """Log current memory usage."""
        if self.device.type == "cuda":
            info = self.get_memory_info()
            self.logger.info(f"{prefix}GPU Memory - Allocated: {info['allocated_gb']:.2f}GB, Reserved: {info['reserved_gb']:.2f}GB")


# Global instances for convenience
logger = setup_logging("rff_utils")
gpu_manager = GPUMemoryManager(logger)
subprocess_runner = RobustSubprocessRunner(logger)