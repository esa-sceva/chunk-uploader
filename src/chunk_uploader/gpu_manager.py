"""GPU memory management utilities."""
import torch
import gc
import time
from typing import Optional


class GPUManager:
    """Manage GPU memory and monitoring."""
    
    def __init__(self, memory_threshold: float = 0.85):
        self.memory_threshold = memory_threshold
        self.cuda_available = torch.cuda.is_available()
    
    def is_available(self) -> bool:
        """Check if CUDA is available."""
        return self.cuda_available
    
    def clear_cache(self):
        """Clear GPU cache."""
        if self.cuda_available:
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_memory_stats(self) -> Optional[dict]:
        """Get current GPU memory statistics."""
        if not self.cuda_available:
            return None
        
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "usage_percent": (reserved / total) * 100
        }
    
    def print_memory_stats(self, stage: str = ""):
        """Print GPU memory statistics."""
        stats = self.get_memory_stats()
        if not stats:
            return False
        
        stage_prefix = f"[{stage}] " if stage else ""
        print(f"{stage_prefix}GPU Memory: "
              f"{stats['allocated_gb']:.1f}GB allocated, "
              f"{stats['reserved_gb']:.1f}GB reserved "
              f"({stats['usage_percent']:.1f}% of {stats['total_gb']:.1f}GB)")
        
        return stats['usage_percent'] > (self.memory_threshold * 100)
    
    def check_and_clear_if_needed(self) -> bool:
        """Check memory usage and clear if threshold exceeded."""
        stats = self.get_memory_stats()
        if not stats:
            return False
        
        usage = stats['usage_percent']
        threshold_percent = self.memory_threshold * 100
        
        if usage > threshold_percent:
            print(f"High GPU memory usage: {usage:.2%}")
            print(f"Clearing GPU cache...")
            self.clear_cache()
            return True
        
        return False
    
    def monitor_with_delay(self, delay_seconds: float = 2.0):
        """Monitor GPU usage for a period before clearing."""
        if not self.cuda_available:
            return
        
        print(f"Monitoring GPU for {delay_seconds}s...")
        time.sleep(delay_seconds)
        self.clear_cache()
        print(f"GPU cache cleared")
    
    def get_device_info(self):
        """Get GPU device information."""
        if not self.cuda_available:
            return None
        
        return {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name()
        }
    
    def print_device_info(self):
        """Print GPU device information."""
        if not self.cuda_available:
            print("CUDA not available - operations will run on CPU")
            return
        
        info = self.get_device_info()
        print(f"GPU Info:")
        print(f"   Device count: {info['device_count']}")
        print(f"   Current device: {info['current_device']}")
        print(f"   Device name: {info['device_name']}")

