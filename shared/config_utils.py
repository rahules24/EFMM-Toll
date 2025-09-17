"""
Configuration Utilities
Shared configuration management utilities for EFMM-Toll system
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class NetworkConfig:
    """Network configuration settings"""
    interface: str = "eth0"
    v2x_frequency: float = 5.9  # GHz
    dsrc_channel: int = 178
    transmission_power: int = 20  # dBm
    communication_range: int = 300  # meters
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'interface': self.interface,
            'v2x_frequency': self.v2x_frequency,
            'dsrc_channel': self.dsrc_channel,
            'transmission_power': self.transmission_power,
            'communication_range': self.communication_range
        }


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_interval: int = 3600  # seconds
    certificate_validity: int = 86400  # seconds
    max_attestation_age: int = 300  # seconds
    privacy_level: str = "high"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'encryption_algorithm': self.encryption_algorithm,
            'key_rotation_interval': self.key_rotation_interval,
            'certificate_validity': self.certificate_validity,
            'max_attestation_age': self.max_attestation_age,
            'privacy_level': self.privacy_level
        }


@dataclass
class PerformanceConfig:
    """Performance configuration settings"""
    max_concurrent_connections: int = 1000
    message_queue_size: int = 10000
    processing_timeout: int = 30  # seconds
    heartbeat_interval: int = 10  # seconds
    cleanup_interval: int = 300  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'max_concurrent_connections': self.max_concurrent_connections,
            'message_queue_size': self.message_queue_size,
            'processing_timeout': self.processing_timeout,
            'heartbeat_interval': self.heartbeat_interval,
            'cleanup_interval': self.cleanup_interval
        }


class ConfigManager:
    """Configuration management utility"""
    
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = config_dir
        self.configs: Dict[str, Any] = {}
    
    def load_config(self, config_name: str, default_config: Any = None) -> Any:
        """Load configuration from file or return default"""
        config_path = os.path.join(self.config_dir, f"{config_name}.yaml")
        
        # TODO: Implement YAML loading when pyyaml is available
        # For now, return default configuration
        if default_config:
            self.configs[config_name] = default_config
            return default_config
        
        return {}
    
    def get_config(self, config_name: str) -> Optional[Any]:
        """Get loaded configuration"""
        return self.configs.get(config_name)
    
    def set_config(self, config_name: str, config: Any):
        """Set configuration programmatically"""
        self.configs[config_name] = config
    
    def get_environment_config(self, key: str, default: str = "") -> str:
        """Get configuration from environment variables"""
        return os.getenv(f"EFMM_{key.upper()}", default)
    
    def create_default_configs(self):
        """Create default configurations for all components"""
        # Network configuration
        network_config = NetworkConfig()
        self.set_config('network', network_config)
        
        # Security configuration  
        security_config = SecurityConfig()
        self.set_config('security', security_config)
        
        # Performance configuration
        performance_config = PerformanceConfig()
        self.set_config('performance', performance_config)
        
        return {
            'network': network_config.to_dict(),
            'security': security_config.to_dict(),
            'performance': performance_config.to_dict()
        }


def get_component_config(component_name: str, config_path: str = None) -> Dict[str, Any]:
    """Get configuration for a specific component"""
    config_manager = ConfigManager()
    
    # Load component-specific configuration
    component_config = config_manager.load_config(component_name)
    
    # Load shared configurations
    shared_configs = config_manager.create_default_configs()
    
    # Merge configurations
    final_config = {
        **shared_configs,
        'component': component_config
    }
    
    return final_config


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """Validate that configuration contains required keys"""
    for key in required_keys:
        if key not in config:
            return False
    return True
