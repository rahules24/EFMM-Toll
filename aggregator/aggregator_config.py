"""
Aggregator Configuration Module
Configuration management for the Federated Aggregator Service
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import os

# YAML import would be added in production (requires pyyaml package)
# import yaml


@dataclass
class AggregatorConfig:
    """Configuration class for Federated Aggregator Service"""
    
    # Service configuration
    service_name: str = "efmm-aggregator"
    host: str = "localhost"
    port: int = 8003
    log_level: str = "INFO"
    
    # Federated Learning configuration
    fl_rounds_max: int = 100
    fl_participants_min: int = 3
    fl_round_timeout: int = 300
    aggregation_strategy: str = "fedavg"
    
    # Privacy configuration
    differential_privacy_epsilon: float = 1.0
    differential_privacy_delta: float = 1e-5
    secure_aggregation_enabled: bool = True
    
    # Model configuration
    model_repository_path: str = "./models"
    model_versions_to_keep: int = 10
    model_validation_threshold: float = 0.85
    
    # Participant management
    participant_timeout_minutes: int = 10
    heartbeat_interval: int = 30
    
    # Storage configuration
    database_path: str = "./aggregator.db"
    audit_log_path: str = "./audit"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'AggregatorConfig':
        """Load configuration from YAML file"""
        # TODO: Implement YAML loading (requires pyyaml package)
        # if os.path.exists(config_path):
        #     with open(config_path, 'r') as file:
        #         config_data = yaml.safe_load(file)
        #         return cls(**config_data)
        # else:
        #     return cls()
        
        # For now, return default configuration
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'service': {
                'name': self.service_name,
                'host': self.host,
                'port': self.port,
                'log_level': self.log_level
            },
            'federated_learning': {
                'rounds_max': self.fl_rounds_max,
                'participants_min': self.fl_participants_min,
                'round_timeout': self.fl_round_timeout,
                'aggregation_strategy': self.aggregation_strategy
            },
            'privacy': {
                'differential_privacy_epsilon': self.differential_privacy_epsilon,
                'differential_privacy_delta': self.differential_privacy_delta,
                'secure_aggregation_enabled': self.secure_aggregation_enabled
            },
            'model': {
                'repository_path': self.model_repository_path,
                'versions_to_keep': self.model_versions_to_keep,
                'validation_threshold': self.model_validation_threshold
            },
            'participants': {
                'timeout_minutes': self.participant_timeout_minutes,
                'heartbeat_interval': self.heartbeat_interval
            },
            'storage': {
                'database_path': self.database_path,
                'audit_log_path': self.audit_log_path
            }
        }
