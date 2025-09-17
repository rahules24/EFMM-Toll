"""
Audit Configuration Module
Configuration management for the Distributed Audit Ledger
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import os


@dataclass
class AuditLedgerConfig:
    """Configuration class for Distributed Audit Ledger"""
    
    # Service configuration
    service_name: str = "efmm-audit-ledger"
    host: str = "localhost"
    port: int = 8004
    log_level: str = "INFO"
    
    # Ledger configuration
    node_id: str = "audit-node-001"
    block_size: int = 10  # Number of records per block
    block_interval: int = 30  # Seconds between block creation
    sync_interval: int = 60  # Seconds between peer synchronization
    
    # Network configuration
    peers: List[str] = None
    consensus_algorithm: str = "proof_of_authority"
    network_timeout: int = 30
    
    # Storage configuration
    blockchain_path: str = "./blockchain"
    backup_interval: int = 3600  # Seconds between backups
    retention_days: int = 365
    
    # Security configuration
    encryption_enabled: bool = True
    digital_signatures_enabled: bool = True
    audit_verification_enabled: bool = True
    
    def __post_init__(self):
        if self.peers is None:
            self.peers = []
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'AuditLedgerConfig':
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
            'ledger': {
                'node_id': self.node_id,
                'block_size': self.block_size,
                'block_interval': self.block_interval,
                'sync_interval': self.sync_interval,
                'peers': self.peers,
                'consensus_algorithm': self.consensus_algorithm,
                'network_timeout': self.network_timeout
            },
            'storage': {
                'blockchain_path': self.blockchain_path,
                'backup_interval': self.backup_interval,
                'retention_days': self.retention_days
            },
            'security': {
                'encryption_enabled': self.encryption_enabled,
                'digital_signatures_enabled': self.digital_signatures_enabled,
                'audit_verification_enabled': self.audit_verification_enabled
            }
        }
