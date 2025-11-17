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
        import yaml
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config_data = yaml.safe_load(file)
                # Flatten the nested structure
                service = config_data.get('service', {})
                ledger = config_data.get('ledger', {})
                storage = config_data.get('storage', {})
                security = config_data.get('security', {})
                api = config_data.get('api', {})
                
                return cls(
                    service_name=service.get('name', 'efmm-audit-ledger'),
                    host=service.get('host', 'localhost'),
                    port=service.get('port', 8004),
                    log_level=service.get('log_level', 'INFO'),
                    node_id=ledger.get('node_id', 'audit-node-001'),
                    block_size=ledger.get('block_size', 10),
                    block_interval=ledger.get('block_interval', 30),
                    sync_interval=ledger.get('sync_interval', 60),
                    peers=ledger.get('peers', []),
                    consensus_algorithm=ledger.get('consensus_algorithm', 'proof_of_authority'),
                    blockchain_path=storage.get('blockchain_path', './blockchain'),
                    backup_interval=storage.get('backup_interval', 3600),
                    retention_days=storage.get('retention_days', 365),
                    encryption_enabled=security.get('encryption_enabled', True),
                    digital_signatures_enabled=security.get('digital_signatures_enabled', True),
                    audit_verification_enabled=security.get('audit_verification_enabled', True)
                )
        else:
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
