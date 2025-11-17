"""
RSU Configuration Module
Configuration classes and validation for RSU Edge Module
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import yaml
from pathlib import Path
import logging


@dataclass
class SensorConfig:
    """Configuration for sensor adapters"""
    enabled: Dict[str, bool] = field(default_factory=dict)
    camera: Dict[str, Any] = field(default_factory=dict)
    lidar: Dict[str, Any] = field(default_factory=dict)
    radar: Dict[str, Any] = field(default_factory=dict)
    loop_detector: Dict[str, Any] = field(default_factory=dict)
    anpr_cameras: List[Dict[str, Any]] = field(default_factory=list)
    rfid_readers: List[Dict[str, Any]] = field(default_factory=list)
    v2x_transceivers: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FusionConfig:
    """Configuration for fusion engine"""
    engine: str = "pytorch"
    model_path: str = "./models/fusion_model.pt"
    confidence_threshold: float = 0.8
    batch_interval_seconds: float = 1.0
    # Model details for fusion engine
    model: Dict[str, Any] = field(default_factory=lambda: {
        'fusion': {'feature_dim': 128},
        'modalities': {
            'anpr': {'feature_dim': 512},
            'rfid': {'feature_dim': 3},
            'v2x': {'feature_dim': 5}
        }
    })
    event_timeout_seconds: int = 30
    min_modalities_for_fusion: int = 1
    min_collection_time_seconds: int = 2


@dataclass
class FederatedLearningConfig:
    """Configuration for federated learning"""
    enabled: bool = True
    participant_id: str = "rsu-001"
    heartbeat_interval_seconds: int = 30
    aggregation_server_url: str = "http://localhost:5000"  # Default URL for aggregation server
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FederatedLearningConfig':
        """Create configuration from dictionary"""
        # Extract sections
        fl_dict = config_dict.get('fl_client', {})
        
        # Set RSU ID consistently across components
        rsu_id = config_dict.get('rsu_id', 'rsu_unknown')
        fl_dict.setdefault('participant_id', rsu_id)
        
        # Ensure aggregation_server_url is propagated to federated learning
        fl_dict.setdefault('aggregation_server_url', config_dict.get('aggregation_server_url', 'http://localhost:5000'))
        
        return cls(
            enabled=config_dict.get('enabled', True),
            participant_id=rsu_id,
            heartbeat_interval_seconds=config_dict.get('heartbeat_interval_seconds', 30),
            aggregation_server_url=config_dict.get('aggregation_server_url', 'http://localhost:5000')
        )


@dataclass
class TokenConfig:
    """Configuration for token orchestrator"""
    token_type: str = "ephemeral"
    default_duration_seconds: int = 300
    refresh_before_seconds: int = 30
    rsu_id: str = ""
    master_key: Optional[bytes] = None
    token_validity_seconds: int = 300
    max_tokens_per_lane: int = 10
    v2x: Dict[str, Any] = field(default_factory=dict)
    qr: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PaymentConfig:
    """Configuration for payment verifier"""
    zk_proof_enabled: bool = True
    payment_timeout_seconds: int = 30
    zk_verification: Dict[str, Any] = field(default_factory=dict)
    max_proof_age_minutes: int = 30


@dataclass
class TEEConfig:
    """Configuration for TEE manager"""
    tee_type: str = "simulation"  # 'sgx', 'trustzone', 'simulation'
    generate_attestations: bool = True
    rsu_id: str = ""
    verifier_version: str = "1.0"


@dataclass
class AuditConfig:
    """Configuration for audit buffer"""
    audit_api_host: str = "audit-ledger"
    audit_api_port: int = 8004
    rsu_id: str = ""
    database_path: str = "data/audit_buffer.db"
    batch_size: int = 100
    batch_interval_seconds: int = 300
    cleanup_age_days: int = 30
    cleanup_interval_hours: int = 24


@dataclass
class RSUConfig:
    """Main RSU configuration"""
    mode: str = "production"  # 'production', 'simulation', 'testing'
    rsu_id: str = ""
    sensors: SensorConfig = field(default_factory=SensorConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    federated_learning: FederatedLearningConfig = field(default_factory=FederatedLearningConfig)
    tokens: TokenConfig = field(default_factory=TokenConfig)
    payments: PaymentConfig = field(default_factory=PaymentConfig)
    tee: TEEConfig = field(default_factory=TEEConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    aggregation_server_url: str = "http://localhost:5000"  # Default URL for aggregation server
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RSUConfig':
        """Create configuration from dictionary"""
        # Extract sections
        sensors_dict = config_dict.get('sensors', {})
        fusion_dict = config_dict.get('fusion', {})
        fl_dict = config_dict.get('fl_client', {})
        tokens_dict = config_dict.get('tokens', {})
        payments_dict = config_dict.get('payments', {})
        tee_dict = config_dict.get('tee', {})
        audit_dict = config_dict.get('audit', {})
        
        # Set RSU ID consistently across components
        rsu_id = config_dict.get('rsu_id', 'rsu_unknown')
        fl_dict.setdefault('participant_id', rsu_id)
        tokens_dict.setdefault('rsu_id', rsu_id)
        tee_dict.setdefault('rsu_id', rsu_id)
        audit_dict.setdefault('rsu_id', rsu_id)
        
        # Ensure aggregation_server_url is propagated to federated learning
        fl_dict.setdefault('aggregation_server_url', config_dict.get('aggregation_server_url', 'http://localhost:5000'))
        
        # Debug log to verify participant_id
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"FL Config: {fl_dict}")
        
        return cls(
            mode=config_dict.get('mode', 'production'),
            rsu_id=rsu_id,
            sensors=SensorConfig(**sensors_dict),
            fusion=FusionConfig(**fusion_dict),
            federated_learning=FederatedLearningConfig(**fl_dict),
            tokens=TokenConfig(**tokens_dict),
            payments=PaymentConfig(**payments_dict),
            tee=TEEConfig(**tee_dict),
            audit=AuditConfig(**audit_dict),
            aggregation_server_url=config_dict.get('aggregation_server_url', 'http://localhost:5000')
        )
