"""
RSU Configuration Module
Configuration classes and validation for RSU Edge Module
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import yaml
from pathlib import Path


@dataclass
class SensorConfig:
    """Configuration for sensor adapters"""
    anpr_cameras: List[Dict[str, Any]] = field(default_factory=list)
    rfid_readers: List[Dict[str, Any]] = field(default_factory=list)
    v2x_transceivers: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FusionConfig:
    """Configuration for fusion engine"""
    model: Dict[str, Any] = field(default_factory=dict)
    model_path: Optional[str] = None
    event_timeout_seconds: int = 30
    min_modalities_for_fusion: int = 1
    min_collection_time_seconds: int = 2


@dataclass
class FederatedLearningConfig:
    """Configuration for federated learning"""
    participant_id: str = ""
    aggregation_server_url: str = ""
    learning_rate: float = 0.001
    batch_size: int = 32
    communication_interval_seconds: int = 30
    training_interval_seconds: int = 60
    min_training_samples: int = 10
    max_buffer_size: int = 1000
    min_update_interval_seconds: int = 300
    use_differential_privacy: bool = True
    use_secure_aggregation: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    supported_modalities: List[str] = field(default_factory=list)
    model_version: str = "1.0"


@dataclass
class TokenConfig:
    """Configuration for token orchestrator"""
    rsu_id: str = ""
    master_key: Optional[bytes] = None
    token_validity_seconds: int = 300
    max_tokens_per_lane: int = 10
    v2x: Dict[str, Any] = field(default_factory=dict)
    qr: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PaymentConfig:
    """Configuration for payment verifier"""
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
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RSUConfig':
        """Create configuration from dictionary"""
        # Extract sections
        sensors_dict = config_dict.get('sensors', {})
        fusion_dict = config_dict.get('fusion', {})
        fl_dict = config_dict.get('federated_learning', {})
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
        
        return cls(
            mode=config_dict.get('mode', 'production'),
            rsu_id=rsu_id,
            sensors=SensorConfig(**sensors_dict),
            fusion=FusionConfig(**fusion_dict),
            federated_learning=FederatedLearningConfig(**fl_dict),
            tokens=TokenConfig(**tokens_dict),
            payments=PaymentConfig(**payments_dict),
            tee=TEEConfig(**tee_dict),
            audit=AuditConfig(**audit_dict)
        )
