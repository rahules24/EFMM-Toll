"""
OBU Configuration Module
Configuration classes for Vehicle OBU Application
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class V2XConfig:
    """V2X communication configuration"""
    protocol: str = "dsrc"
    v2x_interface: str = "wlan0"
    heartbeat_interval_seconds: int = 30


@dataclass
class TokenConfig:
    """Token management configuration"""
    max_tokens: int = 5
    validation_timeout_seconds: int = 30


@dataclass
class WalletConfig:
    """Crypto wallet configuration"""
    zk_proof_curves: str = "secp256k1"
    default_balance: float = 100.0
    auto_topup: bool = False


@dataclass
class PrivacyConfig:
    """Privacy protection configuration"""
    pseudonym_rotation_seconds: int = 3600
    location_anonymization_radius_m: int = 100


@dataclass
class FederatedLearningConfig:
    """Federated learning configuration"""
    enabled: bool = True
    participant_id_prefix: str = "vehicle"
    max_concurrent_updates: int = 1


@dataclass
class OBUConfig:
    """Main OBU configuration"""
    vehicle_id: str = ""
    mode: str = "embedded"  # embedded, mobile, simulation
    v2x: V2XConfig = field(default_factory=V2XConfig)
    tokens: TokenConfig = field(default_factory=TokenConfig)
    wallet: WalletConfig = field(default_factory=WalletConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    federated_learning: FederatedLearningConfig = field(default_factory=FederatedLearningConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OBUConfig':
        """Create configuration from dictionary"""
        return cls(
            vehicle_id=config_dict.get('vehicle_id', 'vehicle_unknown'),
            mode=config_dict.get('mode', 'embedded'),
            v2x=V2XConfig(**config_dict.get('v2x', {})),
            tokens=TokenConfig(**config_dict.get('tokens', {})),
            wallet=WalletConfig(**config_dict.get('wallet', {})),
            privacy=PrivacyConfig(**config_dict.get('privacy', {})),
            federated_learning=FederatedLearningConfig(**config_dict.get('fl_client', {}))
        )
