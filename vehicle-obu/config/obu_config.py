"""
OBU Configuration Module
Configuration classes for Vehicle OBU Application
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class V2XConfig:
    """V2X communication configuration"""
    protocol: str = "DSRC"  # DSRC or C-V2X
    channel: int = 178
    tx_power: int = 20
    capabilities: List[str] = field(default_factory=lambda: ["toll_payment"])
    discovery_interval_seconds: int = 10
    handshake_timeout_seconds: int = 30
    pseudonym_rotation_minutes: int = 5
    discovery: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenConfig:
    """Token management configuration"""
    max_tokens: int = 5
    validation_timeout_seconds: int = 30


@dataclass
class WalletConfig:
    """Crypto wallet configuration"""
    initial_balance: float = 1000.0
    payment_methods: List[str] = field(default_factory=lambda: ["account_balance"])


@dataclass
class PrivacyConfig:
    """Privacy protection configuration"""
    privacy_level: str = "high"  # low, medium, high
    pseudonym_rotation: bool = True
    differential_privacy: bool = True


@dataclass
class FederatedLearningConfig:
    """Federated learning configuration"""
    enabled: bool = False
    server_url: str = ""
    privacy_budget: float = 1.0


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
            federated_learning=FederatedLearningConfig(**config_dict.get('federated_learning', {}))
        )
