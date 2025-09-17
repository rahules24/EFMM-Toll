"""
Common Interfaces
Shared interfaces and abstract base classes for EFMM-Toll system
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import asyncio


class BaseSensorAdapter(ABC):
    """Abstract base class for sensor adapters"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the sensor adapter"""
        pass
    
    @abstractmethod
    async def start_sensing(self):
        """Start sensor data collection"""
        pass
    
    @abstractmethod
    async def stop_sensing(self):
        """Stop sensor data collection"""
        pass
    
    @abstractmethod
    async def get_sensor_data(self) -> Dict[str, Any]:
        """Get current sensor reading"""
        pass
    
    @abstractmethod
    async def calibrate_sensor(self) -> bool:
        """Calibrate the sensor"""
        pass


class BaseMLModel(ABC):
    """Abstract base class for ML models"""
    
    @abstractmethod
    async def load_model(self, model_path: str) -> bool:
        """Load model from file"""
        pass
    
    @abstractmethod
    async def save_model(self, model_path: str) -> bool:
        """Save model to file"""
        pass
    
    @abstractmethod
    async def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train the model"""
        pass
    
    @abstractmethod
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction"""
        pass
    
    @abstractmethod
    async def get_model_weights(self) -> Dict[str, Any]:
        """Get model weights for federated learning"""
        pass
    
    @abstractmethod
    async def set_model_weights(self, weights: Dict[str, Any]):
        """Set model weights from federated learning"""
        pass


class BaseCommunicationInterface(ABC):
    """Abstract base class for communication interfaces"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize communication interface"""
        pass
    
    @abstractmethod
    async def connect(self, endpoint: str) -> bool:
        """Connect to endpoint"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from endpoint"""
        pass
    
    @abstractmethod
    async def send_message(self, message: Dict[str, Any], destination: str) -> bool:
        """Send message to destination"""
        pass
    
    @abstractmethod
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message (non-blocking)"""
        pass
    
    @abstractmethod
    async def broadcast_message(self, message: Dict[str, Any]) -> bool:
        """Broadcast message to all connected peers"""
        pass


class BasePaymentProcessor(ABC):
    """Abstract base class for payment processing"""
    
    @abstractmethod
    async def initialize_wallet(self, wallet_config: Dict[str, Any]) -> bool:
        """Initialize payment wallet"""
        pass
    
    @abstractmethod
    async def get_balance(self) -> float:
        """Get current balance"""
        pass
    
    @abstractmethod
    async def generate_payment_proof(self, amount: float, recipient: str) -> str:
        """Generate zero-knowledge payment proof"""
        pass
    
    @abstractmethod
    async def verify_payment_proof(self, proof: str, amount: float, sender: str) -> bool:
        """Verify zero-knowledge payment proof"""
        pass
    
    @abstractmethod
    async def process_payment(self, amount: float, recipient: str, proof: str) -> bool:
        """Process payment transaction"""
        pass


class BasePrivacyManager(ABC):
    """Abstract base class for privacy management"""
    
    @abstractmethod
    async def generate_pseudonym(self, entity_id: str) -> str:
        """Generate privacy-preserving pseudonym"""
        pass
    
    @abstractmethod
    async def rotate_pseudonym(self, current_pseudonym: str) -> str:
        """Rotate to new pseudonym"""
        pass
    
    @abstractmethod
    async def anonymize_location(self, lat: float, lon: float) -> tuple[float, float]:
        """Anonymize location coordinates"""
        pass
    
    @abstractmethod
    async def add_differential_privacy(self, data: Dict[str, Any], epsilon: float) -> Dict[str, Any]:
        """Add differential privacy noise to data"""
        pass
    
    @abstractmethod
    async def encrypt_sensitive_data(self, data: Dict[str, Any]) -> str:
        """Encrypt sensitive data"""
        pass
    
    @abstractmethod
    async def decrypt_sensitive_data(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt sensitive data"""
        pass


class BaseTokenManager(ABC):
    """Abstract base class for token management"""
    
    @abstractmethod
    async def generate_token(self, entity_id: str, purpose: str, duration: int) -> Dict[str, Any]:
        """Generate ephemeral token"""
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> bool:
        """Validate token"""
        pass
    
    @abstractmethod
    async def revoke_token(self, token_id: str) -> bool:
        """Revoke token"""
        pass
    
    @abstractmethod
    async def refresh_token(self, old_token: str) -> Dict[str, Any]:
        """Refresh expired token"""
        pass
    
    @abstractmethod
    async def get_token_permissions(self, token: str) -> List[str]:
        """Get token permissions"""
        pass


class BaseAuditLogger(ABC):
    """Abstract base class for audit logging"""
    
    @abstractmethod
    async def log_event(self, event_type: str, entity_id: str, details: Dict[str, Any]) -> bool:
        """Log audit event"""
        pass
    
    @abstractmethod
    async def get_audit_trail(self, entity_id: str, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get audit trail for entity"""
        pass
    
    @abstractmethod
    async def verify_audit_integrity(self) -> bool:
        """Verify audit log integrity"""
        pass
    
    @abstractmethod
    async def create_attestation(self, entity_id: str, attestation_data: Dict[str, Any]) -> str:
        """Create TEE attestation"""
        pass
    
    @abstractmethod
    async def verify_attestation(self, attestation_id: str) -> bool:
        """Verify TEE attestation"""
        pass


class ServiceInterface(ABC):
    """Abstract base class for EFMM services"""
    
    @abstractmethod
    async def initialize(self):
        """Initialize the service"""
        pass
    
    @abstractmethod
    async def start(self):
        """Start the service"""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the service"""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        pass
    
    @abstractmethod
    async def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        pass


class ConfigurationInterface(ABC):
    """Abstract base class for configuration management"""
    
    @abstractmethod
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        pass
    
    @abstractmethod
    def save_config(self, config: Dict[str, Any], config_path: str) -> bool:
        """Save configuration to file"""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration"""
        pass
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        pass
