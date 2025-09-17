"""
Protocol Definitions
Common protocol definitions and message formats for EFMM-Toll system
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import json


class MessageType(Enum):
    """Message types for inter-service communication"""
    VEHICLE_DETECTION = "vehicle_detection"
    TOLL_PAYMENT = "toll_payment"
    TOKEN_REQUEST = "token_request"
    TOKEN_RESPONSE = "token_response"
    FL_UPDATE = "fl_update"
    FL_AGGREGATION = "fl_aggregation"
    AUDIT_RECORD = "audit_record"
    ATTESTATION = "attestation"
    HEARTBEAT = "heartbeat"
    STATUS_UPDATE = "status_update"


class EntityType(Enum):
    """Entity types in EFMM system"""
    RSU = "rsu"
    VEHICLE = "vehicle"
    AGGREGATOR = "aggregator"
    AUDIT_LEDGER = "audit_ledger"


@dataclass
class BaseMessage:
    """Base message structure for all communications"""
    message_id: str
    message_type: MessageType
    sender_id: str
    sender_type: EntityType
    timestamp: str
    payload: Dict[str, Any]
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'sender_id': self.sender_id,
            'sender_type': self.sender_type.value,
            'timestamp': self.timestamp,
            'payload': self.payload,
            'signature': self.signature
        }
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseMessage':
        """Create message from dictionary"""
        return cls(
            message_id=data['message_id'],
            message_type=MessageType(data['message_type']),
            sender_id=data['sender_id'],
            sender_type=EntityType(data['sender_type']),
            timestamp=data['timestamp'],
            payload=data['payload'],
            signature=data.get('signature')
        )


@dataclass
class VehicleDetectionMessage:
    """Vehicle detection message payload"""
    vehicle_id: str
    location: Dict[str, float]  # lat, lon
    speed: float
    direction: float
    sensor_data: Dict[str, Any]
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TollPaymentMessage:
    """Toll payment message payload"""
    vehicle_id: str
    amount: float
    location: Dict[str, float]
    payment_proof: str
    token_id: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenRequestMessage:
    """Ephemeral token request payload"""
    requester_id: str
    requester_type: EntityType
    purpose: str
    duration: int  # seconds
    location: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['requester_type'] = self.requester_type.value
        return result


@dataclass
class TokenResponseMessage:
    """Ephemeral token response payload"""
    token_id: str
    token_data: str
    expiry: str
    permissions: List[str]
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FLUpdateMessage:
    """Federated learning model update payload"""
    participant_id: str
    round_number: int
    model_update: str  # Serialized model weights
    update_hash: str
    metrics: Dict[str, float]
    privacy_budget: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FLAggregationMessage:
    """Federated learning aggregation result payload"""
    round_number: int
    aggregated_model: str  # Serialized aggregated model
    model_hash: str
    participants: List[str]
    global_metrics: Dict[str, float]
    next_round_params: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttestationMessage:
    """TEE Attestation message payload"""
    attestation_id: str
    entity_id: str
    entity_type: EntityType
    attestation_data: Dict[str, Any]
    signature: str
    verification_status: str
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['entity_type'] = self.entity_type.value
        return result


class ProtocolHandler:
    """Handler for protocol message creation and parsing"""
    
    @staticmethod
    def create_message(message_type: MessageType, sender_id: str, 
                      sender_type: EntityType, payload: Dict[str, Any],
                      message_id: Optional[str] = None) -> BaseMessage:
        """Create a protocol message"""
        import uuid
        
        if message_id is None:
            message_id = str(uuid.uuid4())
        
        return BaseMessage(
            message_id=message_id,
            message_type=message_type,
            sender_id=sender_id,
            sender_type=sender_type,
            timestamp=datetime.now().isoformat(),
            payload=payload
        )
    
    @staticmethod
    def parse_message(message_data: str) -> BaseMessage:
        """Parse message from JSON string"""
        data = json.loads(message_data)
        return BaseMessage.from_dict(data)
    
    @staticmethod
    def create_vehicle_detection_message(sender_id: str, detection_data: VehicleDetectionMessage) -> BaseMessage:
        """Create vehicle detection message"""
        return ProtocolHandler.create_message(
            MessageType.VEHICLE_DETECTION,
            sender_id,
            EntityType.RSU,
            detection_data.to_dict()
        )
    
    @staticmethod
    def create_toll_payment_message(sender_id: str, payment_data: TollPaymentMessage) -> BaseMessage:
        """Create toll payment message"""
        return ProtocolHandler.create_message(
            MessageType.TOLL_PAYMENT,
            sender_id,
            EntityType.VEHICLE,
            payment_data.to_dict()
        )
    
    @staticmethod
    def create_token_request_message(sender_id: str, sender_type: EntityType, 
                                   request_data: TokenRequestMessage) -> BaseMessage:
        """Create token request message"""
        return ProtocolHandler.create_message(
            MessageType.TOKEN_REQUEST,
            sender_id,
            sender_type,
            request_data.to_dict()
        )
    
    @staticmethod
    def create_fl_update_message(sender_id: str, sender_type: EntityType,
                               update_data: FLUpdateMessage) -> BaseMessage:
        """Create federated learning update message"""
        return ProtocolHandler.create_message(
            MessageType.FL_UPDATE,
            sender_id,
            sender_type,
            update_data.to_dict()
        )
