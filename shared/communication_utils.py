"""
Shared Communication Utilities
Common communication protocols and message formats for EFMM-Toll system
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Standard message format for inter-component communication"""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: str
    signature: Optional[str] = None


class V2XProtocol:
    """Vehicle-to-X communication protocol utilities"""
    
    @staticmethod
    def create_toll_request(vehicle_id: str, rsu_id: str, location: Dict[str, float]) -> Message:
        """Create toll payment request message"""
        return Message(
            message_id=f"toll_req_{vehicle_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            sender_id=vehicle_id,
            receiver_id=rsu_id,
            message_type="TOLL_REQUEST",
            payload={
                "location": location,
                "vehicle_class": "passenger",  # TODO: Determine from vehicle data
                "requested_services": ["toll_payment"],
                "privacy_level": "high"
            },
            timestamp=datetime.now().isoformat()
        )
    
    @staticmethod
    def create_toll_response(rsu_id: str, vehicle_id: str, toll_amount: float, 
                           token: str) -> Message:
        """Create toll payment response message"""
        return Message(
            message_id=f"toll_resp_{rsu_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            sender_id=rsu_id,
            receiver_id=vehicle_id,
            message_type="TOLL_RESPONSE",
            payload={
                "toll_amount": toll_amount,
                "ephemeral_token": token,
                "payment_deadline": (
                    datetime.now().timestamp() + 300  # 5 minutes
                ),
                "accepted_payment_methods": ["zk_proof", "digital_currency"]
            },
            timestamp=datetime.now().isoformat()
        )
    
    @staticmethod
    def create_payment_proof(vehicle_id: str, rsu_id: str, proof: str, 
                           statement: Dict[str, Any]) -> Message:
        """Create zero-knowledge payment proof message"""
        return Message(
            message_id=f"payment_{vehicle_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            sender_id=vehicle_id,
            receiver_id=rsu_id,
            message_type="PAYMENT_PROOF",
            payload={
                "zk_proof": proof,
                "statement": statement,
                "payment_method": "zk_proof"
            },
            timestamp=datetime.now().isoformat()
        )


class FederatedLearningProtocol:
    """Federated learning communication protocol utilities"""
    
    @staticmethod
    def create_training_invitation(aggregator_id: str, participant_id: str,
                                 model_version: str, round_id: int) -> Message:
        """Create FL training round invitation"""
        return Message(
            message_id=f"fl_invite_{round_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            sender_id=aggregator_id,
            receiver_id=participant_id,
            message_type="FL_TRAINING_INVITATION",
            payload={
                "round_id": round_id,
                "model_version": model_version,
                "training_deadline": datetime.now().timestamp() + 3600,  # 1 hour
                "privacy_requirements": {
                    "differential_privacy": True,
                    "secure_aggregation": True
                }
            },
            timestamp=datetime.now().isoformat()
        )
    
    @staticmethod
    def create_model_update(participant_id: str, aggregator_id: str,
                          model_update: Dict[str, Any], round_id: int) -> Message:
        """Create model update submission"""
        return Message(
            message_id=f"fl_update_{participant_id}_{round_id}",
            sender_id=participant_id,
            receiver_id=aggregator_id,
            message_type="FL_MODEL_UPDATE",
            payload={
                "round_id": round_id,
                "model_update": model_update,
                "training_metrics": {
                    "samples_count": 100,  # TODO: Actual sample count
                    "training_loss": 0.1,  # TODO: Actual loss
                    "training_accuracy": 0.95  # TODO: Actual accuracy
                }
            },
            timestamp=datetime.now().isoformat()
        )


class AuditProtocol:
    """Audit ledger communication protocol utilities"""
    
    @staticmethod
    def create_audit_record_submission(sender_id: str, ledger_id: str,
                                     record_type: str, action: str,
                                     details: Dict[str, Any]) -> Message:
        """Create audit record submission"""
        return Message(
            message_id=f"audit_{sender_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            sender_id=sender_id,
            receiver_id=ledger_id,
            message_type="AUDIT_RECORD",
            payload={
                "record_type": record_type,
                "action": action,
                "details": details,
                "entity_attestation": None  # TODO: Add TEE attestation
            },
            timestamp=datetime.now().isoformat()
        )
    
    @staticmethod
    def create_attestation_request(entity_id: str, attestation_service_id: str,
                                 entity_type: str) -> Message:
        """Create TEE attestation request"""
        return Message(
            message_id=f"att_req_{entity_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            sender_id=entity_id,
            receiver_id=attestation_service_id,
            message_type="ATTESTATION_REQUEST",
            payload={
                "entity_type": entity_type,
                "attestation_data": {
                    "hardware_info": "TEE_enabled",
                    "software_version": "1.0.0",
                    "security_level": "high"
                }
            },
            timestamp=datetime.now().isoformat()
        )


class MessageRouter:
    """Message routing and delivery system"""
    
    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self.is_running = False
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register message handler for specific message type"""
        self.handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    async def route_message(self, message: Message) -> bool:
        """Route message to appropriate handler"""
        try:
            if message.message_type in self.handlers:
                handler = self.handlers[message.message_type]
                await handler(message)
                return True
            else:
                logger.warning(f"No handler registered for message type: {message.message_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error routing message {message.message_id}: {e}")
            return False
    
    def serialize_message(self, message: Message) -> str:
        """Serialize message to JSON string"""
        return json.dumps({
            'message_id': message.message_id,
            'sender_id': message.sender_id,
            'receiver_id': message.receiver_id,
            'message_type': message.message_type,
            'payload': message.payload,
            'timestamp': message.timestamp,
            'signature': message.signature
        })
    
    def deserialize_message(self, json_str: str) -> Message:
        """Deserialize JSON string to Message object"""
        data = json.loads(json_str)
        return Message(
            message_id=data['message_id'],
            sender_id=data['sender_id'],
            receiver_id=data['receiver_id'],
            message_type=data['message_type'],
            payload=data['payload'],
            timestamp=data['timestamp'],
            signature=data.get('signature')
        )
