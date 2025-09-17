"""
Audit Verifier
Verifies the integrity and authenticity of audit records
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from ledger_node import AuditRecord, Block

logger = logging.getLogger(__name__)


class AuditVerifier:
    """Verifies audit records and blockchain integrity"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.verification_rules: Dict[str, Any] = config.get('verification_rules', {})
        self.is_running = False
    
    async def initialize(self):
        """Initialize audit verifier"""
        logger.info("Initializing Audit Verifier...")
        self.is_running = True
    
    async def start(self):
        """Start audit verifier"""
        logger.info("Starting Audit Verifier...")
    
    async def stop(self):
        """Stop audit verifier"""
        logger.info("Stopping Audit Verifier...")
        self.is_running = False
    
    async def verify_audit_record(self, record: AuditRecord) -> bool:
        """Verify a single audit record"""
        try:
            # Verify record hash
            expected_hash = record.calculate_hash()
            if record.hash_value != expected_hash:
                logger.error(f"Hash mismatch for record {record.entity_id}")
                return False
            
            # Verify record structure
            if not await self._verify_record_structure(record):
                logger.error(f"Invalid structure for record {record.entity_id}")
                return False
            
            # Verify record content
            if not await self._verify_record_content(record):
                logger.error(f"Invalid content for record {record.entity_id}")
                return False
            
            logger.info(f"Record verified: {record.entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying record {record.entity_id}: {e}")
            return False
    
    async def verify_block(self, block: Block) -> bool:
        """Verify a blockchain block"""
        try:
            # Verify block hash
            expected_hash = block.calculate_hash()
            if block.hash_value != expected_hash:
                logger.error(f"Hash mismatch for block {block.index}")
                return False
            
            # Verify all records in block
            for record in block.records:
                if not await self.verify_audit_record(record):
                    logger.error(f"Invalid record in block {block.index}")
                    return False
            
            logger.info(f"Block verified: {block.index}")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying block {block.index}: {e}")
            return False
    
    async def verify_blockchain(self, blockchain: List[Block]) -> bool:
        """Verify entire blockchain"""
        try:
            if not blockchain:
                logger.warning("Empty blockchain")
                return True
            
            # Verify genesis block
            if blockchain[0].index != 0 or blockchain[0].previous_hash != "0":
                logger.error("Invalid genesis block")
                return False
            
            # Verify each block and chain links
            for i in range(len(blockchain)):
                current_block = blockchain[i]
                
                # Verify block integrity
                if not await self.verify_block(current_block):
                    logger.error(f"Block {i} failed verification")
                    return False
                
                # Verify chain link (except for genesis block)
                if i > 0:
                    previous_block = blockchain[i - 1]
                    if current_block.previous_hash != previous_block.hash_value:
                        logger.error(f"Chain link broken at block {i}")
                        return False
            
            logger.info(f"Blockchain verified: {len(blockchain)} blocks")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying blockchain: {e}")
            return False
    
    async def _verify_record_structure(self, record: AuditRecord) -> bool:
        """Verify audit record has valid structure"""
        required_fields = ['timestamp', 'record_type', 'entity_id', 'action', 'details']
        
        for field in required_fields:
            if not hasattr(record, field) or getattr(record, field) is None:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Verify timestamp format
        try:
            datetime.fromisoformat(record.timestamp.replace('Z', '+00:00'))
        except ValueError:
            logger.error(f"Invalid timestamp format: {record.timestamp}")
            return False
        
        return True
    
    async def _verify_record_content(self, record: AuditRecord) -> bool:
        """Verify audit record content according to business rules"""
        try:
            # Verify record type
            valid_types = self.verification_rules.get('valid_record_types', [
                'toll_payment', 'vehicle_detection', 'token_generation', 
                'fl_participation', 'privacy_event'
            ])
            
            if record.record_type not in valid_types:
                logger.error(f"Invalid record type: {record.record_type}")
                return False
            
            # Verify entity ID format
            if not record.entity_id or len(record.entity_id) < 3:
                logger.error(f"Invalid entity ID: {record.entity_id}")
                return False
            
            # Verify details structure based on record type
            if not await self._verify_details_by_type(record.record_type, record.details):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying record content: {e}")
            return False
    
    async def _verify_details_by_type(self, record_type: str, details: Dict[str, Any]) -> bool:
        """Verify details structure based on record type"""
        try:
            if record_type == 'toll_payment':
                required_fields = ['amount', 'location', 'payment_proof']
                for field in required_fields:
                    if field not in details:
                        logger.error(f"Missing field {field} in toll_payment details")
                        return False
            
            elif record_type == 'vehicle_detection':
                required_fields = ['location', 'timestamp', 'sensor_data']
                for field in required_fields:
                    if field not in details:
                        logger.error(f"Missing field {field} in vehicle_detection details")
                        return False
            
            elif record_type == 'token_generation':
                required_fields = ['token_id', 'expiry', 'purpose']
                for field in required_fields:
                    if field not in details:
                        logger.error(f"Missing field {field} in token_generation details")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying details for {record_type}: {e}")
            return False
