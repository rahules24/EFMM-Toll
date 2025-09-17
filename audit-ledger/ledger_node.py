"""
Ledger Node
Individual node in the distributed audit ledger network
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import dataclass
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class AuditRecord:
    """Audit record structure"""
    timestamp: str
    record_type: str
    entity_id: str
    action: str
    details: Dict[str, Any]
    hash_value: str = ""
    
    def calculate_hash(self) -> str:
        """Calculate hash of the audit record"""
        record_data = {
            'timestamp': self.timestamp,
            'record_type': self.record_type,
            'entity_id': self.entity_id,
            'action': self.action,
            'details': self.details
        }
        
        record_json = json.dumps(record_data, sort_keys=True)
        return hashlib.sha256(record_json.encode()).hexdigest()


@dataclass
class Block:
    """Blockchain block structure"""
    index: int
    timestamp: str
    records: List[AuditRecord]
    previous_hash: str
    nonce: int = 0
    hash_value: str = ""
    
    def calculate_hash(self) -> str:
        """Calculate hash of the block"""
        block_data = {
            'index': self.index,
            'timestamp': self.timestamp,
            'records': [record.__dict__ for record in self.records],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }
        
        block_json = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_json.encode()).hexdigest()


class LedgerNode:
    """Distributed ledger node for audit records"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.node_id = config.get('node_id', 'node-001')
        self.blockchain: List[Block] = []
        self.pending_records: List[AuditRecord] = []
        self.peers: List[str] = config.get('peers', [])
        self.is_running = False
    
    async def initialize(self):
        """Initialize the ledger node"""
        logger.info(f"Initializing Ledger Node: {self.node_id}")
        
        # Create genesis block if blockchain is empty
        if not self.blockchain:
            await self.create_genesis_block()
    
    async def start(self):
        """Start the ledger node"""
        logger.info(f"Starting Ledger Node: {self.node_id}")
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self.process_pending_records())
        asyncio.create_task(self.sync_with_peers())
    
    async def stop(self):
        """Stop the ledger node"""
        logger.info(f"Stopping Ledger Node: {self.node_id}")
        self.is_running = False
    
    async def create_genesis_block(self):
        """Create the genesis block"""
        genesis_block = Block(
            index=0,
            timestamp=datetime.now().isoformat(),
            records=[],
            previous_hash="0"
        )
        genesis_block.hash_value = genesis_block.calculate_hash()
        self.blockchain.append(genesis_block)
        
        logger.info("Genesis block created")
    
    async def add_audit_record(self, record: AuditRecord):
        """Add an audit record to pending records"""
        record.hash_value = record.calculate_hash()
        self.pending_records.append(record)
        
        logger.info(f"Added audit record: {record.record_type} for {record.entity_id}")
    
    async def process_pending_records(self):
        """Process pending records and create blocks"""
        while self.is_running:
            try:
                if len(self.pending_records) >= self.config.get('block_size', 10):
                    await self.create_block()
                
                await asyncio.sleep(self.config.get('block_interval', 30))
                
            except Exception as e:
                logger.error(f"Error processing pending records: {e}")
                await asyncio.sleep(5)
    
    async def create_block(self):
        """Create a new block from pending records"""
        if not self.pending_records:
            return
        
        last_block = self.blockchain[-1] if self.blockchain else None
        previous_hash = last_block.hash_value if last_block else "0"
        
        new_block = Block(
            index=len(self.blockchain),
            timestamp=datetime.now().isoformat(),
            records=self.pending_records.copy(),
            previous_hash=previous_hash
        )
        
        new_block.hash_value = new_block.calculate_hash()
        self.blockchain.append(new_block)
        self.pending_records.clear()
        
        logger.info(f"Created block {new_block.index} with {len(new_block.records)} records")
    
    async def sync_with_peers(self):
        """Synchronize blockchain with peer nodes"""
        while self.is_running:
            try:
                # TODO: Implement peer synchronization
                await asyncio.sleep(self.config.get('sync_interval', 60))
                
            except Exception as e:
                logger.error(f"Error syncing with peers: {e}")
                await asyncio.sleep(10)
    
    async def verify_chain(self) -> bool:
        """Verify the integrity of the blockchain"""
        for i in range(1, len(self.blockchain)):
            current_block = self.blockchain[i]
            previous_block = self.blockchain[i - 1]
            
            # Verify current block hash
            if current_block.hash_value != current_block.calculate_hash():
                logger.error(f"Invalid hash for block {i}")
                return False
            
            # Verify previous hash link
            if current_block.previous_hash != previous_block.hash_value:
                logger.error(f"Invalid previous hash for block {i}")
                return False
        
        return True
