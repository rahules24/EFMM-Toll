"""
Blockchain Storage Manager
Handles persistent storage of blockchain data and audit records
"""

import asyncio
import logging
import sqlite3
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from ledger_node import Block, AuditRecord

logger = logging.getLogger(__name__)


class BlockchainStorage:
    """Manages persistent storage of blockchain data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get('database_path', './audit_ledger.db')
        self.connection = None
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize storage system"""
        logger.info("Initializing Blockchain Storage...")
        
        # Create database connection
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        
        # Create tables
        await self.create_tables()
        self.is_initialized = True
        
        logger.info(f"Blockchain storage initialized: {self.db_path}")
    
    async def create_tables(self):
        """Create database tables"""
        cursor = self.connection.cursor()
        
        # Blocks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blocks (
                block_index INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                hash_value TEXT NOT NULL,
                nonce INTEGER DEFAULT 0,
                record_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Audit records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_index INTEGER,
                timestamp TEXT NOT NULL,
                record_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT NOT NULL,
                hash_value TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (block_index) REFERENCES blocks (block_index)
            )
        ''')
        
        # Attestations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attestations (
                attestation_id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                attestation_data TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                signature TEXT NOT NULL,
                verification_status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.connection.commit()
    
    async def store_block(self, block: Block) -> bool:
        """Store a block in the database"""
        try:
            cursor = self.connection.cursor()
            
            # Store block
            cursor.execute('''
                INSERT INTO blocks (block_index, timestamp, previous_hash, hash_value, nonce, record_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                block.index,
                block.timestamp,
                block.previous_hash,
                block.hash_value,
                block.nonce,
                len(block.records)
            ))
            
            # Store audit records
            for record in block.records:
                cursor.execute('''
                    INSERT INTO audit_records (block_index, timestamp, record_type, entity_id, action, details, hash_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    block.index,
                    record.timestamp,
                    record.record_type,
                    record.entity_id,
                    record.action,
                    json.dumps(record.details),
                    record.hash_value
                ))
            
            self.connection.commit()
            logger.info(f"Stored block {block.index} with {len(block.records)} records")
            return True
            
        except Exception as e:
            logger.error(f"Error storing block {block.index}: {e}")
            self.connection.rollback()
            return False
    
    async def get_block(self, block_index: int) -> Optional[Block]:
        """Retrieve a block by index"""
        try:
            cursor = self.connection.cursor()
            
            # Get block data
            cursor.execute('SELECT * FROM blocks WHERE block_index = ?', (block_index,))
            block_row = cursor.fetchone()
            
            if not block_row:
                return None
            
            # Get audit records for this block
            cursor.execute('SELECT * FROM audit_records WHERE block_index = ?', (block_index,))
            record_rows = cursor.fetchall()
            
            # Reconstruct records
            records = []
            for row in record_rows:
                record = AuditRecord(
                    timestamp=row['timestamp'],
                    record_type=row['record_type'],
                    entity_id=row['entity_id'],
                    action=row['action'],
                    details=json.loads(row['details']),
                    hash_value=row['hash_value']
                )
                records.append(record)
            
            # Reconstruct block
            block = Block(
                index=block_row['block_index'],
                timestamp=block_row['timestamp'],
                records=records,
                previous_hash=block_row['previous_hash'],
                nonce=block_row['nonce'],
                hash_value=block_row['hash_value']
            )
            
            return block
            
        except Exception as e:
            logger.error(f"Error retrieving block {block_index}: {e}")
            return None
    
    async def get_blockchain_height(self) -> int:
        """Get the current blockchain height"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('SELECT MAX(block_index) FROM blocks')
            result = cursor.fetchone()
            return result[0] if result[0] is not None else -1
            
        except Exception as e:
            logger.error(f"Error getting blockchain height: {e}")
            return -1
    
    async def search_audit_records(self, entity_id: Optional[str] = None,
                                 record_type: Optional[str] = None,
                                 start_time: Optional[str] = None,
                                 end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search audit records with filters"""
        try:
            cursor = self.connection.cursor()
            
            query = 'SELECT * FROM audit_records WHERE 1=1'
            params = []
            
            if entity_id:
                query += ' AND entity_id = ?'
                params.append(entity_id)
            
            if record_type:
                query += ' AND record_type = ?'
                params.append(record_type)
            
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time)
            
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time)
            
            query += ' ORDER BY timestamp DESC'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                results.append({
                    'id': row['id'],
                    'block_index': row['block_index'],
                    'timestamp': row['timestamp'],
                    'record_type': row['record_type'],
                    'entity_id': row['entity_id'],
                    'action': row['action'],
                    'details': json.loads(row['details']),
                    'hash_value': row['hash_value']
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching audit records: {e}")
            return []
    
    async def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Blockchain storage connection closed")
