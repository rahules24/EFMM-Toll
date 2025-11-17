"""
Audit Buffer for RSU Edge Module
Local storage and management of audit records before ledger submission

Key features:
- Temporary storage of attestations and audit records
- Batch processing for efficient ledger submission
- Local audit trail for compliance
- Data integrity and tamper detection
- Automatic cleanup and archival
"""

import asyncio
import logging
import sqlite3
import json
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os
from pathlib import Path

from shared.attestation import TEEAttestation

logger = logging.getLogger(__name__)


@dataclass
class AuditRecord:
    """Local audit record structure"""
    record_id: str
    record_type: str  # 'attestation', 'token_event', 'system_event'
    timestamp: datetime
    rsu_id: str
    ephemeral_token_id: Optional[str]
    data: Dict[str, Any]
    hash_chain_prev: Optional[str]
    hash_chain_current: str
    submitted_to_ledger: bool
    submission_timestamp: Optional[datetime]


@dataclass
class BatchSubmission:
    """Batch of records for ledger submission"""
    batch_id: str
    timestamp: datetime
    records: List[AuditRecord]
    batch_hash: str
    submission_status: str  # 'pending', 'submitted', 'confirmed', 'failed'


class HashChainManager:
    """Manages hash chain for audit record integrity"""
    
    def __init__(self):
        self.last_hash = "0" * 64  # Genesis hash
    
    def compute_record_hash(self, record: AuditRecord, prev_hash: str) -> str:
        """Compute hash for audit record"""
        # Create deterministic string representation
        hash_data = {
            'record_id': record.record_id,
            'record_type': record.record_type,
            'timestamp': record.timestamp.isoformat(),
            'rsu_id': record.rsu_id,
            'ephemeral_token_id': record.ephemeral_token_id,
            'data': record.data,
            'prev_hash': prev_hash
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def verify_hash_chain(self, records: List[AuditRecord]) -> bool:
        """Verify integrity of hash chain"""
        if not records:
            return True
        
        # Sort records by timestamp
        sorted_records = sorted(records, key=lambda r: r.timestamp)
        
        expected_prev_hash = self.last_hash
        for record in sorted_records:
            # Verify previous hash
            if record.hash_chain_prev != expected_prev_hash:
                logger.error(f"Hash chain broken at record {record.record_id}")
                return False
            
            # Verify current hash
            expected_current_hash = self.compute_record_hash(record, expected_prev_hash)
            if record.hash_chain_current != expected_current_hash:
                logger.error(f"Invalid hash for record {record.record_id}")
                return False
            
            expected_prev_hash = record.hash_chain_current
        
        return True
    
    def update_last_hash(self, new_hash: str):
        """Update the last hash in chain"""
        self.last_hash = new_hash


class LocalDatabase:
    """SQLite database for local audit storage"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
    
    async def initialize(self):
        """Initialize database schema"""
        try:
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            
            # Create tables
            await self._create_tables()
            
            logger.info(f"Audit database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize audit database: {e}")
            raise
    
    async def _create_tables(self):
        """Create database tables"""
        cursor = self.connection.cursor()
        
        # Audit records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_records (
                record_id TEXT PRIMARY KEY,
                record_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                rsu_id TEXT NOT NULL,
                ephemeral_token_id TEXT,
                data TEXT NOT NULL,
                hash_chain_prev TEXT,
                hash_chain_current TEXT NOT NULL,
                submitted_to_ledger INTEGER DEFAULT 0,
                submission_timestamp TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Batch submissions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS batch_submissions (
                batch_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                batch_hash TEXT NOT NULL,
                submission_status TEXT NOT NULL,
                record_count INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Batch records mapping table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS batch_records (
                batch_id TEXT NOT NULL,
                record_id TEXT NOT NULL,
                PRIMARY KEY (batch_id, record_id),
                FOREIGN KEY (batch_id) REFERENCES batch_submissions(batch_id),
                FOREIGN KEY (record_id) REFERENCES audit_records(record_id)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_records_timestamp ON audit_records(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_records_token ON audit_records(ephemeral_token_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_records_submitted ON audit_records(submitted_to_ledger)')
        
        self.connection.commit()
    
    async def store_record(self, record: AuditRecord):
        """Store audit record in database"""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            INSERT INTO audit_records (
                record_id, record_type, timestamp, rsu_id, ephemeral_token_id,
                data, hash_chain_prev, hash_chain_current, submitted_to_ledger,
                submission_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.record_id,
            record.record_type,
            record.timestamp.isoformat(),
            record.rsu_id,
            record.ephemeral_token_id,
            json.dumps(record.data),
            record.hash_chain_prev,
            record.hash_chain_current,
            1 if record.submitted_to_ledger else 0,
            record.submission_timestamp.isoformat() if record.submission_timestamp else None
        ))
        
        self.connection.commit()
    
    async def get_unsubmitted_records(self, limit: Optional[int] = None) -> List[AuditRecord]:
        """Get records not yet submitted to ledger"""
        cursor = self.connection.cursor()
        
        query = '''
            SELECT * FROM audit_records 
            WHERE submitted_to_ledger = 0 
            ORDER BY timestamp ASC
        '''
        
        if limit:
            query += f' LIMIT {limit}'
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        return [self._row_to_record(row) for row in rows]
    
    async def mark_records_submitted(self, record_ids: List[str], batch_id: str):
        """Mark records as submitted to ledger"""
        cursor = self.connection.cursor()
        
        placeholders = ','.join(['?' for _ in record_ids])
        cursor.execute(f'''
            UPDATE audit_records 
            SET submitted_to_ledger = 1, submission_timestamp = ?
            WHERE record_id IN ({placeholders})
        ''', [datetime.now().isoformat()] + record_ids)
        
        # Store batch submission info
        cursor.execute('''
            INSERT INTO batch_submissions (batch_id, timestamp, batch_hash, submission_status, record_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (batch_id, datetime.now().isoformat(), "", "submitted", len(record_ids)))
        
        # Store batch-record mappings
        for record_id in record_ids:
            cursor.execute('''
                INSERT INTO batch_records (batch_id, record_id) VALUES (?, ?)
            ''', (batch_id, record_id))
        
        self.connection.commit()
    
    async def get_records_by_token(self, ephemeral_token_id: str) -> List[AuditRecord]:
        """Get all records for a specific token"""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            SELECT * FROM audit_records 
            WHERE ephemeral_token_id = ? 
            ORDER BY timestamp ASC
        ''', (ephemeral_token_id,))
        
        rows = cursor.fetchall()
        return [self._row_to_record(row) for row in rows]
    
    async def cleanup_old_records(self, older_than: datetime) -> int:
        """Clean up old submitted records"""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            DELETE FROM audit_records 
            WHERE submitted_to_ledger = 1 AND timestamp < ?
        ''', (older_than.isoformat(),))
        
        deleted_count = cursor.rowcount
        self.connection.commit()
        
        return deleted_count
    
    def _row_to_record(self, row) -> AuditRecord:
        """Convert database row to AuditRecord"""
        return AuditRecord(
            record_id=row['record_id'],
            record_type=row['record_type'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            rsu_id=row['rsu_id'],
            ephemeral_token_id=row['ephemeral_token_id'],
            data=json.loads(row['data']),
            hash_chain_prev=row['hash_chain_prev'],
            hash_chain_current=row['hash_chain_current'],
            submitted_to_ledger=bool(row['submitted_to_ledger']),
            submission_timestamp=datetime.fromisoformat(row['submission_timestamp']) 
                                if row['submission_timestamp'] else None
        )
    
    async def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()


class AuditBuffer:
    """
    Audit Buffer
    
    Manages local storage of audit records and attestations before
    submission to the distributed ledger.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rsu_id = config['rsu_id']
        self.is_running = False
        
        # Database
        db_path = config.get('database_path', 'data/audit_buffer.db')
        self.database = LocalDatabase(db_path)
        
        # Hash chain management
        self.hash_chain = HashChainManager()
        
        # Batch processing
        self.batch_size = config.get('batch_size', 100)
        self.batch_interval = timedelta(seconds=config.get('batch_interval_seconds', 300))
        self.last_batch_time = datetime.now()
        
        # Background tasks
        self.processing_task = None
        self.cleanup_task = None
        
        # Statistics
        self.stats = {
            'records_stored': 0,
            'records_submitted': 0,
            'batches_submitted': 0,
            'hash_chain_verifications': 0,
            'cleanup_operations': 0
        }
    
    async def initialize(self):
        """Initialize audit buffer"""
        logger.info("Initializing Audit Buffer...")
        
        try:
            # Initialize database
            await self.database.initialize()
            
            # Initialize hash chain with last record
            await self._initialize_hash_chain()
            
            logger.info("Audit Buffer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Audit Buffer: {e}")
            raise
    
    async def start(self):
        """Start audit buffer"""
        if self.is_running:
            return
            
        logger.info("Starting Audit Buffer...")
        self.is_running = True
        
        # Start background tasks
        self.processing_task = asyncio.create_task(self._processing_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Audit Buffer started")
    
    async def stop(self):
        """Stop audit buffer"""
        if not self.is_running:
            return
            
        logger.info("Stopping Audit Buffer...")
        self.is_running = False
        
        # Cancel tasks
        if self.processing_task:
            self.processing_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Wait for tasks to complete
        tasks = [self.processing_task, self.cleanup_task]
        tasks = [task for task in tasks if task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Close database
        await self.database.close()
        
        logger.info("Audit Buffer stopped")
    
    async def store_attestation(self, attestation: TEEAttestation):
        """Store TEE attestation as audit record"""
        try:
            record_data = {
                'attestation_id': attestation.attestation_id,
                'payment_proof_id': attestation.payment_proof_id,
                'verification_result': attestation.verification_result,
                'amount': attestation.amount,
                'tee_signature': attestation.tee_signature.hex(),
                'has_sealed_evidence': attestation.sealed_evidence is not None
            }
            
            await self._store_audit_record(
                record_type='attestation',
                ephemeral_token_id=attestation.ephemeral_token_id,
                data=record_data
            )
            
            logger.info(f"Stored attestation {attestation.attestation_id}")
            
        except Exception as e:
            logger.error(f"Error storing attestation: {e}")
    
    async def store_token_event(self, event_type: str, token_id: str, 
                               event_data: Dict[str, Any]):
        """Store token-related event"""
        try:
            record_data = {
                'event_type': event_type,
                'event_data': event_data
            }
            
            await self._store_audit_record(
                record_type='token_event',
                ephemeral_token_id=token_id,
                data=record_data
            )
            
            logger.debug(f"Stored token event: {event_type} for {token_id}")
            
        except Exception as e:
            logger.error(f"Error storing token event: {e}")
    
    async def store_system_event(self, event_type: str, event_data: Dict[str, Any]):
        """Store system-level event"""
        try:
            record_data = {
                'event_type': event_type,
                'event_data': event_data
            }
            
            await self._store_audit_record(
                record_type='system_event',
                ephemeral_token_id=None,
                data=record_data
            )
            
            logger.debug(f"Stored system event: {event_type}")
            
        except Exception as e:
            logger.error(f"Error storing system event: {e}")
    
    async def _store_audit_record(self, record_type: str, 
                                 ephemeral_token_id: Optional[str],
                                 data: Dict[str, Any]):
        """Store audit record with hash chain"""
        import secrets
        
        record_id = secrets.token_urlsafe(16)
        timestamp = datetime.now()
        
        # Compute hash chain
        prev_hash = self.hash_chain.last_hash
        
        record = AuditRecord(
            record_id=record_id,
            record_type=record_type,
            timestamp=timestamp,
            rsu_id=self.rsu_id,
            ephemeral_token_id=ephemeral_token_id,
            data=data,
            hash_chain_prev=prev_hash,
            hash_chain_current="",  # Will be computed
            submitted_to_ledger=False,
            submission_timestamp=None
        )
        
        # Compute current hash
        current_hash = self.hash_chain.compute_record_hash(record, prev_hash)
        record.hash_chain_current = current_hash
        
        # Store in database
        await self.database.store_record(record)
        
        # Update hash chain
        self.hash_chain.update_last_hash(current_hash)
        
        self.stats['records_stored'] += 1
    
    async def _initialize_hash_chain(self):
        """Initialize hash chain from last stored record"""
        try:
            # TODO: Get last record from database and initialize hash chain
            # For now, start with genesis hash
            self.hash_chain.last_hash = "0" * 64
            
            logger.debug("Hash chain initialized")
            
        except Exception as e:
            logger.error(f"Error initializing hash chain: {e}")
    
    async def _processing_loop(self):
        """Background processing loop for batch submission"""
        while self.is_running:
            try:
                # Check if batch should be submitted
                if await self._should_submit_batch():
                    await self._submit_batch()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in audit processing loop: {e}")
                await asyncio.sleep(60)
    
    async def _should_submit_batch(self) -> bool:
        """Determine if batch should be submitted"""
        # Get unsubmitted records count
        unsubmitted_records = await self.database.get_unsubmitted_records(limit=1)
        
        if not unsubmitted_records:
            return False
        
        # Check batch size threshold
        all_unsubmitted = await self.database.get_unsubmitted_records()
        if len(all_unsubmitted) >= self.batch_size:
            return True
        
        # Check time threshold
        if datetime.now() - self.last_batch_time >= self.batch_interval:
            return True
        
        return False
    
    async def _submit_batch(self):
        """Submit batch of records to ledger"""
        try:
            logger.info("Submitting audit batch to ledger...")
            
            # Get records to submit
            records = await self.database.get_unsubmitted_records(limit=self.batch_size)
            
            if not records:
                return
            
            # Verify hash chain integrity
            if not self.hash_chain.verify_hash_chain(records):
                logger.error("Hash chain integrity check failed!")
                return
            
            # Create batch
            import secrets
            batch_id = secrets.token_urlsafe(16)
            
            # TODO: Submit to actual distributed ledger
            # For now, simulate submission
            await asyncio.sleep(0.5)  # Simulate network latency
            
            # Mark records as submitted
            record_ids = [record.record_id for record in records]
            await self.database.mark_records_submitted(record_ids, batch_id)
            
            self.stats['records_submitted'] += len(records)
            self.stats['batches_submitted'] += 1
            self.last_batch_time = datetime.now()
            
            logger.info(f"Submitted batch {batch_id} with {len(records)} records")
            
        except Exception as e:
            logger.error(f"Error submitting audit batch: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.is_running:
            try:
                # Clean up old records
                cleanup_age = timedelta(days=self.config.get('cleanup_age_days', 30))
                cutoff_time = datetime.now() - cleanup_age
                
                deleted_count = await self.database.cleanup_old_records(cutoff_time)
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old audit records")
                    self.stats['cleanup_operations'] += 1
                
                # Sleep for cleanup interval
                cleanup_interval = self.config.get('cleanup_interval_hours', 24)
                await asyncio.sleep(cleanup_interval * 3600)
                
            except Exception as e:
                logger.error(f"Error in audit cleanup loop: {e}")
                await asyncio.sleep(3600)  # Retry after 1 hour
    
    async def get_token_audit_trail(self, ephemeral_token_id: str) -> List[AuditRecord]:
        """Get complete audit trail for a token"""
        try:
            records = await self.database.get_records_by_token(ephemeral_token_id)
            
            # Verify hash chain for these records
            if records and not self.hash_chain.verify_hash_chain(records):
                logger.warning(f"Hash chain verification failed for token {ephemeral_token_id}")
            
            self.stats['hash_chain_verifications'] += 1
            
            return records
            
        except Exception as e:
            logger.error(f"Error getting audit trail: {e}")
            return []
    
    def get_audit_stats(self) -> Dict[str, Any]:
        """Get audit buffer statistics"""
        return {
            **self.stats,
            'rsu_id': self.rsu_id,
            'current_hash': self.hash_chain.last_hash[:16] + "...",
            'batch_size': self.batch_size,
            'batch_interval_seconds': self.batch_interval.total_seconds()
        }
