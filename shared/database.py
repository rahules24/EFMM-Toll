"""
Database Utilities
Shared database operations for EFMM-Toll system
"""

import asyncio
import sqlite3
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from contextlib import asynccontextmanager
import json

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Async database manager for SQLite operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection_pool: List[sqlite3.Connection] = []
        self.pool_size = 5
        self.lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize database connection pool"""
        logger.info(f"Initializing database: {self.db_path}")
        
        for _ in range(self.pool_size):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            self.connection_pool.append(conn)
        
        # Create base tables
        await self.create_base_tables()
    
    async def close(self):
        """Close all database connections"""
        for conn in self.connection_pool:
            conn.close()
        self.connection_pool.clear()
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        async with self.lock:
            if self.connection_pool:
                conn = self.connection_pool.pop()
            else:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
        
        try:
            yield conn
        finally:
            async with self.lock:
                self.connection_pool.append(conn)
    
    async def execute_query(self, query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """Execute SELECT query and return results"""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    async def execute_update(self, query: str, params: Tuple = ()) -> int:
        """Execute INSERT/UPDATE/DELETE query and return affected rows"""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount
    
    async def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """Execute multiple queries with different parameters"""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
    
    async def create_base_tables(self):
        """Create base tables used across modules"""
        
        # Audit records table
        audit_table = """
        CREATE TABLE IF NOT EXISTS audit_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT,  -- JSON string
            timestamp TEXT NOT NULL,
            hash_value TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Transactions table
        transactions_table = """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT UNIQUE NOT NULL,
            vehicle_id TEXT NOT NULL,
            rsu_id TEXT NOT NULL,
            amount REAL NOT NULL,
            token_id TEXT,
            proof_hash TEXT,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Cache table for temporary data
        cache_table = """
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT,  -- JSON string
            expires_at DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Entity registry table
        entities_table = """
        CREATE TABLE IF NOT EXISTS entities (
            entity_id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,  -- 'rsu', 'vehicle', 'aggregator'
            public_key TEXT,
            status TEXT DEFAULT 'active',
            metadata TEXT,  -- JSON string
            last_seen DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        tables = [audit_table, transactions_table, cache_table, entities_table]
        
        for table_sql in tables:
            await self.execute_update(table_sql)
        
        logger.info("Base tables created successfully")


class AuditLogger:
    """Database audit logger"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def log_audit_record(self, record_type: str, entity_id: str, 
                              action: str, details: Dict[str, Any]) -> bool:
        """Log audit record to database"""
        try:
            from shared.crypto_utils import CryptoUtils
            
            details_json = json.dumps(details)
            timestamp = datetime.now().isoformat()
            
            # Calculate hash for integrity
            hash_data = f"{record_type}:{entity_id}:{action}:{details_json}:{timestamp}"
            hash_value = CryptoUtils.hash_data(hash_data.encode())
            
            query = """
            INSERT INTO audit_records (record_type, entity_id, action, details, timestamp, hash_value)
            VALUES (?, ?, ?, ?, ?, ?)
            """
            
            await self.db_manager.execute_update(
                query, 
                (record_type, entity_id, action, details_json, timestamp, hash_value)
            )
            
            logger.debug(f"Logged audit record: {record_type} for {entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging audit record: {e}")
            return False
    
    async def get_audit_records(self, entity_id: Optional[str] = None, 
                               record_type: Optional[str] = None,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit records with optional filters"""
        query = "SELECT * FROM audit_records WHERE 1=1"
        params = []
        
        if entity_id:
            query += " AND entity_id = ?"
            params.append(entity_id)
        
        if record_type:
            query += " AND record_type = ?"
            params.append(record_type)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        return await self.db_manager.execute_query(query, tuple(params))


class TransactionManager:
    """Database transaction manager"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def create_transaction(self, transaction_data: Dict[str, Any]) -> bool:
        """Create new transaction record"""
        try:
            query = """
            INSERT INTO transactions (transaction_id, vehicle_id, rsu_id, amount, 
                                    token_id, proof_hash, status, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            await self.db_manager.execute_update(query, (
                transaction_data['transaction_id'],
                transaction_data['vehicle_id'],
                transaction_data['rsu_id'],
                transaction_data['amount'],
                transaction_data.get('token_id'),
                transaction_data.get('proof_hash'),
                transaction_data['status'],
                transaction_data['timestamp']
            ))
            
            logger.info(f"Created transaction: {transaction_data['transaction_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating transaction: {e}")
            return False
    
    async def update_transaction_status(self, transaction_id: str, status: str) -> bool:
        """Update transaction status"""
        try:
            query = "UPDATE transactions SET status = ? WHERE transaction_id = ?"
            rows_affected = await self.db_manager.execute_update(query, (status, transaction_id))
            
            if rows_affected > 0:
                logger.info(f"Updated transaction {transaction_id} status to {status}")
                return True
            else:
                logger.warning(f"Transaction not found: {transaction_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating transaction status: {e}")
            return False
    
    async def get_transaction(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get transaction by ID"""
        query = "SELECT * FROM transactions WHERE transaction_id = ?"
        results = await self.db_manager.execute_query(query, (transaction_id,))
        return results[0] if results else None
