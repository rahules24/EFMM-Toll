"""
Audit Trail Manager
Manages audit trail queries and verification for the distributed ledger
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from ledger_node import AuditRecord, Block

logger = logging.getLogger(__name__)


@dataclass
class AuditQuery:
    """Audit query structure"""
    query_id: str
    entity_id: Optional[str] = None
    record_type: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    action: Optional[str] = None


class AuditTrailManager:
    """Manager for audit trail queries and verification"""
    
    def __init__(self, config: Dict[str, Any], ledger_node):
        self.config = config
        self.ledger_node = ledger_node
        self.query_cache: Dict[str, List[AuditRecord]] = {}
        self.is_running = False
    
    async def initialize(self):
        """Initialize audit trail manager"""
        logger.info("Initializing Audit Trail Manager...")
        self.is_running = True
    
    async def start(self):
        """Start audit trail manager"""
        logger.info("Starting Audit Trail Manager...")
        
        # Start cache cleanup task
        asyncio.create_task(self.cleanup_query_cache())
    
    async def stop(self):
        """Stop audit trail manager"""
        logger.info("Stopping Audit Trail Manager...")
        self.is_running = False
    
    async def query_audit_records(self, query: AuditQuery) -> List[AuditRecord]:
        """Query audit records based on criteria"""
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query)
            if cache_key in self.query_cache:
                logger.info(f"Returning cached results for query: {query.query_id}")
                return self.query_cache[cache_key]
            
            # Search through blockchain
            matching_records = []
            
            for block in self.ledger_node.blockchain:
                for record in block.records:
                    if self._matches_query(record, query):
                        matching_records.append(record)
            
            # Cache results
            self.query_cache[cache_key] = matching_records
            
            logger.info(f"Found {len(matching_records)} records for query: {query.query_id}")
            return matching_records
            
        except Exception as e:
            logger.error(f"Error querying audit records: {e}")
            return []
    
    async def verify_audit_trail(self, entity_id: str, start_time: str, end_time: str) -> Dict[str, Any]:
        """Verify audit trail integrity for an entity"""
        try:
            query = AuditQuery(
                query_id=f"verify_{entity_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                entity_id=entity_id,
                start_time=start_time,
                end_time=end_time
            )
            
            records = await self.query_audit_records(query)
            
            # Verify record integrity
            integrity_results = []
            for record in records:
                expected_hash = record.calculate_hash()
                is_valid = expected_hash == record.hash_value
                
                integrity_results.append({
                    'record_hash': record.hash_value,
                    'expected_hash': expected_hash,
                    'is_valid': is_valid,
                    'timestamp': record.timestamp
                })
            
            # Calculate overall integrity score
            valid_records = sum(1 for result in integrity_results if result['is_valid'])
            integrity_score = valid_records / len(records) if records else 1.0
            
            verification_result = {
                'entity_id': entity_id,
                'verification_timestamp': datetime.now().isoformat(),
                'total_records': len(records),
                'valid_records': valid_records,
                'integrity_score': integrity_score,
                'details': integrity_results
            }
            
            logger.info(f"Audit trail verification for {entity_id}: {integrity_score:.2%} integrity")
            return verification_result
            
        except Exception as e:
            logger.error(f"Error verifying audit trail for {entity_id}: {e}")
            return {}
    
    async def get_entity_timeline(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get chronological timeline of entity activities"""
        try:
            query = AuditQuery(
                query_id=f"timeline_{entity_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                entity_id=entity_id
            )
            
            records = await self.query_audit_records(query)
            
            # Sort by timestamp
            sorted_records = sorted(records, key=lambda r: r.timestamp)
            
            # Create timeline
            timeline = []
            for record in sorted_records:
                timeline.append({
                    'timestamp': record.timestamp,
                    'action': record.action,
                    'record_type': record.record_type,
                    'details': record.details,
                    'hash': record.hash_value
                })
            
            logger.info(f"Generated timeline for {entity_id} with {len(timeline)} events")
            return timeline
            
        except Exception as e:
            logger.error(f"Error generating timeline for {entity_id}: {e}")
            return []
    
    async def cleanup_query_cache(self):
        """Clean up old cached queries"""
        while self.is_running:
            try:
                # Remove cache entries older than 1 hour
                cache_timeout = self.config.get('cache_timeout', 3600)
                
                # TODO: Implement proper cache expiry based on timestamps
                if len(self.query_cache) > 100:  # Simple size-based cleanup
                    # Remove oldest entries
                    keys_to_remove = list(self.query_cache.keys())[:50]
                    for key in keys_to_remove:
                        del self.query_cache[key]
                
                await asyncio.sleep(300)  # Clean every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                await asyncio.sleep(300)
    
    def _matches_query(self, record: AuditRecord, query: AuditQuery) -> bool:
        """Check if record matches query criteria"""
        # Entity ID filter
        if query.entity_id and record.entity_id != query.entity_id:
            return False
        
        # Record type filter
        if query.record_type and record.record_type != query.record_type:
            return False
        
        # Action filter
        if query.action and record.action != query.action:
            return False
        
        # Time range filter
        if query.start_time or query.end_time:
            record_time = datetime.fromisoformat(record.timestamp.replace('Z', '+00:00'))
            
            if query.start_time:
                start_time = datetime.fromisoformat(query.start_time.replace('Z', '+00:00'))
                if record_time < start_time:
                    return False
            
            if query.end_time:
                end_time = datetime.fromisoformat(query.end_time.replace('Z', '+00:00'))
                if record_time > end_time:
                    return False
        
        return True
    
    def _generate_cache_key(self, query: AuditQuery) -> str:
        """Generate cache key for query"""
        key_parts = [
            query.entity_id or "all",
            query.record_type or "all",
            query.action or "all",
            query.start_time or "all",
            query.end_time or "all"
        ]
        return "_".join(key_parts)
