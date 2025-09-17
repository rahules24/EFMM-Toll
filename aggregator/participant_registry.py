"""
Participant Registry
Manages registration and tracking of federated learning participants
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ParticipantRegistry:
    """Registry for federated learning participants (RSUs and vehicles)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_running = False
        self.participants: Dict[str, Dict[str, Any]] = {}
        self.participant_timeout = timedelta(minutes=config.get('timeout_minutes', 10))
    
    async def initialize(self):
        """Initialize participant registry"""
        logger.info("Initializing Participant Registry...")
        self.is_running = True
    
    async def start(self):
        """Start participant registry"""
        pass
    
    async def stop(self):
        """Stop participant registry"""
        self.is_running = False
    
    async def register_participant(self, participant_id: str, participant_info: Dict[str, Any]) -> bool:
        """Register a new participant"""
        try:
            self.participants[participant_id] = {
                **participant_info,
                'registered_at': datetime.now(),
                'last_seen': datetime.now(),
                'status': 'active'
            }
            
            logger.info(f"Registered participant: {participant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering participant {participant_id}: {e}")
            return False
    
    async def get_active_participants(self) -> List[Dict[str, Any]]:
        """Get list of active participants"""
        current_time = datetime.now()
        active_participants = []
        
        for participant_id, info in self.participants.items():
            if (current_time - info['last_seen']) <= self.participant_timeout:
                active_participants.append({
                    'participant_id': participant_id,
                    **info
                })
        
        return active_participants
    
    async def update_participant_heartbeat(self, participant_id: str):
        """Update participant last seen timestamp"""
        if participant_id in self.participants:
            self.participants[participant_id]['last_seen'] = datetime.now()
