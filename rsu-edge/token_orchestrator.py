"""
Token Orchestrator for RSU Edge Module
Manages ephemeral token issuance, validation, and lifecycle

Key features:
- Ephemeral token generation and validation
- V2X/DSRC handshake management
- Token lifecycle management
- Anti-replay and security measures
- Integration with payment verification
"""

import asyncio
import logging
import secrets
import hashlib
import hmac
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import json

logger = logging.getLogger(__name__)


@dataclass
class EphemeralToken:
    """Ephemeral token for vehicle identification"""
    token_id: str
    vehicle_pseudonym: Optional[str]
    issued_at: datetime
    expires_at: datetime
    lane_id: Optional[str]
    rsu_id: str
    token_secret: bytes
    handshake_method: str  # 'v2x', 'rfid', 'qr', 'ble'
    status: str  # 'active', 'used', 'expired', 'revoked'
    metadata: Dict[str, Any]


@dataclass
class TokenValidationResult:
    """Result of token validation"""
    is_valid: bool
    token: Optional[EphemeralToken]
    error_message: Optional[str]
    validation_timestamp: datetime


@dataclass
class HandshakeRequest:
    """Vehicle handshake request for token issuance"""
    request_id: str
    timestamp: datetime
    handshake_method: str
    vehicle_pseudonym: Optional[str]
    challenge_response: Optional[str]
    lane_id: Optional[str]
    metadata: Dict[str, Any]


class TokenCrypto:
    """Cryptographic utilities for token management"""
    
    def __init__(self, master_key: bytes):
        self.master_key = master_key
    
    def generate_token_secret(self, token_id: str, timestamp: datetime) -> bytes:
        """Generate cryptographic secret for token"""
        key_material = f"{token_id}:{timestamp.isoformat()}".encode()
        return hmac.new(self.master_key, key_material, hashlib.sha256).digest()
    
    def validate_token_secret(self, token_id: str, timestamp: datetime, 
                             provided_secret: bytes) -> bool:
        """Validate token secret"""
        expected_secret = self.generate_token_secret(token_id, timestamp)
        return hmac.compare_digest(expected_secret, provided_secret)
    
    def generate_challenge(self) -> str:
        """Generate challenge for handshake authentication"""
        return secrets.token_urlsafe(32)
    
    def verify_challenge_response(self, challenge: str, response: str, 
                                 shared_secret: bytes) -> bool:
        """Verify challenge-response authentication"""
        expected_response = hmac.new(
            shared_secret, 
            challenge.encode(), 
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected_response, response)


class V2XHandshakeManager:
    """Handles V2X/DSRC handshake protocol for token issuance"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_handshakes: Dict[str, HandshakeRequest] = {}
        self.handshake_timeout = timedelta(seconds=config.get('handshake_timeout_seconds', 30))
    
    async def initiate_handshake(self, vehicle_pseudonym: str, 
                                lane_id: Optional[str] = None) -> str:
        """Initiate V2X handshake with vehicle"""
        request_id = secrets.token_urlsafe(16)
        
        handshake_request = HandshakeRequest(
            request_id=request_id,
            timestamp=datetime.now(),
            handshake_method='v2x',
            vehicle_pseudonym=vehicle_pseudonym,
            challenge_response=None,
            lane_id=lane_id,
            metadata={}
        )
        
        self.active_handshakes[request_id] = handshake_request
        
        # TODO: Send V2X handshake initiation message to vehicle
        logger.debug(f"Initiated V2X handshake {request_id} with vehicle {vehicle_pseudonym}")
        
        return request_id
    
    async def complete_handshake(self, request_id: str, 
                                challenge_response: str) -> bool:
        """Complete V2X handshake with challenge response"""
        if request_id not in self.active_handshakes:
            logger.warning(f"Unknown handshake request: {request_id}")
            return False
        
        handshake = self.active_handshakes[request_id]
        
        # Check timeout
        if datetime.now() - handshake.timestamp > self.handshake_timeout:
            logger.warning(f"Handshake {request_id} timed out")
            del self.active_handshakes[request_id]
            return False
        
        # TODO: Verify challenge response with vehicle's shared secret
        # For now, simulate successful verification
        handshake.challenge_response = challenge_response
        
        logger.info(f"V2X handshake {request_id} completed successfully")
        return True
    
    def cleanup_expired_handshakes(self):
        """Clean up expired handshake requests"""
        current_time = datetime.now()
        expired_requests = []
        
        for request_id, handshake in self.active_handshakes.items():
            if current_time - handshake.timestamp > self.handshake_timeout:
                expired_requests.append(request_id)
        
        for request_id in expired_requests:
            del self.active_handshakes[request_id]
            logger.debug(f"Cleaned up expired handshake {request_id}")


class QRHandshakeManager:
    """Handles QR code handshake for legacy vehicles"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.qr_display_timeout = timedelta(seconds=config.get('qr_display_timeout_seconds', 60))
        self.active_qr_codes: Dict[str, datetime] = {}
    
    async def generate_qr_handshake(self, lane_id: Optional[str] = None) -> str:
        """Generate QR code for vehicle handshake"""
        qr_id = secrets.token_urlsafe(16)
        
        # QR code contains handshake endpoint and temporary ID
        qr_data = {
            'handshake_id': qr_id,
            'rsu_endpoint': self.config.get('rsu_endpoint'),
            'expires_at': (datetime.now() + self.qr_display_timeout).isoformat(),
            'lane_id': lane_id
        }
        
        qr_string = json.dumps(qr_data)
        self.active_qr_codes[qr_id] = datetime.now()
        
        # TODO: Display QR code on RSU display/screen
        logger.info(f"Generated QR handshake code: {qr_id}")
        
        return qr_string
    
    async def process_qr_handshake(self, qr_id: str, 
                                  vehicle_data: Dict[str, Any]) -> bool:
        """Process handshake from QR code scan"""
        if qr_id not in self.active_qr_codes:
            logger.warning(f"Unknown QR handshake ID: {qr_id}")
            return False
        
        # Check timeout
        qr_timestamp = self.active_qr_codes[qr_id]
        if datetime.now() - qr_timestamp > self.qr_display_timeout:
            logger.warning(f"QR handshake {qr_id} expired")
            del self.active_qr_codes[qr_id]
            return False
        
        # Validate vehicle data
        # TODO: Implement proper vehicle data validation
        
        del self.active_qr_codes[qr_id]
        logger.info(f"QR handshake {qr_id} processed successfully")
        
        return True


class TokenOrchestrator:
    """
    Token Orchestrator
    
    Manages the complete lifecycle of ephemeral tokens including:
    - Token issuance through various handshake methods
    - Token validation and verification
    - Token expiration and cleanup
    - Integration with payment verification
    """
    
    def __init__(self, config: Dict[str, Any], event_queue: asyncio.Queue):
        self.config = config
        self.event_queue = event_queue
        self.rsu_id = config['rsu_id']
        self.is_running = False
        
        # Cryptography
        master_key = config.get('master_key', secrets.token_bytes(32))
        self.token_crypto = TokenCrypto(master_key)
        
        # Handshake managers
        self.v2x_manager = V2XHandshakeManager(config.get('v2x', {}))
        self.qr_manager = QRHandshakeManager(config.get('qr', {}))
        
        # Token storage and management
        self.active_tokens: Dict[str, EphemeralToken] = {}
        self.token_history: Dict[str, EphemeralToken] = {}  # For audit purposes
        self.used_token_ids: Set[str] = set()  # Anti-replay protection
        
        # Configuration
        self.token_validity_duration = timedelta(
            seconds=config.get('token_validity_seconds', 300)  # 5 minutes
        )
        self.max_tokens_per_lane = config.get('max_tokens_per_lane', 10)
        
        # Background tasks
        self.cleanup_task = None
        self.monitoring_task = None
        
        # Statistics
        self.stats = {
            'tokens_issued': 0,
            'tokens_validated': 0,
            'tokens_expired': 0,
            'handshakes_completed': 0,
            'validation_failures': 0
        }
    
    async def initialize(self):
        """Initialize token orchestrator"""
        logger.info(f"Initializing Token Orchestrator for RSU {self.rsu_id}")
        
        try:
            # Initialize handshake managers
            # TODO: Initialize V2X transceiver connections
            # TODO: Initialize QR display hardware
            
            logger.info("Token Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Token Orchestrator: {e}")
            raise
    
    async def start(self):
        """Start token orchestrator"""
        if self.is_running:
            return
            
        logger.info("Starting Token Orchestrator...")
        self.is_running = True
        
        # Start background tasks
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Token Orchestrator started")
    
    async def stop(self):
        """Stop token orchestrator"""
        if not self.is_running:
            return
            
        logger.info("Stopping Token Orchestrator...")
        self.is_running = False
        
        # Cancel tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        # Wait for tasks to complete
        tasks = [self.cleanup_task, self.monitoring_task]
        tasks = [task for task in tasks if task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Token Orchestrator stopped")
    
    async def issue_token(self, handshake_request: HandshakeRequest) -> Optional[EphemeralToken]:
        """Issue ephemeral token after successful handshake"""
        try:
            # Generate unique token ID
            token_id = self._generate_token_id()
            
            # Check for replay attacks
            if token_id in self.used_token_ids:
                logger.warning(f"Token ID replay detected: {token_id}")
                return None
            
            # Check lane capacity
            if self._check_lane_capacity(handshake_request.lane_id):
                logger.warning(f"Lane capacity exceeded for {handshake_request.lane_id}")
                return None
            
            # Create token
            now = datetime.now()
            token = EphemeralToken(
                token_id=token_id,
                vehicle_pseudonym=handshake_request.vehicle_pseudonym,
                issued_at=now,
                expires_at=now + self.token_validity_duration,
                lane_id=handshake_request.lane_id,
                rsu_id=self.rsu_id,
                token_secret=self.token_crypto.generate_token_secret(token_id, now),
                handshake_method=handshake_request.handshake_method,
                status='active',
                metadata=handshake_request.metadata.copy()
            )
            
            # Store token
            self.active_tokens[token_id] = token
            self.used_token_ids.add(token_id)
            
            # Update statistics
            self.stats['tokens_issued'] += 1
            
            logger.info(f"Issued token {token_id} for vehicle {handshake_request.vehicle_pseudonym}")
            
            # Notify event queue
            await self._notify_token_issued(token)
            
            return token
            
        except Exception as e:
            logger.error(f"Error issuing token: {e}")
            return None
    
    async def validate_token(self, token_id: str, 
                           provided_secret: Optional[bytes] = None) -> TokenValidationResult:
        """Validate ephemeral token"""
        try:
            validation_time = datetime.now()
            
            # Check if token exists
            if token_id not in self.active_tokens:
                return TokenValidationResult(
                    is_valid=False,
                    token=None,
                    error_message="Token not found",
                    validation_timestamp=validation_time
                )
            
            token = self.active_tokens[token_id]
            
            # Check token status
            if token.status != 'active':
                return TokenValidationResult(
                    is_valid=False,
                    token=token,
                    error_message=f"Token status is {token.status}",
                    validation_timestamp=validation_time
                )
            
            # Check expiration
            if validation_time > token.expires_at:
                token.status = 'expired'
                return TokenValidationResult(
                    is_valid=False,
                    token=token,
                    error_message="Token expired",
                    validation_timestamp=validation_time
                )
            
            # Validate cryptographic secret if provided
            if provided_secret:
                if not self.token_crypto.validate_token_secret(
                    token_id, token.issued_at, provided_secret
                ):
                    self.stats['validation_failures'] += 1
                    return TokenValidationResult(
                        is_valid=False,
                        token=token,
                        error_message="Invalid token secret",
                        validation_timestamp=validation_time
                    )
            
            # Token is valid
            self.stats['tokens_validated'] += 1
            
            return TokenValidationResult(
                is_valid=True,
                token=token,
                error_message=None,
                validation_timestamp=validation_time
            )
            
        except Exception as e:
            logger.error(f"Error validating token {token_id}: {e}")
            return TokenValidationResult(
                is_valid=False,
                token=None,
                error_message=f"Validation error: {e}",
                validation_timestamp=datetime.now()
            )
    
    async def mark_token_used(self, token_id: str) -> bool:
        """Mark token as used after successful toll processing"""
        try:
            if token_id not in self.active_tokens:
                logger.warning(f"Attempted to mark unknown token as used: {token_id}")
                return False
            
            token = self.active_tokens[token_id]
            token.status = 'used'
            
            # Move to history for audit purposes
            self.token_history[token_id] = token
            
            logger.info(f"Token {token_id} marked as used")
            
            # Notify event queue
            await self._notify_token_used(token)
            
            return True
            
        except Exception as e:
            logger.error(f"Error marking token as used: {e}")
            return False
    
    async def revoke_token(self, token_id: str, reason: str) -> bool:
        """Revoke token (security measure)"""
        try:
            if token_id not in self.active_tokens:
                logger.warning(f"Attempted to revoke unknown token: {token_id}")
                return False
            
            token = self.active_tokens[token_id]
            token.status = 'revoked'
            token.metadata['revocation_reason'] = reason
            token.metadata['revoked_at'] = datetime.now().isoformat()
            
            logger.warning(f"Token {token_id} revoked: {reason}")
            
            # Notify event queue
            await self._notify_token_revoked(token, reason)
            
            return True
            
        except Exception as e:
            logger.error(f"Error revoking token: {e}")
            return False
    
    def _generate_token_id(self) -> str:
        """Generate unique token ID"""
        # Include RSU ID and timestamp for uniqueness
        timestamp_ms = int(time.time() * 1000)
        random_part = secrets.token_urlsafe(16)
        return f"{self.rsu_id}_{timestamp_ms}_{random_part}"
    
    def _check_lane_capacity(self, lane_id: Optional[str]) -> bool:
        """Check if lane has reached token capacity"""
        if lane_id is None:
            return False
        
        lane_token_count = sum(
            1 for token in self.active_tokens.values()
            if token.lane_id == lane_id and token.status == 'active'
        )
        
        return lane_token_count >= self.max_tokens_per_lane
    
    async def _cleanup_loop(self):
        """Background task to clean up expired tokens"""
        while self.is_running:
            try:
                current_time = datetime.now()
                expired_tokens = []
                
                # Find expired tokens
                for token_id, token in self.active_tokens.items():
                    if current_time > token.expires_at and token.status == 'active':
                        expired_tokens.append(token_id)
                
                # Clean up expired tokens
                for token_id in expired_tokens:
                    token = self.active_tokens[token_id]
                    token.status = 'expired'
                    self.token_history[token_id] = token
                    del self.active_tokens[token_id]
                    self.stats['tokens_expired'] += 1
                    
                    logger.debug(f"Cleaned up expired token {token_id}")
                
                # Clean up expired handshakes
                self.v2x_manager.cleanup_expired_handshakes()
                
                await asyncio.sleep(30)  # Cleanup every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(30)
    
    async def _monitoring_loop(self):
        """Background task for monitoring and metrics"""
        while self.is_running:
            try:
                # Log statistics periodically
                logger.info(f"Token stats: {self.stats}")
                logger.info(f"Active tokens: {len(self.active_tokens)}")
                
                await asyncio.sleep(300)  # Log every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(300)
    
    async def _notify_token_issued(self, token: EphemeralToken):
        """Notify event queue of token issuance"""
        try:
            event = {
                'type': 'token_issued',
                'token_id': token.token_id,
                'timestamp': token.issued_at,
                'vehicle_pseudonym': token.vehicle_pseudonym,
                'lane_id': token.lane_id
            }
            await self.event_queue.put(event)
        except Exception as e:
            logger.error(f"Error notifying token issuance: {e}")
    
    async def _notify_token_used(self, token: EphemeralToken):
        """Notify event queue of token usage"""
        try:
            event = {
                'type': 'token_used',
                'token_id': token.token_id,
                'timestamp': datetime.now(),
                'vehicle_pseudonym': token.vehicle_pseudonym,
                'lane_id': token.lane_id
            }
            await self.event_queue.put(event)
        except Exception as e:
            logger.error(f"Error notifying token usage: {e}")
    
    async def _notify_token_revoked(self, token: EphemeralToken, reason: str):
        """Notify event queue of token revocation"""
        try:
            event = {
                'type': 'token_revoked',
                'token_id': token.token_id,
                'timestamp': datetime.now(),
                'reason': reason,
                'vehicle_pseudonym': token.vehicle_pseudonym,
                'lane_id': token.lane_id
            }
            await self.event_queue.put(event)
        except Exception as e:
            logger.error(f"Error notifying token revocation: {e}")
    
    def get_token_stats(self) -> Dict[str, Any]:
        """Get token management statistics"""
        return {
            **self.stats,
            'active_tokens': len(self.active_tokens),
            'token_history_size': len(self.token_history),
            'rsu_id': self.rsu_id
        }
    
    def get_active_tokens(self) -> List[Dict[str, Any]]:
        """Get list of active tokens (for debugging/monitoring)"""
        return [
            {
                'token_id': token.token_id,
                'vehicle_pseudonym': token.vehicle_pseudonym,
                'issued_at': token.issued_at.isoformat(),
                'expires_at': token.expires_at.isoformat(),
                'lane_id': token.lane_id,
                'status': token.status,
                'handshake_method': token.handshake_method
            }
            for token in self.active_tokens.values()
        ]
