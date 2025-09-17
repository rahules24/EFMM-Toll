"""
V2X Client for Vehicle OBU
Handles Vehicle-to-Infrastructure (V2I) communication with RSUs

Key features:
- DSRC/C-V2X protocol support
- RSU discovery and handshake
- Message authentication and security
- Adaptive communication parameters
- Privacy-preserving pseudonym management
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import secrets
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class RSUInfo:
    """Information about discovered RSU"""
    rsu_id: str
    signal_strength: float
    last_seen: datetime
    capabilities: List[str]
    trust_score: float
    distance_estimate: Optional[float] = None
    lane_coverage: Optional[List[str]] = None


@dataclass
class V2XMessage:
    """V2X message structure"""
    message_type: str  # 'BSM', 'PSM', 'RSA', 'MAPEM', 'SPATEM', etc.
    sender_id: str
    timestamp: datetime
    payload: Dict[str, Any]
    signature: Optional[bytes] = None
    sequence_number: Optional[int] = None


@dataclass
class HandshakeSession:
    """Active handshake session with RSU"""
    session_id: str
    rsu_id: str
    started_at: datetime
    challenge: Optional[str]
    response: Optional[str]
    status: str  # 'initiated', 'challenged', 'completed', 'failed'
    ephemeral_key: Optional[bytes] = None


class V2XProtocolStack:
    """V2X protocol stack implementation"""
    
    def __init__(self, protocol_type: str, config: Dict[str, Any]):
        self.protocol_type = protocol_type  # 'DSRC' or 'C-V2X'
        self.config = config
        self.is_initialized = False
        
        # Communication parameters
        self.channel = config.get('channel', 178)  # Default DSRC channel
        self.tx_power = config.get('tx_power', 20)  # dBm
        self.data_rate = config.get('data_rate', 6)  # Mbps
        
        # Message queues
        self.tx_queue = asyncio.Queue()
        self.rx_queue = asyncio.Queue()
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'handshakes_completed': 0,
            'connection_errors': 0
        }
    
    async def initialize(self):
        """Initialize V2X protocol stack"""
        logger.info(f"Initializing {self.protocol_type} protocol stack...")
        
        try:
            if self.protocol_type == 'DSRC':
                await self._initialize_dsrc()
            elif self.protocol_type == 'C-V2X':
                await self._initialize_cv2x()
            else:
                raise ValueError(f"Unsupported protocol: {self.protocol_type}")
            
            self.is_initialized = True
            logger.info(f"{self.protocol_type} protocol stack initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.protocol_type}: {e}")
            raise
    
    async def _initialize_dsrc(self):
        """Initialize DSRC hardware/software stack"""
        # TODO: Initialize DSRC radio hardware
        # - Configure radio parameters (channel, power, etc.)
        # - Set up MAC layer
        # - Initialize security services
        logger.info("DSRC stack initialization (placeholder)")
    
    async def _initialize_cv2x(self):
        """Initialize C-V2X (Cellular V2X) stack"""
        # TODO: Initialize C-V2X modem
        # - Configure PC5 interface for direct communication
        # - Set up Uu interface for cellular communication
        # - Initialize security credentials
        logger.info("C-V2X stack initialization (placeholder)")
    
    async def send_message(self, message: V2XMessage, destination: Optional[str] = None):
        """Send V2X message"""
        try:
            # Add to transmission queue
            await self.tx_queue.put((message, destination))
            
        except Exception as e:
            logger.error(f"Error queuing V2X message: {e}")
            raise
    
    async def receive_message(self, timeout: float = 1.0) -> Optional[V2XMessage]:
        """Receive V2X message"""
        try:
            message = await asyncio.wait_for(self.rx_queue.get(), timeout=timeout)
            self.stats['messages_received'] += 1
            return message
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error receiving V2X message: {e}")
            raise
    
    async def start_communication(self):
        """Start V2X communication tasks"""
        # Start background tasks for TX/RX
        await asyncio.gather(
            self._tx_task(),
            self._rx_task(),
            return_exceptions=True
        )
    
    async def _tx_task(self):
        """Transmission task"""
        while self.is_initialized:
            try:
                message, destination = await self.tx_queue.get()
                
                # TODO: Transmit message via radio
                # For simulation, just log the message
                logger.debug(f"Transmitting {message.message_type} to {destination or 'broadcast'}")
                
                # Simulate transmission delay
                await asyncio.sleep(0.01)
                
                self.stats['messages_sent'] += 1
                
            except Exception as e:
                logger.error(f"Error in transmission task: {e}")
                self.stats['connection_errors'] += 1
    
    async def _rx_task(self):
        """Reception task"""
        while self.is_initialized:
            try:
                # TODO: Receive messages from radio
                # For simulation, create mock messages periodically
                
                await asyncio.sleep(0.5)  # Simulate reception interval
                
                # Simulate occasional message reception
                import random
                if random.random() < 0.1:  # 10% chance of message
                    mock_message = V2XMessage(
                        message_type='RSA',  # Road Side Alert
                        sender_id=f'rsu_{random.randint(1, 10):03d}',
                        timestamp=datetime.now(),
                        payload={
                            'alert_type': 'toll_point_ahead',
                            'distance': random.randint(100, 500)
                        }
                    )
                    
                    await self.rx_queue.put(mock_message)
                
            except Exception as e:
                logger.error(f"Error in reception task: {e}")
                self.stats['connection_errors'] += 1


class RSUDiscoveryManager:
    """Manages discovery and tracking of nearby RSUs"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.discovered_rsus: Dict[str, RSUInfo] = {}
        self.discovery_timeout = timedelta(seconds=config.get('discovery_timeout_seconds', 30))
        
    def process_rsu_announcement(self, message: V2XMessage) -> Optional[RSUInfo]:
        """Process RSU announcement message"""
        try:
            rsu_id = message.sender_id
            payload = message.payload
            
            # Extract RSU information
            signal_strength = payload.get('signal_strength', -50.0)
            capabilities = payload.get('capabilities', [])
            
            # Calculate trust score based on various factors
            trust_score = self._calculate_trust_score(rsu_id, payload)
            
            # Create or update RSU info
            rsu_info = RSUInfo(
                rsu_id=rsu_id,
                signal_strength=signal_strength,
                last_seen=message.timestamp,
                capabilities=capabilities,
                trust_score=trust_score,
                distance_estimate=payload.get('distance'),
                lane_coverage=payload.get('lane_coverage')
            )
            
            self.discovered_rsus[rsu_id] = rsu_info
            
            logger.info(f"Discovered/Updated RSU: {rsu_id}, trust: {trust_score:.2f}")
            
            return rsu_info
            
        except Exception as e:
            logger.error(f"Error processing RSU announcement: {e}")
            return None
    
    def _calculate_trust_score(self, rsu_id: str, payload: Dict[str, Any]) -> float:
        """Calculate trust score for RSU"""
        # TODO: Implement comprehensive trust scoring
        # Factors to consider:
        # - Signal strength and consistency
        # - Certificate validity
        # - Behavioral patterns
        # - Reputation from other vehicles
        # - Geographic consistency
        
        base_score = 0.5
        
        # Adjust based on signal strength
        signal_strength = payload.get('signal_strength', -60)
        if signal_strength > -40:
            base_score += 0.3
        elif signal_strength > -50:
            base_score += 0.2
        elif signal_strength > -60:
            base_score += 0.1
        
        # Adjust based on capabilities
        capabilities = payload.get('capabilities', [])
        if 'toll_collection' in capabilities:
            base_score += 0.1
        if 'secure_payment' in capabilities:
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def get_trusted_rsus(self, min_trust_score: float = 0.7) -> List[RSUInfo]:
        """Get RSUs above trust threshold"""
        trusted_rsus = []
        current_time = datetime.now()
        
        for rsu_info in self.discovered_rsus.values():
            # Check if RSU is still active
            if current_time - rsu_info.last_seen <= self.discovery_timeout:
                if rsu_info.trust_score >= min_trust_score:
                    trusted_rsus.append(rsu_info)
        
        return trusted_rsus
    
    def cleanup_expired_rsus(self):
        """Clean up expired RSU entries"""
        current_time = datetime.now()
        expired_rsus = []
        
        for rsu_id, rsu_info in self.discovered_rsus.items():
            if current_time - rsu_info.last_seen > self.discovery_timeout:
                expired_rsus.append(rsu_id)
        
        for rsu_id in expired_rsus:
            del self.discovered_rsus[rsu_id]
            logger.debug(f"Cleaned up expired RSU: {rsu_id}")


class V2XClient:
    """
    V2X Client for Vehicle OBU
    
    Manages all V2X communication including RSU discovery,
    handshake protocols, and message exchange.
    """
    
    def __init__(self, config: Dict[str, Any], event_queue: asyncio.Queue):
        self.config = config
        self.event_queue = event_queue
        self.is_active = False
        
        # Protocol stack
        protocol_type = config.get('protocol', 'DSRC')
        self.protocol_stack = V2XProtocolStack(protocol_type, config)
        
        # RSU management
        self.rsu_discovery = RSUDiscoveryManager(config.get('discovery', {}))
        
        # Handshake management
        self.active_handshakes: Dict[str, HandshakeSession] = {}
        self.handshake_timeout = timedelta(seconds=config.get('handshake_timeout_seconds', 30))
        
        # Message callbacks
        self.message_callbacks: Dict[str, List[Callable]] = {}
        
        # Background tasks
        self.communication_task = None
        self.discovery_task = None
        self.cleanup_task = None
        
        # Vehicle identity
        self.vehicle_pseudonym = None
        self.pseudonym_rotation_interval = timedelta(
            minutes=config.get('pseudonym_rotation_minutes', 5)
        )
        self.last_pseudonym_rotation = datetime.now()
    
    async def initialize(self):
        """Initialize V2X client"""
        logger.info("Initializing V2X Client...")
        
        try:
            # Initialize protocol stack
            await self.protocol_stack.initialize()
            
            # Generate initial pseudonym
            await self._rotate_pseudonym()
            
            logger.info("V2X Client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize V2X Client: {e}")
            raise
    
    async def start(self):
        """Start V2X client"""
        if self.is_active:
            return
            
        logger.info("Starting V2X Client...")
        self.is_active = True
        
        # Start background tasks
        self.communication_task = asyncio.create_task(self._communication_loop())
        self.discovery_task = asyncio.create_task(self._discovery_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Start protocol stack communication
        asyncio.create_task(self.protocol_stack.start_communication())
        
        logger.info("V2X Client started")
    
    async def stop(self):
        """Stop V2X client"""
        if not self.is_active:
            return
            
        logger.info("Stopping V2X Client...")
        self.is_active = False
        
        # Cancel tasks
        tasks = [self.communication_task, self.discovery_task, self.cleanup_task]
        for task in tasks:
            if task:
                task.cancel()
        
        # Wait for tasks to complete
        if tasks:
            await asyncio.gather(*[t for t in tasks if t], return_exceptions=True)
        
        logger.info("V2X Client stopped")
    
    async def complete_handshake(self, rsu_id: str, handshake_data: Dict[str, Any], 
                                pseudonym: str) -> bool:
        """Complete handshake with RSU"""
        try:
            session_id = f"{rsu_id}_{secrets.token_urlsafe(8)}"
            
            # Create handshake session
            session = HandshakeSession(
                session_id=session_id,
                rsu_id=rsu_id,
                started_at=datetime.now(),
                challenge=handshake_data.get('challenge'),
                response=None,
                status='challenged'
            )
            
            self.active_handshakes[session_id] = session
            
            # Generate response to challenge
            if session.challenge:
                # TODO: Generate proper cryptographic response
                response_data = hashlib.sha256(
                    f"{session.challenge}:{pseudonym}".encode()
                ).hexdigest()
                
                session.response = response_data
                
                # Send handshake response
                response_message = V2XMessage(
                    message_type='HANDSHAKE_RESPONSE',
                    sender_id=pseudonym,
                    timestamp=datetime.now(),
                    payload={
                        'session_id': session_id,
                        'challenge_response': response_data,
                        'vehicle_capabilities': self.config.get('capabilities', [])
                    }
                )
                
                await self.protocol_stack.send_message(response_message, rsu_id)
                
                session.status = 'completed'
                self.protocol_stack.stats['handshakes_completed'] += 1
                
                logger.info(f"Handshake completed with RSU: {rsu_id}")
                return True
            else:
                session.status = 'failed'
                logger.error(f"No challenge in handshake data from RSU: {rsu_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error completing handshake with RSU {rsu_id}: {e}")
            return False
    
    async def send_payment_proof(self, rsu_id: str, payment_proof: Dict[str, Any]):
        """Send payment proof to RSU"""
        try:
            payment_message = V2XMessage(
                message_type='PAYMENT_PROOF',
                sender_id=self.vehicle_pseudonym,
                timestamp=datetime.now(),
                payload={
                    'payment_proof': payment_proof,
                    'rsu_id': rsu_id
                }
            )
            
            await self.protocol_stack.send_message(payment_message, rsu_id)
            
            logger.info(f"Payment proof sent to RSU: {rsu_id}")
            
        except Exception as e:
            logger.error(f"Error sending payment proof to RSU {rsu_id}: {e}")
    
    async def _communication_loop(self):
        """Main communication loop"""
        while self.is_active:
            try:
                # Receive and process V2X messages
                message = await self.protocol_stack.receive_message(timeout=1.0)
                
                if message:
                    await self._process_received_message(message)
                
                # Check for pseudonym rotation
                await self._check_pseudonym_rotation()
                
            except Exception as e:
                logger.error(f"Error in communication loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_received_message(self, message: V2XMessage):
        """Process received V2X message"""
        try:
            logger.debug(f"Received {message.message_type} from {message.sender_id}")
            
            if message.message_type == 'RSA':  # Road Side Alert
                await self._handle_rsa_message(message)
            elif message.message_type == 'RSU_ANNOUNCEMENT':
                await self._handle_rsu_announcement(message)
            elif message.message_type == 'HANDSHAKE_INITIATION':
                await self._handle_handshake_initiation(message)
            elif message.message_type == 'TOKEN_OFFER':
                await self._handle_token_offer(message)
            elif message.message_type == 'PAYMENT_REQUEST':
                await self._handle_payment_request(message)
            
            # Call registered callbacks
            callbacks = self.message_callbacks.get(message.message_type, [])
            for callback in callbacks:
                try:
                    await callback(message)
                except Exception as e:
                    logger.error(f"Error in message callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _handle_rsa_message(self, message: V2XMessage):
        """Handle Road Side Alert message"""
        payload = message.payload
        alert_type = payload.get('alert_type')
        
        if alert_type == 'toll_point_ahead':
            distance = payload.get('distance', 0)
            logger.info(f"Toll point ahead: {distance}m")
            
            # Notify main application
            await self.event_queue.put({
                'type': 'toll_point_ahead',
                'rsu_id': message.sender_id,
                'distance': distance,
                'timestamp': message.timestamp
            })
    
    async def _handle_rsu_announcement(self, message: V2XMessage):
        """Handle RSU announcement message"""
        rsu_info = self.rsu_discovery.process_rsu_announcement(message)
        
        if rsu_info:
            # Notify main application of RSU discovery
            await self.event_queue.put({
                'type': 'rsu_discovered',
                'rsu_id': rsu_info.rsu_id,
                'rsu_info': rsu_info,
                'timestamp': message.timestamp
            })
    
    async def _handle_handshake_initiation(self, message: V2XMessage):
        """Handle handshake initiation from RSU"""
        await self.event_queue.put({
            'type': 'handshake_initiated',
            'rsu_id': message.sender_id,
            'handshake_data': message.payload,
            'timestamp': message.timestamp
        })
    
    async def _handle_token_offer(self, message: V2XMessage):
        """Handle token offer from RSU"""
        await self.event_queue.put({
            'type': 'token_offered',
            'rsu_id': message.sender_id,
            'token_data': message.payload,
            'timestamp': message.timestamp
        })
    
    async def _handle_payment_request(self, message: V2XMessage):
        """Handle payment request from RSU"""
        await self.event_queue.put({
            'type': 'payment_requested',
            'rsu_id': message.sender_id,
            'payment_request': message.payload,
            'timestamp': message.timestamp
        })
    
    async def _discovery_loop(self):
        """RSU discovery loop"""
        while self.is_active:
            try:
                # Send discovery beacon
                discovery_message = V2XMessage(
                    message_type='VEHICLE_BEACON',
                    sender_id=self.vehicle_pseudonym,
                    timestamp=datetime.now(),
                    payload={
                        'vehicle_type': 'passenger',
                        'capabilities': self.config.get('capabilities', []),
                        'seeking_services': ['toll_collection']
                    }
                )
                
                await self.protocol_stack.send_message(discovery_message)
                
                # Wait for discovery interval
                await asyncio.sleep(self.config.get('discovery_interval_seconds', 10))
                
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_loop(self):
        """Cleanup expired sessions and RSUs"""
        while self.is_active:
            try:
                # Clean up expired handshake sessions
                current_time = datetime.now()
                expired_sessions = []
                
                for session_id, session in self.active_handshakes.items():
                    if current_time - session.started_at > self.handshake_timeout:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.active_handshakes[session_id]
                    logger.debug(f"Cleaned up expired handshake session: {session_id}")
                
                # Clean up expired RSUs
                self.rsu_discovery.cleanup_expired_rsus()
                
                await asyncio.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def _rotate_pseudonym(self):
        """Rotate vehicle pseudonym for privacy"""
        self.vehicle_pseudonym = f"veh_{secrets.token_urlsafe(8)}"
        self.last_pseudonym_rotation = datetime.now()
        logger.debug(f"Rotated pseudonym to: {self.vehicle_pseudonym}")
    
    async def _check_pseudonym_rotation(self):
        """Check if pseudonym should be rotated"""
        if (datetime.now() - self.last_pseudonym_rotation) >= self.pseudonym_rotation_interval:
            await self._rotate_pseudonym()
    
    def register_message_callback(self, message_type: str, callback: Callable):
        """Register callback for specific message type"""
        if message_type not in self.message_callbacks:
            self.message_callbacks[message_type] = []
        self.message_callbacks[message_type].append(callback)
    
    def get_discovered_rsus(self) -> List[RSUInfo]:
        """Get list of discovered RSUs"""
        return list(self.rsu_discovery.discovered_rsus.values())
    
    def get_v2x_stats(self) -> Dict[str, Any]:
        """Get V2X communication statistics"""
        return {
            **self.protocol_stack.stats,
            'active_handshakes': len(self.active_handshakes),
            'discovered_rsus': len(self.rsu_discovery.discovered_rsus),
            'current_pseudonym': self.vehicle_pseudonym,
            'protocol_type': self.protocol_stack.protocol_type
        }
