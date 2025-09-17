"""
Vehicle OBU Main Application
Ephemeral Federated Multi-Modal Tolling System - Vehicle Side

This module orchestrates all vehicle OBU components including V2X communication,
token management, payment wallet, and optional federated learning participation.
"""

import asyncio
import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any

from v2x_client import V2XClient
from token_holder import TokenHolder
from fl_client import VehicleFLClient
from wallet import CryptoWallet
from privacy_manager import PrivacyManager
from config.obu_config import OBUConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VehicleOBUService:
    """
    Main Vehicle OBU Service
    
    Coordinates all OBU components and handles:
    - RSU discovery and handshake
    - Token acquisition and management
    - Payment proof generation
    - Privacy protection
    - Optional federated learning
    """
    
    def __init__(self, config: OBUConfig):
        self.config = config
        self.running = False
        
        # Initialize components
        self.v2x_client = None
        self.token_holder = None
        self.fl_client = None
        self.wallet = None
        self.privacy_manager = None
        
        # Communication queues
        self.rsu_events_queue = asyncio.Queue()
        self.toll_events_queue = asyncio.Queue()
        
        # State management
        self.current_tokens = {}
        self.active_toll_sessions = {}
        
    async def initialize(self):
        """Initialize all OBU components"""
        logger.info("Initializing Vehicle OBU Service...")
        
        try:
            # Initialize privacy manager first
            self.privacy_manager = PrivacyManager(self.config.privacy)
            await self.privacy_manager.initialize()
            
            # Initialize V2X client
            self.v2x_client = V2XClient(
                config=self.config.v2x,
                event_queue=self.rsu_events_queue
            )
            await self.v2x_client.initialize()
            
            # Initialize token holder
            self.token_holder = TokenHolder(
                config=self.config.tokens,
                privacy_manager=self.privacy_manager
            )
            await self.token_holder.initialize()
            
            # Initialize crypto wallet
            self.wallet = CryptoWallet(
                config=self.config.wallet,
                privacy_manager=self.privacy_manager
            )
            await self.wallet.initialize()
            
            # Initialize federated learning client (optional)
            if self.config.federated_learning.enabled:
                self.fl_client = VehicleFLClient(
                    config=self.config.federated_learning,
                    privacy_manager=self.privacy_manager
                )
                await self.fl_client.initialize()
            
            logger.info("Vehicle OBU Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vehicle OBU Service: {e}")
            raise
    
    async def start(self):
        """Start the Vehicle OBU Service"""
        if self.running:
            logger.warning("Vehicle OBU Service is already running")
            return
            
        logger.info("Starting Vehicle OBU Service...")
        self.running = True
        
        # Start all components
        tasks = [
            self.v2x_client.start(),
            self.token_holder.start(),
            self.wallet.start(),
            self._toll_coordinator()
        ]
        
        if self.fl_client:
            tasks.append(self.fl_client.start())
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in Vehicle OBU Service: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the Vehicle OBU Service"""
        if not self.running:
            return
            
        logger.info("Stopping Vehicle OBU Service...")
        self.running = False
        
        # Stop all components gracefully
        stop_tasks = []
        if self.v2x_client:
            stop_tasks.append(self.v2x_client.stop())
        if self.token_holder:
            stop_tasks.append(self.token_holder.stop())
        if self.wallet:
            stop_tasks.append(self.wallet.stop())
        if self.fl_client:
            stop_tasks.append(self.fl_client.stop())
        if self.privacy_manager:
            stop_tasks.append(self.privacy_manager.stop())
            
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        logger.info("Vehicle OBU Service stopped")
    
    async def _toll_coordinator(self):
        """
        Main toll coordination logic
        
        Handles the complete toll transaction flow:
        1. RSU discovery and handshake
        2. Token acquisition
        3. Payment proof generation
        4. Transaction completion
        """
        logger.info("Starting toll coordinator...")
        
        while self.running:
            try:
                # Handle RSU events
                await self._handle_rsu_events()
                
                # Handle toll transactions
                await self._handle_toll_transactions()
                
                # Update privacy parameters
                await self._update_privacy_parameters()
                
                await asyncio.sleep(0.1)  # Prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in toll coordinator: {e}")
                await asyncio.sleep(1)  # Back off on error
    
    async def _handle_rsu_events(self):
        """Handle events from RSU interactions"""
        try:
            while not self.rsu_events_queue.empty():
                event = await self.rsu_events_queue.get()
                
                if event['type'] == 'rsu_discovered':
                    await self._handle_rsu_discovery(event)
                elif event['type'] == 'handshake_initiated':
                    await self._handle_handshake_initiation(event)
                elif event['type'] == 'token_offered':
                    await self._handle_token_offer(event)
                elif event['type'] == 'payment_requested':
                    await self._handle_payment_request(event)
                
        except asyncio.QueueEmpty:
            pass
        except Exception as e:
            logger.error(f"Error handling RSU events: {e}")
    
    async def _handle_rsu_discovery(self, event: Dict[str, Any]):
        """Handle RSU discovery event"""
        rsu_id = event['rsu_id']
        logger.info(f"Discovered RSU: {rsu_id}")
        
        # TODO: Implement RSU discovery logic
        # - Evaluate RSU trustworthiness
        # - Check privacy policies
        # - Initiate handshake if appropriate
    
    async def _handle_handshake_initiation(self, event: Dict[str, Any]):
        """Handle handshake initiation from RSU"""
        rsu_id = event['rsu_id']
        handshake_data = event['handshake_data']
        
        logger.info(f"Handshake initiated by RSU: {rsu_id}")
        
        try:
            # Generate ephemeral pseudonym for this interaction
            pseudonym = await self.privacy_manager.generate_ephemeral_pseudonym()
            
            # Complete handshake through V2X client
            success = await self.v2x_client.complete_handshake(
                rsu_id, handshake_data, pseudonym
            )
            
            if success:
                logger.info(f"Handshake completed with RSU: {rsu_id}")
            else:
                logger.warning(f"Handshake failed with RSU: {rsu_id}")
                
        except Exception as e:
            logger.error(f"Error in handshake with RSU {rsu_id}: {e}")
    
    async def _handle_token_offer(self, event: Dict[str, Any]):
        """Handle ephemeral token offer from RSU"""
        rsu_id = event['rsu_id']
        token_data = event['token_data']
        
        logger.info(f"Token offered by RSU: {rsu_id}")
        
        try:
            # Validate token offer
            if await self.token_holder.validate_token_offer(token_data):
                # Accept and store token
                token = await self.token_holder.accept_token(token_data)
                self.current_tokens[rsu_id] = token
                
                logger.info(f"Token accepted from RSU: {rsu_id}")
            else:
                logger.warning(f"Invalid token offer from RSU: {rsu_id}")
                
        except Exception as e:
            logger.error(f"Error handling token offer from RSU {rsu_id}: {e}")
    
    async def _handle_payment_request(self, event: Dict[str, Any]):
        """Handle payment request for toll transaction"""
        rsu_id = event['rsu_id']
        payment_request = event['payment_request']
        
        logger.info(f"Payment requested by RSU: {rsu_id}, amount: ${payment_request['amount']}")
        
        try:
            # Get token for this RSU
            if rsu_id not in self.current_tokens:
                logger.error(f"No token available for RSU: {rsu_id}")
                return
            
            token = self.current_tokens[rsu_id]
            
            # Generate payment proof
            payment_proof = await self.wallet.generate_payment_proof(
                amount=payment_request['amount'],
                ephemeral_token_id=token.token_id,
                payment_method=payment_request.get('method', 'account_balance')
            )
            
            if payment_proof:
                # Send payment proof to RSU
                await self.v2x_client.send_payment_proof(rsu_id, payment_proof)
                
                # Create toll session record
                session_id = f"{rsu_id}_{token.token_id}"
                self.active_toll_sessions[session_id] = {
                    'rsu_id': rsu_id,
                    'token': token,
                    'amount': payment_request['amount'],
                    'payment_proof': payment_proof,
                    'timestamp': event['timestamp']
                }
                
                logger.info(f"Payment proof sent to RSU: {rsu_id}")
            else:
                logger.error(f"Failed to generate payment proof for RSU: {rsu_id}")
                
        except Exception as e:
            logger.error(f"Error handling payment request from RSU {rsu_id}: {e}")
    
    async def _handle_toll_transactions(self):
        """Handle ongoing toll transactions"""
        # TODO: Implement toll transaction monitoring
        # - Check transaction status
        # - Handle confirmations/rejections
        # - Clean up completed transactions
        pass
    
    async def _update_privacy_parameters(self):
        """Update privacy parameters based on context"""
        # TODO: Implement adaptive privacy management
        # - Adjust pseudonym rotation frequency
        # - Update differential privacy parameters
        # - Monitor privacy budget usage
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current OBU status"""
        status = {
            'running': self.running,
            'vehicle_id': self.config.vehicle_id,
            'components': {
                'v2x_client': self.v2x_client.is_active if self.v2x_client else False,
                'token_holder': self.token_holder.is_running if self.token_holder else False,
                'wallet': self.wallet.is_active if self.wallet else False,
                'fl_client': self.fl_client.is_running if self.fl_client else False
            },
            'current_tokens': len(self.current_tokens),
            'active_sessions': len(self.active_toll_sessions)
        }
        
        return status


def load_config(config_path: str) -> OBUConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return OBUConfig.from_dict(config_dict)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Vehicle OBU Application')
    parser.add_argument('--config', 
                       default='config/obu_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--mode',
                       choices=['embedded', 'mobile', 'simulation'],
                       default='embedded',
                       help='Operating mode')
    parser.add_argument('--vehicle-id',
                       help='Override vehicle ID from config')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    config = load_config(str(config_path))
    config.mode = args.mode
    
    if args.vehicle_id:
        config.vehicle_id = args.vehicle_id
    
    # Create and run OBU service
    obu_service = VehicleOBUService(config)
    
    try:
        await obu_service.initialize()
        await obu_service.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        return 1
    finally:
        await obu_service.stop()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
