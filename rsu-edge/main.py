"""
RSU Edge Module Main Entry Point
Ephemeral Federated Multi-Modal Tolling System

This module orchestrates all RSU components including sensor adapters,
fusion engine, federated learning client, token management, and payment verification.
"""

import asyncio
import logging
import argparse
import yaml
import sys
from pathlib import Path
from typing import Dict, Any
from dataclasses import asdict

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sensor_adapters import SensorManager
from fusion_engine import MultiModalFusionEngine
from fl_client import FederatedLearningClient
from token_orchestrator import TokenOrchestrator
from payment_verifier import PaymentVerifier
from audit_buffer import AuditBuffer
from config.rsu_config import RSUConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


class RSUEdgeService:
    """
    Main RSU Edge Service that coordinates all components
    
    Handles:
    - Component initialization and lifecycle management
    - Event coordination between components
    - Error handling and recovery
    - Performance monitoring and metrics
    """
    
    def __init__(self, config: RSUConfig):
        self.config = config
        self.running = False
        
        # Initialize components
        self.sensor_manager = None
        self.fusion_engine = None
        self.fl_client = None
        self.token_orchestrator = None
        self.payment_verifier = None
        self.audit_buffer = None
        
        # Event queues for inter-component communication
        self.sensor_data_queue = asyncio.Queue()
        self.token_events_queue = asyncio.Queue()
        self.payment_events_queue = asyncio.Queue()
        
    async def initialize(self):
        """Initialize all RSU components"""
        logger.info("Initializing RSU Edge Service...")
        
        try:
            # Initialize audit buffer first (required by other components)
            # Convert dataclass configurations to plain dictionaries for components
            self.audit_buffer = AuditBuffer(asdict(self.config.audit))
            await self.audit_buffer.initialize()
            
            # Initialize sensor manager
            self.sensor_manager = SensorManager(
                config=asdict(self.config.sensors),
                data_queue=self.sensor_data_queue
            )
            await self.sensor_manager.initialize()
            
            # Initialize fusion engine
            self.fusion_engine = MultiModalFusionEngine(
                config=asdict(self.config.fusion),
                sensor_queue=self.sensor_data_queue
            )
            await self.fusion_engine.initialize()
            
            # Initialize federated learning client
            logger.debug(f"FLClient Config: {asdict(self.config.federated_learning)}")
            self.fl_client = FederatedLearningClient(
                config=asdict(self.config.federated_learning),
                fusion_engine=self.fusion_engine
            )
            await self.fl_client.initialize()
            
            # Initialize token orchestrator
            self.token_orchestrator = TokenOrchestrator(
                config=asdict(self.config.tokens),
                event_queue=self.token_events_queue
            )
            await self.token_orchestrator.initialize()
            
            # Initialize payment verifier with TEE support
            self.payment_verifier = PaymentVerifier(
                config=asdict(self.config.payments),
                tee_config=asdict(self.config.tee),
                event_queue=self.payment_events_queue,
                audit_buffer=self.audit_buffer
            )
            await self.payment_verifier.initialize()
            
            logger.info("RSU Edge Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RSU Edge Service: {e}")
            raise
    
    async def start(self):
        """Start the RSU Edge Service"""
        if self.running:
            logger.warning("RSU Edge Service is already running")
            return
            
        logger.info("Starting RSU Edge Service...")
        self.running = True
        
        # Start all components
        tasks = [
            self.sensor_manager.start(),
            self.fusion_engine.start(),
            self.fl_client.start(),
            self.token_orchestrator.start(),
            self.payment_verifier.start(),
            self._event_coordinator()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in RSU Edge Service: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the RSU Edge Service"""
        if not self.running:
            return
            
        logger.info("Stopping RSU Edge Service...")
        self.running = False
        
        # Stop all components gracefully
        stop_tasks = []
        if self.sensor_manager:
            stop_tasks.append(self.sensor_manager.stop())
        if self.fusion_engine:
            stop_tasks.append(self.fusion_engine.stop())
        if self.fl_client:
            stop_tasks.append(self.fl_client.stop())
        if self.token_orchestrator:
            stop_tasks.append(self.token_orchestrator.stop())
        if self.payment_verifier:
            stop_tasks.append(self.payment_verifier.stop())
        if self.audit_buffer:
            stop_tasks.append(self.audit_buffer.stop())
            
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        logger.info("RSU Edge Service stopped")
    
    async def _event_coordinator(self):
        """
        Coordinate events between components
        
        This is the main event loop that handles:
        - Vehicle detection and token issuance
        - Multi-modal fusion and matching
        - Payment verification and attestation
        - Audit record creation
        """
        logger.info("Starting event coordinator...")
        
        while self.running:
            try:
                # TODO: Implement event coordination logic
                # - Handle sensor fusion results
                # - Coordinate token validation with payment verification
                # - Generate audit attestations
                # - Handle federated learning triggers
                
                await asyncio.sleep(0.1)  # Prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in event coordinator: {e}")
                await asyncio.sleep(1)  # Back off on error


def load_config(config_path: str) -> RSUConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return RSUConfig.from_dict(config_dict)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='RSU Edge Module')
    parser.add_argument('--config', 
                       default='config/rsu_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--mode',
                       choices=['production', 'simulation', 'testing'],
                       default='production',
                       help='Operating mode')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    config = load_config(str(config_path))
    config.mode = args.mode
    
    # Create and run RSU service
    rsu_service = RSUEdgeService(config)
    
    try:
        await rsu_service.initialize()
        await rsu_service.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        return 1
    finally:
        await rsu_service.stop()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
