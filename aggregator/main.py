"""
Federated Learning Aggregator Service
Coordinates distributed training across RSUs and vehicles

Main entry point for the aggregation service that handles:
- Participant registration and management
- Secure model update aggregation  
- Global model distribution
- Privacy-preserving protocols
- Training round coordination
"""

import asyncio
import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any

from aggregation_server import AggregationServer
from model_repository import ModelRepository
from privacy_engine import PrivacyEngine
from participant_registry import ParticipantRegistry
from config.aggregator_config import AggregatorConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FederatedAggregatorService:
    """
    Main Federated Learning Aggregator Service
    
    Orchestrates all aggregation components and provides:
    - Centralized coordination of federated training
    - Privacy-preserving aggregation protocols
    - Model versioning and distribution
    - Participant lifecycle management
    """
    
    def __init__(self, config: AggregatorConfig):
        self.config = config
        self.running = False
        
        # Core components
        self.aggregation_server = None
        self.model_repository = None
        self.privacy_engine = None
        self.participant_registry = None
        
        # Service state
        self.current_round = 0
        self.training_active = False
    
    async def initialize(self):
        """Initialize all aggregator components"""
        logger.info("Initializing Federated Aggregator Service...")
        
        try:
            # Initialize model repository first
            self.model_repository = ModelRepository(self.config.model_repository)
            await self.model_repository.initialize()
            
            # Initialize privacy engine
            self.privacy_engine = PrivacyEngine(self.config.privacy)
            await self.privacy_engine.initialize()
            
            # Initialize participant registry
            self.participant_registry = ParticipantRegistry(self.config.participants)
            await self.participant_registry.initialize()
            
            # Initialize aggregation server
            self.aggregation_server = AggregationServer(
                config=self.config.aggregation,
                model_repository=self.model_repository,
                privacy_engine=self.privacy_engine,
                participant_registry=self.participant_registry
            )
            await self.aggregation_server.initialize()
            
            logger.info("Federated Aggregator Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Federated Aggregator Service: {e}")
            raise
    
    async def start(self):
        """Start the aggregator service"""
        if self.running:
            logger.warning("Federated Aggregator Service is already running")
            return
            
        logger.info("Starting Federated Aggregator Service...")
        self.running = True
        
        # Start all components
        tasks = [
            self.aggregation_server.start(),
            self.participant_registry.start(),
            self._training_coordinator()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in Federated Aggregator Service: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the aggregator service"""
        if not self.running:
            return
            
        logger.info("Stopping Federated Aggregator Service...")
        self.running = False
        
        # Stop all components gracefully
        stop_tasks = []
        if self.aggregation_server:
            stop_tasks.append(self.aggregation_server.stop())
        if self.participant_registry:
            stop_tasks.append(self.participant_registry.stop())
        if self.model_repository:
            stop_tasks.append(self.model_repository.stop())
        if self.privacy_engine:
            stop_tasks.append(self.privacy_engine.stop())
            
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        logger.info("Federated Aggregator Service stopped")
    
    async def _training_coordinator(self):
        """
        Main training coordination loop
        
        Manages the federated learning rounds:
        1. Wait for sufficient participants
        2. Initiate training round
        3. Collect model updates
        4. Perform secure aggregation
        5. Distribute updated global model
        """
        logger.info("Starting training coordinator...")
        
        while self.running:
            try:
                # Check if training should start
                if await self._should_start_training_round():
                    await self._execute_training_round()
                
                # Wait between round checks
                await asyncio.sleep(self.config.aggregation.round_check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in training coordinator: {e}")
                await asyncio.sleep(60)  # Back off on error
    
    async def _should_start_training_round(self) -> bool:
        """Determine if a new training round should start"""
        if self.training_active:
            return False
        
        # Check minimum participants
        active_participants = await self.participant_registry.get_active_participants()
        min_participants = self.config.aggregation.min_participants_per_round
        
        if len(active_participants) < min_participants:
            logger.debug(f"Insufficient participants: {len(active_participants)}/{min_participants}")
            return False
        
        # Check time since last round
        if hasattr(self, 'last_round_time'):
            min_interval = self.config.aggregation.min_round_interval_seconds
            time_since_last = (asyncio.get_event_loop().time() - self.last_round_time)
            if time_since_last < min_interval:
                return False
        
        return True
    
    async def _execute_training_round(self):
        """Execute a complete federated training round"""
        self.current_round += 1
        self.training_active = True
        
        logger.info(f"Starting federated training round {self.current_round}")
        
        try:
            # Start the training round on aggregation server
            round_result = await self.aggregation_server.execute_training_round(
                round_number=self.current_round
            )
            
            if round_result['success']:
                logger.info(f"Training round {self.current_round} completed successfully")
                logger.info(f"Participants: {round_result['num_participants']}")
                logger.info(f"Convergence: {round_result.get('convergence_metric', 'N/A')}")
            else:
                logger.warning(f"Training round {self.current_round} failed: {round_result.get('error')}")
            
            self.last_round_time = asyncio.get_event_loop().time()
            
        except Exception as e:
            logger.error(f"Error executing training round {self.current_round}: {e}")
        
        finally:
            self.training_active = False
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status"""
        return {
            'running': self.running,
            'current_round': self.current_round,
            'training_active': self.training_active,
            'components': {
                'aggregation_server': self.aggregation_server.is_running if self.aggregation_server else False,
                'model_repository': self.model_repository.is_active if self.model_repository else False,
                'participant_registry': self.participant_registry.is_running if self.participant_registry else False,
                'privacy_engine': self.privacy_engine.is_active if self.privacy_engine else False
            }
        }


def load_config(config_path: str) -> AggregatorConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return AggregatorConfig.from_dict(config_dict)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Federated Learning Aggregator Service')
    parser.add_argument('--config', 
                       default='config/aggregator_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--enable-dashboard',
                       action='store_true',
                       help='Enable web dashboard')
    parser.add_argument('--port',
                       type=int,
                       default=8080,
                       help='Dashboard port')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    config = load_config(str(config_path))
    
    # Enable dashboard if requested
    if args.enable_dashboard:
        config.dashboard.enabled = True
        config.dashboard.port = args.port
    
    # Create and run aggregator service
    aggregator_service = FederatedAggregatorService(config)
    
    try:
        await aggregator_service.initialize()
        await aggregator_service.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        return 1
    finally:
        await aggregator_service.stop()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
