"""
Federated Learning Client for RSU Edge Module
Participates in distributed model training while preserving privacy

Key features:
- Secure aggregation for privacy preservation
- Local model updates from fusion results
- Differential privacy integration
- Model compression and quantization
- Communication with aggregation server
"""

import asyncio
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pickle
import hashlib

from fusion_engine import MultiModalFusionEngine, FusionResult

logger = logging.getLogger(__name__)


@dataclass
class TrainingBatch:
    """Batch of training data for federated learning"""
    inputs: Dict[str, torch.Tensor]
    targets: Dict[str, torch.Tensor]
    weights: torch.Tensor
    metadata: Dict[str, Any]


@dataclass
class ModelUpdate:
    """Model update for federated aggregation"""
    update_id: str
    round_number: int
    participant_id: str
    timestamp: datetime
    model_deltas: Dict[str, torch.Tensor]
    num_samples: int
    loss_value: float
    privacy_budget_used: float


class PrivacyMechanism:
    """Differential privacy mechanism for federated learning"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, 
                 sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.privacy_budget_used = 0.0
    
    def add_noise_to_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to gradients"""
        if self.privacy_budget_used >= self.epsilon:
            logger.warning("Privacy budget exhausted, skipping noise addition")
            return gradients
        
        noisy_gradients = {}
        noise_scale = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
        for name, grad in gradients.items():
            if grad is not None:
                noise = torch.normal(0, noise_scale, size=grad.shape)
                noisy_gradients[name] = grad + noise
            else:
                noisy_gradients[name] = grad
        
        # Update privacy budget
        self.privacy_budget_used += self.epsilon / 10  # Conservative budget usage
        
        return noisy_gradients
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget"""
        return max(0.0, self.epsilon - self.privacy_budget_used)


class SecureAggregation:
    """Secure aggregation protocol for federated learning"""
    
    def __init__(self, participant_id: str, aggregation_config: Dict[str, Any]):
        self.participant_id = participant_id
        self.config = aggregation_config
        self.shared_keys: Dict[str, bytes] = {}
        self.masked_updates: Dict[str, torch.Tensor] = {}
    
    def generate_shared_key(self, other_participant_id: str) -> bytes:
        """Generate shared key with another participant for secure aggregation"""
        # TODO: Implement proper key exchange (Diffie-Hellman, etc.)
        # For now, use a deterministic key based on participant IDs
        key_material = f"{self.participant_id}:{other_participant_id}"
        return hashlib.sha256(key_material.encode()).digest()
    
    def mask_model_update(self, model_update: Dict[str, torch.Tensor],
                         participant_ids: List[str]) -> Dict[str, torch.Tensor]:
        """
        Apply secure aggregation masks to model update
        
        Each participant adds/subtracts random masks that cancel out
        during aggregation, hiding individual contributions.
        """
        masked_update = {}
        
        for param_name, param_tensor in model_update.items():
            mask = torch.zeros_like(param_tensor)
            
            # Generate pairwise masks with other participants
            for other_id in participant_ids:
                if other_id != self.participant_id:
                    # Generate shared key if not exists
                    if other_id not in self.shared_keys:
                        self.shared_keys[other_id] = self.generate_shared_key(other_id)
                    
                    # Generate mask from shared key
                    key = self.shared_keys[other_id]
                    np.random.seed(int.from_bytes(key[:4], 'big'))
                    
                    mask_values = torch.tensor(
                        np.random.normal(0, 0.1, param_tensor.shape),
                        dtype=param_tensor.dtype
                    )
                    
                    # Add or subtract based on participant ID ordering
                    if self.participant_id < other_id:
                        mask += mask_values
                    else:
                        mask -= mask_values
            
            masked_update[param_name] = param_tensor + mask
        
        return masked_update
    
    def unmask_aggregated_update(self, aggregated_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remove masks from aggregated update (done by aggregation server)"""
        # Masks should cancel out during aggregation
        return aggregated_update


class FederatedLearningClient:
    """
    Federated Learning Client for RSU
    
    Participates in distributed training of the multi-modal fusion model
    while preserving privacy through differential privacy and secure aggregation.
    """
    
    def __init__(self, config: Dict[str, Any], fusion_engine: MultiModalFusionEngine):
        self.config = config
        self.fusion_engine = fusion_engine
        self.participant_id = config['participant_id']
        self.is_running = False
        
        # Privacy and security
        self.privacy_mechanism = PrivacyMechanism(
            epsilon=config.get('dp_epsilon', 1.0),
            delta=config.get('dp_delta', 1e-5)
        )
        self.secure_aggregation = SecureAggregation(
            self.participant_id, 
            config.get('secure_aggregation', {})
        )
        
        # Training state
        self.local_model = None
        self.optimizer = None
        self.training_data_buffer: List[TrainingBatch] = []
        self.current_round = 0
        self.last_update_time = None
        
        # Communication
        self.aggregation_server_url = config['aggregation_server_url']
        self.communication_task = None
        self.training_task = None
        
        # Metrics
        self.training_stats = {
            'rounds_participated': 0,
            'updates_sent': 0,
            'updates_received': 0,
            'avg_local_loss': 0.0,
            'privacy_budget_remaining': self.privacy_mechanism.epsilon
        }
    
    async def initialize(self):
        """Initialize federated learning client"""
        logger.info(f"Initializing FL client {self.participant_id}")
        
        try:
            # Get model from fusion engine
            self.local_model = self.fusion_engine.fusion_model
            
            # Initialize optimizer
            self.optimizer = torch.optim.Adam(
                self.local_model.parameters(),
                lr=self.config.get('learning_rate', 0.001)
            )
            
            # Initialize communication with aggregation server
            await self._register_with_server()
            
            logger.info(f"FL client {self.participant_id} initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize FL client: {e}")
            raise
    
    async def start(self):
        """Start federated learning client"""
        if self.is_running:
            return
            
        logger.info(f"Starting FL client {self.participant_id}")
        self.is_running = True
        
        # Start background tasks
        self.communication_task = asyncio.create_task(self._communication_loop())
        self.training_task = asyncio.create_task(self._training_loop())
        
        logger.info(f"FL client {self.participant_id} started")
    
    async def stop(self):
        """Stop federated learning client"""
        if not self.is_running:
            return
            
        logger.info(f"Stopping FL client {self.participant_id}")
        self.is_running = False
        
        # Cancel tasks
        if self.communication_task:
            self.communication_task.cancel()
        if self.training_task:
            self.training_task.cancel()
        
        # Wait for tasks to complete
        tasks = [self.communication_task, self.training_task]
        tasks = [task for task in tasks if task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"FL client {self.participant_id} stopped")
    
    async def _register_with_server(self):
        """Register with aggregation server"""
        # TODO: Implement actual HTTP/gRPC registration
        logger.info(f"Registering {self.participant_id} with aggregation server")
        
        registration_data = {
            'participant_id': self.participant_id,
            'capabilities': {
                'modalities': list(self.config.get('supported_modalities', [])),
                'privacy_level': self.privacy_mechanism.epsilon,
                'model_version': self.config.get('model_version', '1.0')
            }
        }
        
        # Simulate registration success
        logger.info(f"Registration successful for {self.participant_id}")
    
    async def _communication_loop(self):
        """Main communication loop with aggregation server"""
        while self.is_running:
            try:
                # Check for new global model updates
                await self._check_for_global_updates()
                
                # Send local updates if ready
                if self._should_send_local_update():
                    await self._send_local_update()
                
                await asyncio.sleep(self.config.get('communication_interval_seconds', 30))
                
            except Exception as e:
                logger.error(f"Error in communication loop: {e}")
                await asyncio.sleep(10)  # Back off on error
    
    async def _training_loop(self):
        """Main local training loop"""
        while self.is_running:
            try:
                # Collect training data from fusion results
                await self._collect_training_data()
                
                # Perform local training if enough data available
                if len(self.training_data_buffer) >= self.config.get('min_training_samples', 10):
                    await self._perform_local_training()
                
                await asyncio.sleep(self.config.get('training_interval_seconds', 60))
                
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                await asyncio.sleep(10)  # Back off on error
    
    async def _collect_training_data(self):
        """Collect training data from recent fusion results"""
        try:
            # Get recent completed events from fusion engine
            # TODO: Implement method to get training data from fusion engine
            # For now, simulate data collection
            
            if len(self.training_data_buffer) < self.config.get('max_buffer_size', 1000):
                # Simulate adding training sample
                sample_data = self._create_synthetic_training_sample()
                if sample_data:
                    self.training_data_buffer.append(sample_data)
            
        except Exception as e:
            logger.error(f"Error collecting training data: {e}")
    
    def _create_synthetic_training_sample(self) -> Optional[TrainingBatch]:
        """Create synthetic training sample for testing"""
        # TODO: Replace with actual data from fusion results
        try:
            # Create mock input data
            inputs = {
                'anpr': torch.randn(1, 512),
                'rfid': torch.randn(1, 3),
                'v2x': torch.randn(1, 5)
            }
            
            # Create mock targets
            targets = {
                'match_confidence': torch.tensor([0.9]),
                'recommended_charge': torch.tensor([5.50])
            }
            
            weights = torch.tensor([1.0])
            
            metadata = {
                'timestamp': datetime.now(),
                'lane_id': 'lane_1',
                'event_id': f"synthetic_{np.random.randint(1000)}"
            }
            
            return TrainingBatch(inputs, targets, weights, metadata)
            
        except Exception as e:
            logger.error(f"Error creating synthetic training sample: {e}")
            return None
    
    async def _perform_local_training(self):
        """Perform local training on collected data"""
        if not self.training_data_buffer:
            return
        
        logger.info(f"Performing local training with {len(self.training_data_buffer)} samples")
        
        try:
            self.local_model.train()
            total_loss = 0.0
            num_batches = 0
            
            # Create batches from buffer
            batch_size = self.config.get('batch_size', 32)
            
            for i in range(0, len(self.training_data_buffer), batch_size):
                batch_samples = self.training_data_buffer[i:i + batch_size]
                
                # Combine samples into batch
                batch_inputs = self._combine_batch_inputs(batch_samples)
                batch_targets = self._combine_batch_targets(batch_samples)
                batch_weights = torch.cat([sample.weights for sample in batch_samples])
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.local_model(
                    batch_inputs, 
                    {mod: 0.8 for mod in batch_inputs.keys()}  # Mock confidences
                )
                
                # Compute loss
                loss = self._compute_loss(outputs, batch_targets, batch_weights)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    self.local_model.parameters(), 
                    max_norm=self.config.get('grad_clip_norm', 1.0)
                )
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            self.training_stats['avg_local_loss'] = avg_loss
            
            # Clear training buffer
            self.training_data_buffer.clear()
            
            self.local_model.eval()
            
            logger.info(f"Local training completed, avg loss: {avg_loss:.4f}")
            
        except Exception as e:
            logger.error(f"Error in local training: {e}")
    
    def _combine_batch_inputs(self, batch_samples: List[TrainingBatch]) -> Dict[str, torch.Tensor]:
        """Combine input data from multiple samples into batch"""
        combined_inputs = {}
        
        # Get all modalities present in samples
        all_modalities = set()
        for sample in batch_samples:
            all_modalities.update(sample.inputs.keys())
        
        for modality in all_modalities:
            modality_tensors = []
            for sample in batch_samples:
                if modality in sample.inputs:
                    modality_tensors.append(sample.inputs[modality])
                else:
                    # Handle missing modality with zeros
                    # TODO: Use proper missing data handling
                    if modality_tensors:
                        shape = modality_tensors[0].shape
                        modality_tensors.append(torch.zeros(shape))
            
            if modality_tensors:
                combined_inputs[modality] = torch.cat(modality_tensors, dim=0)
        
        return combined_inputs
    
    def _combine_batch_targets(self, batch_samples: List[TrainingBatch]) -> Dict[str, torch.Tensor]:
        """Combine target data from multiple samples into batch"""
        combined_targets = {}
        
        # Get all target types
        all_targets = set()
        for sample in batch_samples:
            all_targets.update(sample.targets.keys())
        
        for target in all_targets:
            target_tensors = [
                sample.targets[target] for sample in batch_samples 
                if target in sample.targets
            ]
            if target_tensors:
                combined_targets[target] = torch.cat(target_tensors, dim=0)
        
        return combined_targets
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor],
                     weights: torch.Tensor) -> torch.Tensor:
        """Compute training loss"""
        total_loss = 0.0
        
        # Match confidence loss
        if 'match_confidence' in outputs and 'match_confidence' in targets:
            match_loss = nn.functional.mse_loss(
                outputs['match_confidence'].squeeze(),
                targets['match_confidence'].squeeze(),
                reduction='none'
            )
            total_loss += torch.mean(match_loss * weights)
        
        # Charge prediction loss
        if 'recommended_charge' in outputs and 'recommended_charge' in targets:
            charge_loss = nn.functional.mse_loss(
                outputs['recommended_charge'].squeeze(),
                targets['recommended_charge'].squeeze(),
                reduction='none'
            )
            total_loss += torch.mean(charge_loss * weights)
        
        return total_loss
    
    def _should_send_local_update(self) -> bool:
        """Determine if local update should be sent to server"""
        if self.last_update_time is None:
            return True
        
        time_since_update = datetime.now() - self.last_update_time
        min_update_interval = timedelta(
            seconds=self.config.get('min_update_interval_seconds', 300)
        )
        
        return time_since_update >= min_update_interval
    
    async def _send_local_update(self):
        """Send local model update to aggregation server"""
        try:
            logger.info(f"Sending local update from {self.participant_id}")
            
            # Compute model deltas
            model_deltas = self._compute_model_deltas()
            
            # Apply differential privacy
            if self.config.get('use_differential_privacy', True):
                model_deltas = self.privacy_mechanism.add_noise_to_gradients(model_deltas)
                self.training_stats['privacy_budget_remaining'] = (
                    self.privacy_mechanism.get_remaining_budget()
                )
            
            # Apply secure aggregation masking
            if self.config.get('use_secure_aggregation', True):
                # TODO: Get list of other participants from server
                other_participants = ['rsu_001', 'rsu_002', 'rsu_003']  # Mock
                model_deltas = self.secure_aggregation.mask_model_update(
                    model_deltas, other_participants
                )
            
            # Create update message
            update = ModelUpdate(
                update_id=f"{self.participant_id}_{self.current_round}_{datetime.now().timestamp()}",
                round_number=self.current_round,
                participant_id=self.participant_id,
                timestamp=datetime.now(),
                model_deltas=model_deltas,
                num_samples=len(self.training_data_buffer),
                loss_value=self.training_stats['avg_local_loss'],
                privacy_budget_used=self.privacy_mechanism.privacy_budget_used
            )
            
            # Send update to server
            await self._transmit_update_to_server(update)
            
            self.last_update_time = datetime.now()
            self.training_stats['updates_sent'] += 1
            
            logger.info(f"Local update sent successfully from {self.participant_id}")
            
        except Exception as e:
            logger.error(f"Error sending local update: {e}")
    
    def _compute_model_deltas(self) -> Dict[str, torch.Tensor]:
        """Compute model parameter deltas for federated update"""
        # TODO: Implement proper delta computation
        # For now, return current gradients
        model_deltas = {}
        
        for name, param in self.local_model.named_parameters():
            if param.grad is not None:
                model_deltas[name] = param.grad.clone()
            else:
                model_deltas[name] = torch.zeros_like(param)
        
        return model_deltas
    
    async def _transmit_update_to_server(self, update: ModelUpdate):
        """Transmit model update to aggregation server"""
        # TODO: Implement actual HTTP/gRPC transmission
        logger.debug(f"Transmitting update {update.update_id} to server")
        
        # Simulate network transmission
        await asyncio.sleep(0.1)
        
        # Serialize update for transmission
        # serialized_update = pickle.dumps(update)
        logger.debug(f"Update {update.update_id} transmitted successfully")
    
    async def _check_for_global_updates(self):
        """Check for and download global model updates from server"""
        try:
            # TODO: Implement actual server communication
            logger.debug(f"Checking for global updates for {self.participant_id}")
            
            # Simulate checking for updates
            await asyncio.sleep(0.1)
            
            # Simulate occasional global update
            if np.random.random() < 0.1:  # 10% chance
                await self._apply_global_update()
            
        except Exception as e:
            logger.error(f"Error checking for global updates: {e}")
    
    async def _apply_global_update(self):
        """Apply global model update to local model"""
        try:
            logger.info(f"Applying global update to {self.participant_id}")
            
            # TODO: Download and apply actual global model update
            # For now, simulate update application
            
            self.current_round += 1
            self.training_stats['updates_received'] += 1
            self.training_stats['rounds_participated'] += 1
            
            logger.info(f"Global update applied, now at round {self.current_round}")
            
        except Exception as e:
            logger.error(f"Error applying global update: {e}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get federated learning statistics"""
        return {
            **self.training_stats,
            'current_round': self.current_round,
            'participant_id': self.participant_id,
            'buffer_size': len(self.training_data_buffer),
            'model_parameters': sum(p.numel() for p in self.local_model.parameters())
        }
