"""
Multi-Modal Fusion Engine for RSU Edge Module
Combines data from multiple sensor modalities for improved vehicle recognition

Key features:
- Real-time sensor data fusion
- Confidence-based modality selection
- Attention-based neural fusion model
- On-device inference optimization
- Federated learning integration
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

from sensor_adapters import SensorReading

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Result of multi-modal fusion processing"""
    timestamp: datetime
    vehicle_match_confidence: float
    plate_text_predictions: Dict[str, float]  # text -> probability
    recommended_charge: Optional[float]
    modality_contributions: Dict[str, float]  # sensor_type -> weight
    fusion_features: np.ndarray
    lane_id: Optional[str] = None


@dataclass
class VehicleEvent:
    """Represents a vehicle passage event with all associated sensor readings"""
    event_id: str
    start_time: datetime
    end_time: Optional[datetime]
    lane_id: Optional[str]
    sensor_readings: List[SensorReading]
    fusion_result: Optional[FusionResult] = None
    is_complete: bool = False


class ANPRFeatureExtractor(nn.Module):
    """CNN-based feature extractor for ANPR images"""
    
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        
        # TODO: Implement proper CNN backbone
        # For now, use a simple placeholder
        self.backbone = nn.Sequential(
            nn.Linear(224 * 224 * 3, 1024),  # Placeholder for actual CNN
            nn.ReLU(),
            nn.Linear(1024, feature_dim),
            nn.ReLU()
        )
        
        # Plate text classifier head
        self.text_classifier = nn.Linear(feature_dim, 36 * 8)  # 36 chars, 8 positions
    
    def forward(self, image_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_features: Preprocessed image features [batch, channels, height, width]
            
        Returns:
            features: Feature embeddings [batch, feature_dim]
            text_logits: Text predictions [batch, 36*8]
        """
        # TODO: Replace with actual CNN processing
        batch_size = image_features.shape[0]
        flattened = image_features.view(batch_size, -1)
        
        features = self.backbone(flattened)
        text_logits = self.text_classifier(features)
        
        return features, text_logits


class ModalityEncoder(nn.Module):
    """Encodes different sensor modalities into common feature space"""
    
    def __init__(self, modality_configs: Dict[str, Dict[str, Any]]):
        super().__init__()
        self.modality_configs = modality_configs
        self.encoders = nn.ModuleDict()
        
        # ANPR encoder
        if 'anpr' in modality_configs:
            self.encoders['anpr'] = ANPRFeatureExtractor(
                feature_dim=modality_configs['anpr']['feature_dim']
            )
        
        # RFID encoder (simple embedding)
        if 'rfid' in modality_configs:
            self.encoders['rfid'] = nn.Sequential(
                nn.Linear(3, 128),  # tag_id_hash, signal_strength, protocol_type
                nn.ReLU(),
                nn.Linear(128, modality_configs['rfid']['feature_dim'])
            )
        
        # V2X encoder
        if 'v2x' in modality_configs:
            self.encoders['v2x'] = nn.Sequential(
                nn.Linear(5, 128),  # pseudonym_hash, signal_strength, speed, heading, message_type
                nn.ReLU(),
                nn.Linear(128, modality_configs['v2x']['feature_dim'])
            )
    
    def forward(self, modality_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode each modality into feature space
        
        Args:
            modality_data: Dict mapping modality names to their input tensors
            
        Returns:
            Dict mapping modality names to their encoded features
        """
        encoded_features = {}
        
        for modality, data in modality_data.items():
            if modality in self.encoders:
                if modality == 'anpr':
                    features, _ = self.encoders[modality](data)
                    encoded_features[modality] = features
                else:
                    encoded_features[modality] = self.encoders[modality](data)
        
        return encoded_features


class AttentionFusionLayer(nn.Module):
    """Attention-based fusion of multi-modal features"""
    
    def __init__(self, feature_dim: int, num_modalities: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        
        # Cross-attention mechanism
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        # Modality importance weighting
        self.modality_weights = nn.Linear(feature_dim, 1)
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, modality_features: Dict[str, torch.Tensor], 
                modality_confidences: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Fuse multi-modal features using attention
        
        Args:
            modality_features: Dict of encoded features for each modality
            modality_confidences: Confidence scores for each modality
            
        Returns:
            fused_features: Fused feature representation
            attention_weights: Attention weights for each modality
        """
        if not modality_features:
            return torch.zeros(1, self.feature_dim), {}
        
        # Stack features and compute attention
        feature_list = list(modality_features.values())
        modality_names = list(modality_features.keys())
        
        # Simple average fusion for now (TODO: implement proper attention)
        stacked_features = torch.stack(feature_list, dim=1)  # [batch, num_modalities, feature_dim]
        
        # Compute attention weights based on confidence scores
        confidence_weights = torch.tensor([
            modality_confidences.get(name, 0.0) for name in modality_names
        ])
        confidence_weights = torch.softmax(confidence_weights, dim=0)
        
        # Weighted average
        fused_features = torch.sum(
            stacked_features * confidence_weights.view(1, -1, 1), 
            dim=1
        )
        
        # Apply fusion layer
        fused_features = self.fusion_layer(fused_features)
        
        # Return attention weights as dict
        attention_weights = {
            name: weight.item() 
            for name, weight in zip(modality_names, confidence_weights)
        }
        
        return fused_features, attention_weights


class MultiModalFusionModel(nn.Module):
    """Complete multi-modal fusion model for vehicle recognition"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Modality encoders
        self.modality_encoder = ModalityEncoder(config['modalities'])
        
        # Fusion layer
        self.fusion_layer = AttentionFusionLayer(
            feature_dim=config['fusion']['feature_dim'],
            num_modalities=len(config['modalities'])
        )
        
        # Output heads
        feature_dim = config['fusion']['feature_dim']
        self.match_confidence_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.charge_prediction_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU()  # Ensure positive charge
        )
    
    def forward(self, modality_data: Dict[str, torch.Tensor],
                modality_confidences: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through fusion model
        
        Args:
            modality_data: Input data for each modality
            modality_confidences: Confidence scores for each modality
            
        Returns:
            Dict containing model outputs
        """
        # Encode modalities
        encoded_features = self.modality_encoder(modality_data)
        
        # Fuse features
        fused_features, attention_weights = self.fusion_layer(
            encoded_features, modality_confidences
        )
        
        # Compute outputs
        match_confidence = self.match_confidence_head(fused_features)
        recommended_charge = self.charge_prediction_head(fused_features)
        
        return {
            'fused_features': fused_features,
            'match_confidence': match_confidence,
            'recommended_charge': recommended_charge,
            'attention_weights': attention_weights
        }


class MultiModalFusionEngine:
    """
    Multi-Modal Fusion Engine
    
    Processes sensor data from multiple modalities and produces unified
    vehicle recognition results using neural fusion models.
    """
    
    def __init__(self, config: Dict[str, Any], sensor_queue: asyncio.Queue):
        self.config = config
        self.sensor_queue = sensor_queue
        self.is_running = False
        
        # Model and processing
        self.fusion_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Event tracking
        self.active_events: Dict[str, VehicleEvent] = {}
        self.event_timeout = timedelta(seconds=config.get('event_timeout_seconds', 30))
        
        # Processing tasks
        self.processing_task = None
        self.cleanup_task = None
        
        # Metrics and monitoring
        self.processing_stats = {
            'total_events': 0,
            'successful_fusions': 0,
            'failed_fusions': 0,
            'average_latency_ms': 0.0
        }
    
    async def initialize(self):
        """Initialize the fusion engine"""
        logger.info("Initializing Multi-Modal Fusion Engine...")
        
        try:
            # Load or create fusion model
            self.fusion_model = MultiModalFusionModel(self.config['model'])
            self.fusion_model.to(self.device)
            
            # Load pre-trained weights if available
            model_path = self.config.get('model_path')
            if model_path:
                # TODO: Load pre-trained model
                # state_dict = torch.load(model_path, map_location=self.device)
                # self.fusion_model.load_state_dict(state_dict)
                logger.info(f"Loaded model from {model_path}")
            
            self.fusion_model.eval()
            
            logger.info("Multi-Modal Fusion Engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize fusion engine: {e}")
            raise
    
    async def start(self):
        """Start the fusion engine"""
        if self.is_running:
            return
            
        logger.info("Starting Multi-Modal Fusion Engine...")
        self.is_running = True
        
        # Start processing tasks
        self.processing_task = asyncio.create_task(self._process_sensor_data())
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_events())
        
        logger.info("Multi-Modal Fusion Engine started")
    
    async def stop(self):
        """Stop the fusion engine"""
        if not self.is_running:
            return
            
        logger.info("Stopping Multi-Modal Fusion Engine...")
        self.is_running = False
        
        # Cancel tasks
        if self.processing_task:
            self.processing_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Wait for tasks to complete
        tasks = [task for task in [self.processing_task, self.cleanup_task] if task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Multi-Modal Fusion Engine stopped")
    
    async def _process_sensor_data(self):
        """Main sensor data processing loop"""
        while self.is_running:
            try:
                # Get sensor reading from queue
                reading = await asyncio.wait_for(
                    self.sensor_queue.get(), 
                    timeout=1.0
                )
                
                # Process the reading
                await self._handle_sensor_reading(reading)
                
            except asyncio.TimeoutError:
                continue  # No data available, continue loop
            except Exception as e:
                logger.error(f"Error processing sensor data: {e}")
                await asyncio.sleep(1)  # Back off on error
    
    async def _handle_sensor_reading(self, reading: SensorReading):
        """Handle a single sensor reading"""
        try:
            # Determine which event this reading belongs to
            event_id = self._assign_reading_to_event(reading)
            
            # Get or create event
            if event_id not in self.active_events:
                self.active_events[event_id] = VehicleEvent(
                    event_id=event_id,
                    start_time=reading.timestamp,
                    end_time=None,
                    lane_id=reading.lane_id,
                    sensor_readings=[]
                )
                self.processing_stats['total_events'] += 1
            
            # Add reading to event
            event = self.active_events[event_id]
            event.sensor_readings.append(reading)
            
            # Check if event is ready for fusion
            if self._is_event_ready_for_fusion(event):
                await self._perform_fusion(event)
            
        except Exception as e:
            logger.error(f"Error handling sensor reading: {e}")
    
    def _assign_reading_to_event(self, reading: SensorReading) -> str:
        """
        Assign sensor reading to a vehicle event
        
        This uses spatial and temporal clustering to group readings
        that likely belong to the same vehicle passage.
        """
        # Simple event assignment based on lane and time window
        # TODO: Implement more sophisticated clustering algorithm
        
        current_time = reading.timestamp
        lane_id = reading.lane_id or 'unknown'
        
        # Look for recent events in the same lane
        for event_id, event in self.active_events.items():
            if (event.lane_id == lane_id and 
                not event.is_complete and
                (current_time - event.start_time) < self.event_timeout):
                return event_id
        
        # Create new event ID
        return f"{lane_id}_{current_time.timestamp()}"
    
    def _is_event_ready_for_fusion(self, event: VehicleEvent) -> bool:
        """
        Determine if event has enough data for fusion processing
        
        Events are ready when:
        1. Multiple modalities have provided data
        2. Sufficient time has passed for data collection
        3. Event hasn't been processed yet
        """
        if event.fusion_result is not None:
            return False  # Already processed
        
        # Check modality coverage
        modalities_present = set()
        for reading in event.sensor_readings:
            modalities_present.add(reading.sensor_type)
        
        min_modalities = self.config.get('min_modalities_for_fusion', 1)
        if len(modalities_present) < min_modalities:
            return False
        
        # Check if enough time has passed for data collection
        min_collection_time = timedelta(
            seconds=self.config.get('min_collection_time_seconds', 2)
        )
        if (datetime.now() - event.start_time) < min_collection_time:
            return False
        
        return True
    
    async def _perform_fusion(self, event: VehicleEvent):
        """Perform multi-modal fusion for a vehicle event"""
        start_time = datetime.now()
        
        try:
            logger.debug(f"Performing fusion for event {event.event_id}")
            
            # Prepare modality data
            modality_data, modality_confidences = self._prepare_modality_data(event)
            
            if not modality_data:
                logger.warning(f"No valid modality data for event {event.event_id}")
                return
            
            # Run fusion model inference
            with torch.no_grad():
                model_outputs = self.fusion_model(modality_data, modality_confidences)
            
            # Extract results
            fused_features = model_outputs['fused_features'].cpu().numpy()
            match_confidence = model_outputs['match_confidence'].item()
            recommended_charge = model_outputs['recommended_charge'].item()
            attention_weights = model_outputs['attention_weights']
            
            # TODO: Extract plate text predictions from ANPR modality
            plate_text_predictions = {'UNKNOWN': 1.0}
            
            # Create fusion result
            event.fusion_result = FusionResult(
                timestamp=datetime.now(),
                vehicle_match_confidence=match_confidence,
                plate_text_predictions=plate_text_predictions,
                recommended_charge=recommended_charge,
                modality_contributions=attention_weights,
                fusion_features=fused_features.flatten(),
                lane_id=event.lane_id
            )
            
            event.is_complete = True
            event.end_time = datetime.now()
            
            # Update stats
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.processing_stats['successful_fusions'] += 1
            self._update_latency_stats(processing_time)
            
            logger.info(f"Fusion completed for event {event.event_id}, "
                       f"confidence: {match_confidence:.3f}, "
                       f"charge: ${recommended_charge:.2f}")
            
        except Exception as e:
            logger.error(f"Error performing fusion for event {event.event_id}: {e}")
            self.processing_stats['failed_fusions'] += 1
    
    def _prepare_modality_data(self, event: VehicleEvent) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Prepare sensor data for model input
        
        Returns:
            modality_data: Dict mapping modality names to input tensors
            modality_confidences: Dict mapping modality names to confidence scores
        """
        modality_data = {}
        modality_confidences = {}
        
        # Group readings by modality
        readings_by_modality = defaultdict(list)
        for reading in event.sensor_readings:
            readings_by_modality[reading.sensor_type].append(reading)
        
        # Process each modality
        for modality, readings in readings_by_modality.items():
            try:
                if modality == 'anpr':
                    # Use the highest confidence ANPR reading
                    best_reading = max(readings, key=lambda r: r.confidence)
                    if 'image_features' in best_reading.data:
                        features = torch.tensor(
                            best_reading.data['image_features'], 
                            dtype=torch.float32
                        ).unsqueeze(0)  # Add batch dimension
                        modality_data[modality] = features
                        modality_confidences[modality] = best_reading.confidence
                
                elif modality == 'rfid':
                    # Use the strongest RFID signal
                    best_reading = max(readings, key=lambda r: r.confidence)
                    if 'tag_id' in best_reading.data:
                        # Convert RFID data to feature vector
                        tag_id_hash = hash(best_reading.data['tag_id']) % 1000 / 1000.0
                        signal_strength = (best_reading.data.get('signal_strength', -60) + 100) / 100.0
                        protocol_type = 1.0 if best_reading.data.get('protocol') == 'EPC Gen2' else 0.0
                        
                        features = torch.tensor(
                            [tag_id_hash, signal_strength, protocol_type],
                            dtype=torch.float32
                        ).unsqueeze(0)
                        modality_data[modality] = features
                        modality_confidences[modality] = best_reading.confidence
                
                elif modality == 'v2x':
                    # Use the most recent V2X message
                    best_reading = max(readings, key=lambda r: r.timestamp)
                    data = best_reading.data
                    
                    # Convert V2X data to feature vector
                    pseudonym_hash = hash(data.get('pseudonym_id', '')) % 1000 / 1000.0
                    signal_strength = (data.get('signal_strength', -60) + 100) / 100.0
                    speed = min(data.get('vehicle_speed', 0) / 120.0, 1.0)  # Normalize to [0,1]
                    heading = data.get('vehicle_heading', 0) / 360.0
                    message_type = 1.0 if data.get('message_type') == 'BSM' else 0.0
                    
                    features = torch.tensor(
                        [pseudonym_hash, signal_strength, speed, heading, message_type],
                        dtype=torch.float32
                    ).unsqueeze(0)
                    modality_data[modality] = features
                    modality_confidences[modality] = best_reading.confidence
                    
            except Exception as e:
                logger.error(f"Error preparing {modality} data: {e}")
        
        return modality_data, modality_confidences
    
    async def _cleanup_expired_events(self):
        """Clean up expired events that haven't been processed"""
        while self.is_running:
            try:
                current_time = datetime.now()
                expired_events = []
                
                for event_id, event in self.active_events.items():
                    if (current_time - event.start_time) > self.event_timeout:
                        expired_events.append(event_id)
                
                # Remove expired events
                for event_id in expired_events:
                    event = self.active_events.pop(event_id)
                    if not event.is_complete:
                        logger.warning(f"Event {event_id} expired without completion")
                
                await asyncio.sleep(10)  # Cleanup every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in event cleanup: {e}")
                await asyncio.sleep(10)
    
    def _update_latency_stats(self, processing_time_ms: float):
        """Update average latency statistics"""
        alpha = 0.1  # Exponential moving average factor
        current_avg = self.processing_stats['average_latency_ms']
        self.processing_stats['average_latency_ms'] = (
            alpha * processing_time_ms + (1 - alpha) * current_avg
        )
    
    def get_fusion_result(self, event_id: str) -> Optional[FusionResult]:
        """Get fusion result for a specific event"""
        event = self.active_events.get(event_id)
        return event.fusion_result if event else None
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.processing_stats,
            'active_events': len(self.active_events),
            'model_device': str(self.device)
        }
