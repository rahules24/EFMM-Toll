# Federated Learning Aggregator Service

The Federated Learning Aggregator Service coordinates distributed training across RSUs and vehicles while preserving privacy.

## Key Features
- **Secure Aggregation**: Privacy-preserving model update aggregation
- **Differential Privacy**: Additional privacy protection layer
- **Model Versioning**: Track and manage model versions
- **Participant Management**: RSU and vehicle registration/coordination
- **Performance Monitoring**: Training metrics and convergence tracking

## Components
- **Aggregation Server**: Main coordination service
- **Model Repository**: Versioned model storage
- **Privacy Engine**: Differential privacy mechanisms
- **Participant Registry**: Active participant tracking
- **Metrics Dashboard**: Training progress visualization

## Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│     RSUs        │────│   Aggregator     │────│    Vehicles     │
│ (FL Clients)    │    │    Service       │    │ (FL Clients)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                       ┌──────────────┐
                       │    Model     │
                       │  Repository  │ 
                       └──────────────┘
```

## Usage
```bash
# Start aggregation service
python main.py --config config/aggregator_config.yaml

# Run with monitoring dashboard
python main.py --enable-dashboard --port 8080
```
