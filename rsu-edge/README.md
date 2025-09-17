# RSU Edge Module

The RSU Edge Module runs on roadside compute infrastructure and provides:
- Multi-modal sensor integration (ANPR, RFID, V2X)
- On-device fusion inference
- Federated learning participation
- Ephemeral token management
- Zero-knowledge payment verification
- TEE-backed attestations

## Components
- **Sensor Adapters**: Camera ANPR, RFID readers, V2X stack
- **Fusion Engine**: Multi-modal vehicle recognition
- **FL Client**: Federated learning participant
- **Token Orchestrator**: Ephemeral token issuance and validation
- **Payment Verifier**: ZK proof verification with TEE
- **Audit Buffer**: Local attestation storage

## Usage
```bash
# Start RSU service
python main.py --config config/rsu_config.yaml

# Run in simulation mode
python main.py --mode simulation --config config/sim_config.yaml
```
