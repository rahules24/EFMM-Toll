# Vehicle OBU (On-Board Unit) Application

The Vehicle OBU Application runs in vehicles or as mobile applications and provides:
- V2X/DSRC communication with RSUs
- Ephemeral token management and storage
- Cryptographic wallet for payment proofs
- Optional federated learning participation
- Privacy-preserving toll transactions

## Components
- **V2X Client**: Vehicle-to-infrastructure communication
- **Token Holder**: Ephemeral token storage and validation
- **FL Client**: Optional federated learning participation
- **Wallet**: Cryptographic payment proof generation
- **Privacy Manager**: Personal data protection

## Supported Platforms
- **Embedded OBU**: ARM-based automotive hardware
- **Mobile App**: iOS/Android smartphone application
- **Telematics**: Integration with existing vehicle systems

## Usage
```bash
# Start OBU application
python main.py --config config/obu_config.yaml

# Run in mobile simulation mode
python main.py --mode mobile --config config/mobile_config.yaml

# Run vehicle simulator
python simulator.py --route config/test_route.json
```
