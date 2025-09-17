# EFMM-Toll System Architecture

## Overview

The Ephemeral Federated Multi-Modal Tolling (EFMM-Toll) system is a comprehensive solution for privacy-preserving, federated learning-enhanced electronic toll collection. The system consists of four main components that work together to provide secure, efficient, and privacy-focused toll collection services.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EFMM-Toll System                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐│
│  │   RSU Edge  │  │ Vehicle OBU │  │ Aggregator  │  │  Audit   ││
│  │   Module    │  │     App     │  │   Service   │  │  Ledger  ││
│  │             │  │             │  │             │  │          ││
│  │ • Detection │  │ • V2X Comm  │  │ • FL Coord  │  │ • Chain  ││
│  │ • Fusion    │  │ • Wallet    │  │ • Privacy   │  │ • Verify ││
│  │ • Tokens    │  │ • Privacy   │  │ • Models    │  │ • Audit  ││
│  │ • Payment   │  │ • FL Client │  │ • Registry  │  │ • TEE    ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────┘│
└─────────────────────────────────────────────────────────────────┘
                                │
                        ┌───────────────┐
                        │ Shared Utils  │
                        │ • Crypto      │
                        │ • V2X Proto   │
                        │ • Database    │
                        └───────────────┘
```

## Component Architecture

### 1. RSU Edge Module

**Purpose**: Roadside Unit for vehicle detection and toll collection
**Location**: Deployed at toll collection points

#### Components:
- **Sensor Adapters**: Interface with cameras, LiDAR, radar
- **Fusion Engine**: Multi-modal sensor data fusion using ML
- **Token Orchestrator**: Ephemeral token lifecycle management
- **Payment Verifier**: Zero-knowledge proof verification with TEE
- **FL Client**: Federated learning participation
- **Audit Buffer**: Local audit record storage

#### Data Flow:
```
Sensors → Fusion → Detection → Token Gen → Payment → Audit
    ↓                              ↓           ↓        ↓
  FL Model                    V2X Comm    ZK Verify  Ledger
```

### 2. Vehicle OBU App

**Purpose**: On-board unit for vehicles
**Location**: Installed in vehicles

#### Components:
- **V2X Client**: Vehicle-to-infrastructure communication
- **Token Holder**: Ephemeral token management
- **Wallet**: Balance management and payment processing
- **FL Client**: Federated learning participation
- **Privacy Manager**: Pseudonym rotation and location obfuscation

#### Data Flow:
```
V2X Receive → Token Process → Payment Gen → Privacy → FL Update
     ↓              ↓             ↓          ↓         ↓
  RSU Comm      Wallet Check   ZK Proof   Anonymize  Model
```

### 3. Federated Aggregator Service

**Purpose**: Coordinate federated learning across RSUs and vehicles
**Location**: Cloud or edge infrastructure

#### Components:
- **Aggregation Server**: FL round coordination
- **Model Repository**: Model versioning and storage
- **Privacy Engine**: Secure aggregation and differential privacy
- **Participant Registry**: RSU and vehicle registration

#### Data Flow:
```
Registration → Round Init → Collect Updates → Aggregate → Distribute
     ↓             ↓             ↓             ↓           ↓
  Validate    Send Challenge  Verify Update  Privacy    New Model
```

### 4. Distributed Audit Ledger

**Purpose**: Immutable audit trail for all system operations
**Location**: Distributed across multiple nodes

#### Components:
- **Ledger Node**: Blockchain node implementation
- **Attestation Manager**: TEE attestation handling
- **Audit Configuration**: System configuration

#### Data Flow:
```
Audit Records → Validation → Block Creation → Consensus → Storage
      ↓             ↓            ↓             ↓          ↓
   TEE Attest   Verify Hash  Add to Chain  Sync Peers  Persist
```

## Data Architecture

### Data Types:

1. **Sensor Data**
   - Camera images/video streams
   - LiDAR point clouds
   - Radar detection data
   - Environmental sensors

2. **Transaction Data**
   - Payment proofs (zero-knowledge)
   - Token exchanges
   - Toll amounts
   - Vehicle classifications

3. **ML Models**
   - Vehicle detection models
   - Classification models
   - Federated learning updates
   - Privacy-preserving aggregated models

4. **Audit Data**
   - Transaction records
   - System events
   - TEE attestations
   - Privacy compliance logs

### Data Flow Patterns:

1. **Toll Collection Flow**:
   ```
   Vehicle Approach → Detection → Token Exchange → Payment → Receipt
   ```

2. **Federated Learning Flow**:
   ```
   Local Training → Model Update → Aggregation → Distribution → Update
   ```

3. **Audit Flow**:
   ```
   System Event → Audit Record → Validation → Block Addition → Ledger
   ```

## Security Architecture

### Privacy Protection:
- **Zero-Knowledge Proofs**: Payment verification without revealing balance
- **Pseudonym Rotation**: Regular identity changes for vehicles
- **Location Obfuscation**: Approximate location reporting
- **Differential Privacy**: Noise addition to ML model updates

### Security Measures:
- **TEE Integration**: Hardware-based security for critical operations
- **Digital Signatures**: All communications cryptographically signed
- **Encrypted Communication**: End-to-end encryption for V2X messages
- **Audit Trail**: Immutable record of all system operations

### Trust Model:
- **Zero-Trust Architecture**: Verify all entities and communications
- **Decentralized Verification**: Multiple validators for critical operations
- **Hardware Root of Trust**: TEE-based attestations
- **Cryptographic Proofs**: Mathematical verification of claims

## Network Architecture

### Communication Protocols:
- **V2X (DSRC/C-V2X)**: Vehicle-to-infrastructure communication
- **HTTP/REST APIs**: Service-to-service communication
- **WebSocket**: Real-time updates and streaming
- **gRPC**: High-performance inter-service communication

### Network Topology:
```
┌─────────────┐     V2X      ┌─────────────┐
│   Vehicle   │ ◄──────────► │  RSU Edge   │
│    OBU      │              │   Module    │
└─────────────┘              └─────────────┘
                                    │ HTTP/gRPC
                              ┌─────────────┐
                              │ Aggregator  │
                              │  Service    │
                              └─────────────┘
                                    │ HTTP/gRPC
                              ┌─────────────┐
                              │   Audit     │
                              │   Ledger    │
                              └─────────────┘
```

## Scalability Considerations

### Horizontal Scaling:
- **RSU Deployment**: Multiple RSUs per highway segment
- **Load Balancing**: Distribute traffic across aggregator instances
- **Sharded Ledger**: Partition audit ledger for performance
- **Edge Computing**: Process data close to collection points

### Performance Optimization:
- **Caching**: Frequent data cached at multiple levels
- **Batch Processing**: Group operations for efficiency
- **Async Processing**: Non-blocking operations throughout
- **Model Compression**: Efficient ML model representations

### Reliability:
- **Redundancy**: Multiple instances of critical components
- **Failover**: Automatic switching to backup systems
- **Data Replication**: Multiple copies of critical data
- **Health Monitoring**: Continuous system health checks

## Deployment Architecture

### Infrastructure Requirements:
- **RSU Hardware**: Edge computing devices with V2X capability
- **Vehicle Hardware**: OBU with cellular/V2X connectivity
- **Cloud Infrastructure**: Aggregator and ledger hosting
- **Network Connectivity**: V2X, cellular, and internet connectivity

### Deployment Patterns:
- **Containerized Deployment**: Docker containers for all services
- **Orchestration**: Kubernetes for container management
- **CI/CD Pipeline**: Automated testing and deployment
- **Monitoring**: Comprehensive logging and metrics collection
