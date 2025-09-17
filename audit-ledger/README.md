# Distributed Audit Ledger

## Overview
The Distributed Audit Ledger provides a blockchain-based immutable audit trail for the EFMM-Toll system. It ensures transparency, accountability, and tamper-proof record keeping for all toll transactions, vehicle interactions, and system operations.

## Architecture

### Components
- **LedgerNode**: Core blockchain node managing blocks and consensus
- **BlockchainStorage**: Persistent storage for blockchain data using SQLite
- **AttestationManager**: TEE attestation creation and verification
- **AuditAPI**: REST API for querying audit records
- **AuditLedgerService**: Main orchestration service

### Key Features
- Immutable audit records stored in blockchain blocks
- TEE-based attestations for entity verification  
- Distributed consensus for network integrity
- REST API for audit record queries
- Persistent storage with SQLite backend
- Real-time block creation and synchronization

## Usage

### Starting the Service
```python
from main import AuditLedgerService
import asyncio

async def run():
    service = AuditLedgerService("audit_config.yaml")
    await service.run()

asyncio.run(run())
```

### Configuration
Configure the service using `audit_config.yaml`:
```yaml
service:
  name: "efmm-audit-ledger"
  host: "localhost" 
  port: 8004

ledger:
  node_id: "audit-node-001"
  block_size: 10
  block_interval: 30
```

### API Endpoints
- `GET /blocks/{index}` - Retrieve block by index
- `GET /records` - Search audit records with filters
- `GET /attestations/{id}` - Get attestation details
- `GET /health` - Service health check
- `GET /stats` - Blockchain statistics

### Adding Audit Records
Records are automatically added by other EFMM-Toll components:
- RSU transactions and token validations
- Vehicle payment proofs and interactions  
- Aggregator model updates and FL rounds
- System attestations and security events

## Dependencies
- SQLite3 for persistent storage
- Standard Python libraries (asyncio, hashlib, json)
- Future: Web framework for REST API (FastAPI/aiohttp)
- Future: Cryptographic libraries for digital signatures

## Security
- TEE attestations verify entity authenticity
- Blockchain ensures record immutability
- Digital signatures prevent tampering
- Distributed consensus maintains integrity
