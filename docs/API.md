# EFMM-Toll API Documentation

## Overview
This document describes the APIs for all components of the Ephemeral Federated Multi-Modal Tolling (EFMM-Toll) system.

## RSU Edge Module API

### Base URL: `http://localhost:8001`

#### Health Check
- **GET** `/health`
- **Description**: Check service health status
- **Response**: 
  ```json
  {
    "status": "healthy",
    "timestamp": "2024-01-20T12:00:00Z",
    "version": "1.0.0"
  }
  ```

#### Vehicle Detection
- **POST** `/api/v1/detect`
- **Description**: Process vehicle detection from sensors
- **Request Body**:
  ```json
  {
    "sensor_data": {
      "camera": {...},
      "lidar": {...},
      "radar": {...}
    },
    "timestamp": "2024-01-20T12:00:00Z"
  }
  ```
- **Response**:
  ```json
  {
    "detection_id": "det_123456",
    "vehicles": [
      {
        "vehicle_id": "vehicle_001",
        "confidence": 0.95,
        "position": {...},
        "classification": "passenger_car"
      }
    ]
  }
  ```

#### Token Generation
- **POST** `/api/v1/tokens/generate`
- **Description**: Generate ephemeral token for vehicle
- **Request Body**:
  ```json
  {
    "vehicle_id": "vehicle_001",
    "detection_confidence": 0.95,
    "location": {...}
  }
  ```
- **Response**:
  ```json
  {
    "token_id": "token_abc123",
    "expires_at": "2024-01-20T12:05:00Z",
    "challenge": "challenge_data"
  }
  ```

#### Payment Verification
- **POST** `/api/v1/payments/verify`
- **Description**: Verify zero-knowledge payment proof
- **Request Body**:
  ```json
  {
    "token_id": "token_abc123",
    "zk_proof": "proof_data",
    "toll_amount": 2.50
  }
  ```
- **Response**:
  ```json
  {
    "verified": true,
    "transaction_id": "tx_789012",
    "receipt": {...}
  }
  ```

## Vehicle OBU App API

### Base URL: `http://localhost:8002`

#### Wallet Balance
- **GET** `/api/v1/wallet/balance`
- **Description**: Get current wallet balance
- **Response**:
  ```json
  {
    "balance": 25.50,
    "currency": "USD",
    "last_updated": "2024-01-20T12:00:00Z"
  }
  ```

#### Payment Proof Generation
- **POST** `/api/v1/payments/generate-proof`
- **Description**: Generate zero-knowledge payment proof
- **Request Body**:
  ```json
  {
    "toll_amount": 2.50,
    "token_challenge": "challenge_data"
  }
  ```
- **Response**:
  ```json
  {
    "zk_proof": "proof_data",
    "proof_valid": true
  }
  ```

#### Privacy Settings
- **GET** `/api/v1/privacy/settings`
- **Description**: Get current privacy settings
- **Response**:
  ```json
  {
    "pseudonym_rotation_interval": 300,
    "location_obfuscation_enabled": true,
    "data_sharing_level": "minimal"
  }
  ```

## Federated Aggregator API

### Base URL: `http://localhost:8003`

#### Training Round Status
- **GET** `/api/v1/training/status`
- **Description**: Get current federated learning round status
- **Response**:
  ```json
  {
    "current_round": 15,
    "participants_count": 8,
    "round_status": "in_progress",
    "completion_percentage": 75.0
  }
  ```

#### Model Update
- **POST** `/api/v1/models/update`
- **Description**: Submit local model update
- **Request Body**:
  ```json
  {
    "participant_id": "rsu_001",
    "model_update": "encrypted_model_data",
    "round_number": 15
  }
  ```
- **Response**:
  ```json
  {
    "accepted": true,
    "next_round_eta": "2024-01-20T12:30:00Z"
  }
  ```

#### Global Model
- **GET** `/api/v1/models/global`
- **Description**: Get latest global model
- **Response**:
  ```json
  {
    "model_version": "v1.15",
    "model_data": "encrypted_model_data",
    "performance_metrics": {...}
  }
  ```

## Audit Ledger API

### Base URL: `http://localhost:8004`

#### Submit Audit Record
- **POST** `/api/v1/audit/submit`
- **Description**: Submit audit record to ledger
- **Request Body**:
  ```json
  {
    "record_type": "payment_verification",
    "entity_id": "rsu_001",
    "action": "verify_payment",
    "details": {...}
  }
  ```
- **Response**:
  ```json
  {
    "record_id": "audit_123456",
    "block_hash": "block_hash_data",
    "accepted": true
  }
  ```

#### Query Audit Trail
- **GET** `/api/v1/audit/query`
- **Description**: Query audit records
- **Query Parameters**:
  - `entity_id`: Filter by entity ID
  - `record_type`: Filter by record type
  - `from_date`: Start date
  - `to_date`: End date
  - `limit`: Maximum records to return
- **Response**:
  ```json
  {
    "records": [
      {
        "record_id": "audit_123456",
        "timestamp": "2024-01-20T12:00:00Z",
        "record_type": "payment_verification",
        "entity_id": "rsu_001",
        "verified": true
      }
    ],
    "total_count": 150
  }
  ```

#### Blockchain Status
- **GET** `/api/v1/blockchain/status`
- **Description**: Get blockchain status
- **Response**:
  ```json
  {
    "block_height": 1250,
    "last_block_hash": "hash_data",
    "pending_records": 5,
    "node_status": "synchronized"
  }
  ```

## Error Responses

All APIs use standard HTTP status codes and return error responses in the following format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Detailed error message",
    "details": {...}
  },
  "timestamp": "2024-01-20T12:00:00Z"
}
```

### Common Error Codes
- `INVALID_REQUEST`: Malformed request
- `UNAUTHORIZED`: Authentication required
- `FORBIDDEN`: Insufficient permissions
- `NOT_FOUND`: Resource not found
- `RATE_LIMITED`: Too many requests
- `INTERNAL_ERROR`: Server error

## Authentication

All API endpoints require authentication using JWT tokens:

```
Authorization: Bearer <jwt_token>
```

Tokens can be obtained through the authentication endpoint (to be implemented).

## Rate Limiting

API calls are rate-limited per endpoint:
- Health checks: 60 requests/minute
- Data queries: 30 requests/minute
- Transactions: 10 requests/minute

## WebSocket APIs

Real-time updates are available through WebSocket connections:

### RSU Edge Module
- **WS** `/ws/detections` - Real-time vehicle detection updates
- **WS** `/ws/payments` - Real-time payment status updates

### Vehicle OBU App
- **WS** `/ws/toll-events` - Real-time toll collection events

### Federated Aggregator
- **WS** `/ws/training` - Real-time training progress updates

### Audit Ledger
- **WS** `/ws/audit` - Real-time audit record notifications
