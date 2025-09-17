# EFMM-Toll: Ephemeral Federated Multi-Modal Tolling System

## 🚗 Overview

EFMM-Toll is a next-generation intelligent tolling system that combines **roadside infrastructure**, **vehicle on-board units**, **federated learning**, and **blockchain audit trails** to create a privacy-preserving, efficient, and scalable toll collection solution.

### Key Features

- 🛣️ **Multi-Modal Sensor Fusion**: Advanced vehicle detection using cameras, LiDAR, and radar
- 🔐 **Privacy-First Design**: Zero-knowledge proofs and differential privacy protection
- 🤝 **Federated Learning**: Collaborative ML model training without data sharing
- 📋 **Blockchain Audit**: Immutable audit trail with TEE attestations
- ⚡ **Ephemeral Tokens**: Short-lived cryptographic tokens for secure communications
- 🌐 **V2X Communication**: Vehicle-to-infrastructure communication protocols

## System Architecture

### Core Components
1. **RSU Edge Module** - Roadside computation and sensor fusion
2. **Vehicle OBU App** - On-board unit and mobile applications
3. **Federated Aggregator Service** - Model training coordination
4. **Distributed Audit Ledger** - Cryptographic audit trail
5. **Shared Components** - Common utilities and protocols

## Directory Structure
```
efmm-toll/
├── rsu-edge/           # RSU Edge Module
├── vehicle-obu/        # Vehicle OBU Application
├── aggregator/         # Federated Learning Aggregator
├── audit-ledger/       # Distributed Audit Ledger
├── shared/             # Shared Components and Utilities
├── deployment/         # Deployment Configurations
├── docs/              # Documentation
└── tests/             # Integration Tests
```

## 🚀 Quick Start

### Prerequisites

Before running EFMM-Toll, ensure you have the following installed:

- **Python 3.11+** (recommended: 3.11.5 or higher)
- **Docker Desktop** (for containerized deployment)
- **Git** (for cloning the repository)
- **Minimum System Requirements**:
  - 8GB RAM (16GB recommended)
  - 20GB free disk space
  - Multi-core CPU (4+ cores recommended)

### Option 1: Docker Deployment (Recommended)

This is the easiest way to get EFMM-Toll running with all services.

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd Codebase
   ```

2. **Start Docker Desktop**
   - Ensure Docker Desktop is running on your system
   - Verify with: `docker --version` and `docker-compose --version`

3. **Deploy the System**
   ```bash
   # Make deployment script executable (Linux/Mac)
   chmod +x deployment/scripts/deploy.sh
   
   # Run the deployment script
   ./deployment/scripts/deploy.sh
   ```
   
   **For Windows PowerShell:**
   ```powershell
   # Navigate to docker directory
   cd deployment\docker
   
   # Build and start all services
   docker-compose up --build -d
   ```

4. **Verify Services are Running**
   
   Wait 2-3 minutes for all services to start, then check:
   ```bash
   # Check service health
   curl http://localhost:8001/health  # RSU Edge Module
   curl http://localhost:8002/health  # Vehicle OBU
   curl http://localhost:8003/health  # Federated Aggregator
   curl http://localhost:8004/health  # Audit Ledger
   ```

5. **Access Service Dashboards**
   - **RSU Edge Module**: http://localhost:8001
   - **Vehicle OBU**: http://localhost:8002
   - **Federated Aggregator**: http://localhost:8003
   - **Audit Ledger**: http://localhost:8004
   - **Grafana Monitoring**: http://localhost:3000 (admin/admin)
   - **Prometheus Metrics**: http://localhost:9090

### Option 2: Local Development Setup

For development and testing individual components:

1. **Clone and Setup Environment**
   ```bash
   git clone <repository-url>
   cd Codebase
   
   # Create Python virtual environment
   python -m venv efmm-env
   
   # Activate virtual environment
   # On Windows:
   efmm-env\Scripts\activate
   # On Linux/Mac:
   source efmm-env/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   # Install all required packages
   pip install -r deployment/requirements.txt
   ```

3. **Start Services Individually**

   **Terminal 1 - Audit Ledger (Start First)**
   ```bash
   cd audit-ledger
   python main.py
   ```
   Wait for "Audit Ledger Service started" message.

   **Terminal 2 - Federated Aggregator**
   ```bash
   cd aggregator
   python main.py
   ```
   Wait for "Aggregator Service started" message.

   **Terminal 3 - RSU Edge Module**
   ```bash
   cd rsu-edge
   python main.py
   ```
   Wait for "RSU Edge Service started" message.

   **Terminal 4 - Vehicle OBU**
   ```bash
   cd vehicle-obu
   python main.py
   ```
   Wait for "Vehicle OBU Service started" message.

4. **Verify Local Setup**
   ```bash
   # Test each service
   curl http://localhost:8004/health  # Audit Ledger
   curl http://localhost:8003/health  # Aggregator
   curl http://localhost:8001/health  # RSU Edge
   curl http://localhost:8002/health  # Vehicle OBU
   ```

### Testing the System

1. **Basic Functionality Test**
   ```bash
   # Test vehicle detection simulation
   curl -X POST http://localhost:8001/api/simulate/vehicle-detection \
        -H "Content-Type: application/json" \
        -d '{"vehicle_id": "test-vehicle-001", "location": {"lat": 40.7128, "lon": -74.0060}}'
   
   # Test toll payment simulation
   curl -X POST http://localhost:8002/api/simulate/toll-payment \
        -H "Content-Type: application/json" \
        -d '{"amount": 2.50, "rsu_id": "rsu-001"}'
   ```

2. **Check Audit Records**
   ```bash
   # View audit trail
   curl http://localhost:8004/api/audit/records
   ```

3. **Monitor Federated Learning**
   ```bash
   # Check FL participants
   curl http://localhost:8003/api/participants
   
   # View model status
   curl http://localhost:8003/api/models/status
   ```

### Stopping the System

**For Docker Deployment:**
```bash
# Stop all services
./deployment/scripts/stop.sh

# Or manually with docker-compose
cd deployment/docker
docker-compose down

# To also remove volumes and clean up
docker-compose down -v
```

**For Local Development:**
- Press `Ctrl+C` in each terminal running the services
- Deactivate virtual environment: `deactivate`

### Troubleshooting

**Common Issues:**

1. **Port Already in Use**
   ```bash
   # Check what's using the ports
   netstat -ano | findstr :8001  # Windows
   lsof -i :8001                 # Linux/Mac
   ```

2. **Docker Issues**
   ```bash
   # Restart Docker Desktop
   # Clear Docker cache
   docker system prune -a
   ```

3. **Python Dependencies**
   ```bash
   # Upgrade pip and reinstall
   pip install --upgrade pip
   pip install -r deployment/requirements.txt --force-reinstall
   ```

4. **Service Health Check Fails**
   ```bash
   # Check service logs
   docker-compose -f deployment/docker/docker-compose.yml logs <service-name>
   
   # For local development, check terminal output
   ```

### Next Steps

Once the system is running:
1. Explore the Grafana dashboards at http://localhost:3000
2. Review the audit records in the blockchain ledger
3. Test federated learning by running multiple RSU instances
4. Experiment with the V2X communication protocols

## Contributing
[TODO: Add contribution guidelines]

## License
[TODO: Add license information]
