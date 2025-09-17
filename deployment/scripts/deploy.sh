#!/bin/bash
# EFMM-Toll System Deployment Script

set -e

echo "🚀 Starting EFMM-Toll System Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker and try again."
        exit 1
    fi
    
    print_status "Docker is ready ✓"
}

# Check if Docker Compose is installed
check_docker_compose() {
    print_status "Checking Docker Compose installation..."
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose and try again."
        exit 1
    fi
    print_status "Docker Compose is ready ✓"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p config/{rsu,vehicle,aggregator,audit}
    mkdir -p config/{prometheus,grafana/dashboards,postgres}
    mkdir -p logs
    mkdir -p data
    
    print_status "Directories created ✓"
}

# Create default configuration files
create_configs() {
    print_status "Creating default configuration files..."
    
    # Create basic prometheus config
    cat > config/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'efmm-services'
    static_configs:
      - targets: ['rsu-edge:8001', 'vehicle-obu:8002', 'aggregator:8003', 'audit-ledger:8004']
    metrics_path: /metrics
    scrape_interval: 10s
EOF

    # Create postgres init script
    cat > config/postgres/init.sql << EOF
-- EFMM-Toll Database Initialization
CREATE DATABASE efmm_toll;
CREATE USER efmm_user WITH PASSWORD 'efmm_password';
GRANT ALL PRIVILEGES ON DATABASE efmm_toll TO efmm_user;
EOF

    print_status "Configuration files created ✓"
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    cd deployment/docker
    
    docker-compose build --no-cache
    
    if [ $? -eq 0 ]; then
        print_status "Docker images built successfully ✓"
    else
        print_error "Failed to build Docker images"
        exit 1
    fi
    
    cd ../..
}

# Start services
start_services() {
    print_status "Starting EFMM-Toll services..."
    
    cd deployment/docker
    
    # Start infrastructure services first
    docker-compose up -d redis postgres prometheus grafana
    
    # Wait for infrastructure to be ready
    print_status "Waiting for infrastructure services to be ready..."
    sleep 10
    
    # Start EFMM services
    docker-compose up -d audit-ledger aggregator rsu-edge vehicle-obu
    
    if [ $? -eq 0 ]; then
        print_status "Services started successfully ✓"
    else
        print_error "Failed to start services"
        exit 1
    fi
    
    cd ../..
}

# Check service health
check_health() {
    print_status "Checking service health..."
    
    # Wait for services to start
    sleep 30
    
    services=("audit-ledger:8004" "aggregator:8003" "rsu-edge:8001" "vehicle-obu:8002")
    
    for service in "${services[@]}"; do
        service_name=$(echo $service | cut -d':' -f1)
        port=$(echo $service | cut -d':' -f2)
        
        if curl -f -s http://localhost:$port/health > /dev/null; then
            print_status "$service_name is healthy ✓"
        else
            print_warning "$service_name health check failed"
        fi
    done
}

# Display service URLs
display_urls() {
    print_status "EFMM-Toll System is running!"
    echo ""
    echo "Service URLs:"
    echo "  - RSU Edge:          http://localhost:8001"
    echo "  - Vehicle OBU:       http://localhost:8002"
    echo "  - Aggregator:        http://localhost:8003"
    echo "  - Audit Ledger:      http://localhost:8004"
    echo "  - Grafana:           http://localhost:3000 (admin/admin)"
    echo "  - Prometheus:        http://localhost:9090"
    echo ""
    echo "To stop the system: ./deployment/scripts/stop.sh"
    echo "To view logs: docker-compose -f deployment/docker/docker-compose.yml logs -f"
}

# Main deployment flow
main() {
    print_status "EFMM-Toll System Deployment Starting..."
    
    check_docker
    check_docker_compose
    create_directories
    create_configs
    build_images
    start_services
    check_health
    display_urls
    
    print_status "Deployment completed successfully! 🎉"
}

# Handle script arguments
case "${1:-}" in
    "build")
        build_images
        ;;
    "start")
        start_services
        ;;
    "health")
        check_health
        ;;
    *)
        main
        ;;
esac
