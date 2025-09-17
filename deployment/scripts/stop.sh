#!/bin/bash
# EFMM-Toll System Stop Script

set -e

echo "🛑 Stopping EFMM-Toll System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Stop services
stop_services() {
    print_status "Stopping EFMM-Toll services..."
    
    cd deployment/docker
    
    docker-compose down
    
    if [ $? -eq 0 ]; then
        print_status "Services stopped successfully ✓"
    else
        print_error "Error stopping services"
        exit 1
    fi
    
    cd ../..
}

# Clean up (optional)
cleanup() {
    if [ "${1:-}" = "--clean" ]; then
        print_warning "Cleaning up volumes and images..."
        
        cd deployment/docker
        
        # Remove volumes
        docker-compose down -v
        
        # Remove images
        docker rmi $(docker images "efmm/*" -q) 2>/dev/null || true
        
        print_status "Cleanup completed ✓"
        
        cd ../..
    fi
}

# Main
main() {
    stop_services
    cleanup "$1"
    
    print_status "EFMM-Toll System stopped successfully! 🏁"
}

main "$@"
