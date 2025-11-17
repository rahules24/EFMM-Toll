"""
Distributed Audit Ledger - Main Service
Blockchain-based audit trail for EFMM-Toll system
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any
from ledger_node import LedgerNode
from audit_config import AuditLedgerConfig
from attestation_manager import AttestationManager
from audit_verifier import AuditVerifier
from consensus_manager import ConsensusManager

logger = logging.getLogger(__name__)


class AuditLedgerService:
    """Main service for the distributed audit ledger"""
    
    def __init__(self, config_path: str = "audit_config.yaml"):
        self.config = AuditLedgerConfig.from_yaml(config_path)
        self.ledger_node = None
        self.attestation_manager = None
        self.audit_verifier = None
        self.consensus_manager = None
        self.is_running = False
    
    async def initialize(self):
        """Initialize the audit ledger service"""
        logger.info("Initializing Audit Ledger Service...")
        
        # Initialize ledger node
        self.ledger_node = LedgerNode(self.config.to_dict()['ledger'])
        await self.ledger_node.initialize()
        
        # Initialize attestation manager
        self.attestation_manager = AttestationManager(self.config.to_dict()['security'])
        await self.attestation_manager.initialize()
        
        # Initialize audit verifier
        self.audit_verifier = AuditVerifier(self.config.to_dict()['ledger'])
        await self.audit_verifier.initialize()
        
        # Initialize consensus manager
        self.consensus_manager = ConsensusManager(self.config.to_dict()['ledger'])
        await self.consensus_manager.initialize()
        
        logger.info("All audit ledger components initialized")
    
    async def start(self):
        """Start the audit ledger service"""
        logger.info("Starting Audit Ledger Service...")
        
        if not self.ledger_node:
            await self.initialize()
        
        # Start all components
        await self.attestation_manager.start()
        await self.audit_verifier.start()
        await self.consensus_manager.start()
        await self.ledger_node.start()
        
        self.is_running = True
        logger.info(f"Audit Ledger Service started on {self.config.host}:{self.config.port}")
    
    async def stop(self):
        """Stop the audit ledger service"""
        logger.info("Stopping Audit Ledger Service...")
        
        self.is_running = False
        
        # Stop all components
        if self.consensus_manager:
            await self.consensus_manager.stop()
        
        if self.audit_verifier:
            await self.audit_verifier.stop()
        
        if self.ledger_node:
            await self.ledger_node.stop()
        
        if self.attestation_manager:
            await self.attestation_manager.stop()
    
    async def run(self):
        """Run the audit ledger service"""
        try:
            await self.start()
            
            # Keep service running
            while self.is_running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Error in audit ledger service: {e}")
        finally:
            await self.stop()


async def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    service = AuditLedgerService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
