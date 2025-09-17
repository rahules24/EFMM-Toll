"""
Consensus Engine
Implements consensus mechanism for the distributed audit ledger
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from ledger_node import Block

logger = logging.getLogger(__name__)


@dataclass
class ConsensusVote:
    """Vote structure for consensus"""
    voter_id: str
    block_hash: str
    vote: str  # 'approve', 'reject'
    timestamp: str
    signature: str


class ConsensusEngine:
    """Consensus engine for distributed ledger agreement"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.node_id = config.get('node_id', 'node-001')
        self.consensus_algorithm = config.get('consensus_algorithm', 'proof_of_authority')
        self.pending_blocks: Dict[str, Block] = {}
        self.votes: Dict[str, List[ConsensusVote]] = {}
        self.authorized_nodes = config.get('authorized_nodes', [self.node_id])
        self.is_running = False
    
    async def initialize(self):
        """Initialize consensus engine"""
        logger.info(f"Initializing Consensus Engine: {self.consensus_algorithm}")
        self.is_running = True
    
    async def start(self):
        """Start consensus engine"""
        logger.info("Starting Consensus Engine...")
        
        # Start consensus monitoring task
        asyncio.create_task(self.monitor_consensus())
    
    async def stop(self):
        """Stop consensus engine"""
        logger.info("Stopping Consensus Engine...")
        self.is_running = False
    
    async def propose_block(self, block: Block) -> bool:
        """Propose a new block for consensus"""
        try:
            if not await self._is_authorized_proposer():
                logger.warning(f"Node {self.node_id} not authorized to propose blocks")
                return False
            
            block_hash = block.hash_value
            self.pending_blocks[block_hash] = block
            self.votes[block_hash] = []
            
            # Broadcast block proposal to other nodes
            await self._broadcast_block_proposal(block)
            
            logger.info(f"Proposed block {block.index} with hash {block_hash[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error proposing block: {e}")
            return False
    
    async def vote_on_block(self, block_hash: str, vote: str) -> bool:
        """Vote on a proposed block"""
        try:
            if not await self._is_authorized_voter():
                logger.warning(f"Node {self.node_id} not authorized to vote")
                return False
            
            if block_hash not in self.pending_blocks:
                logger.warning(f"Block {block_hash[:8]}... not found for voting")
                return False
            
            # Create vote
            vote_obj = ConsensusVote(
                voter_id=self.node_id,
                block_hash=block_hash,
                vote=vote,
                timestamp=datetime.now().isoformat(),
                signature=await self._sign_vote(block_hash, vote)
            )
            
            self.votes[block_hash].append(vote_obj)
            
            # Broadcast vote to other nodes
            await self._broadcast_vote(vote_obj)
            
            logger.info(f"Voted {vote} on block {block_hash[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error voting on block: {e}")
            return False
    
    async def check_consensus(self, block_hash: str) -> Optional[str]:
        """Check if consensus has been reached for a block"""
        try:
            if block_hash not in self.votes:
                return None
            
            votes = self.votes[block_hash]
            
            if self.consensus_algorithm == 'proof_of_authority':
                return await self._check_poa_consensus(votes)
            elif self.consensus_algorithm == 'simple_majority':
                return await self._check_majority_consensus(votes)
            else:
                logger.error(f"Unknown consensus algorithm: {self.consensus_algorithm}")
                return None
                
        except Exception as e:
            logger.error(f"Error checking consensus: {e}")
            return None
    
    async def finalize_block(self, block_hash: str) -> bool:
        """Finalize a block that has reached consensus"""
        try:
            consensus_result = await self.check_consensus(block_hash)
            
            if consensus_result == 'approved':
                # Remove from pending
                if block_hash in self.pending_blocks:
                    del self.pending_blocks[block_hash]
                if block_hash in self.votes:
                    del self.votes[block_hash]
                
                logger.info(f"Block {block_hash[:8]}... finalized and approved")
                return True
            elif consensus_result == 'rejected':
                # Remove rejected block
                if block_hash in self.pending_blocks:
                    del self.pending_blocks[block_hash]
                if block_hash in self.votes:
                    del self.votes[block_hash]
                
                logger.info(f"Block {block_hash[:8]}... rejected by consensus")
                return False
            else:
                # Consensus not yet reached
                return False
                
        except Exception as e:
            logger.error(f"Error finalizing block: {e}")
            return False
    
    async def monitor_consensus(self):
        """Monitor pending blocks for consensus"""
        while self.is_running:
            try:
                pending_hashes = list(self.pending_blocks.keys())
                
                for block_hash in pending_hashes:
                    await self.finalize_block(block_hash)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in consensus monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _is_authorized_proposer(self) -> bool:
        """Check if node is authorized to propose blocks"""
        return self.node_id in self.authorized_nodes
    
    async def _is_authorized_voter(self) -> bool:
        """Check if node is authorized to vote"""
        return self.node_id in self.authorized_nodes
    
    async def _check_poa_consensus(self, votes: List[ConsensusVote]) -> Optional[str]:
        """Check Proof of Authority consensus"""
        # In PoA, we need majority of authorized nodes to approve
        authorized_votes = [v for v in votes if v.voter_id in self.authorized_nodes]
        
        if len(authorized_votes) < len(self.authorized_nodes) // 2 + 1:
            return None  # Not enough votes yet
        
        approve_votes = sum(1 for v in authorized_votes if v.vote == 'approve')
        reject_votes = sum(1 for v in authorized_votes if v.vote == 'reject')
        
        if approve_votes > len(self.authorized_nodes) // 2:
            return 'approved'
        elif reject_votes > len(self.authorized_nodes) // 2:
            return 'rejected'
        else:
            return None  # Still waiting for more votes
    
    async def _check_majority_consensus(self, votes: List[ConsensusVote]) -> Optional[str]:
        """Check simple majority consensus"""
        total_votes = len(votes)
        if total_votes == 0:
            return None
        
        approve_votes = sum(1 for v in votes if v.vote == 'approve')
        reject_votes = sum(1 for v in votes if v.vote == 'reject')
        
        if approve_votes > total_votes // 2:
            return 'approved'
        elif reject_votes > total_votes // 2:
            return 'rejected'
        else:
            return None
    
    async def _sign_vote(self, block_hash: str, vote: str) -> str:
        """Sign a vote (placeholder)"""
        # TODO: Implement actual cryptographic signing
        import hashlib
        vote_data = f"{self.node_id}:{block_hash}:{vote}"
        return hashlib.sha256(vote_data.encode()).hexdigest()
    
    async def _broadcast_block_proposal(self, block: Block):
        """Broadcast block proposal to other nodes (placeholder)"""
        # TODO: Implement actual network broadcasting
        logger.debug(f"Broadcasting block proposal: {block.hash_value[:8]}...")
    
    async def _broadcast_vote(self, vote: ConsensusVote):
        """Broadcast vote to other nodes (placeholder)"""
        # TODO: Implement actual network broadcasting
        logger.debug(f"Broadcasting vote: {vote.vote} for {vote.block_hash[:8]}...")
