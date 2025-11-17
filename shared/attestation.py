from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class TEEAttestation:
    """TEE-generated attestation for payment verification"""
    attestation_id: str
    rsu_id: str
    ephemeral_token_id: str
    payment_proof_id: str
    verification_result: bool
    amount: float
    timestamp: datetime
    tee_signature: bytes
    sealed_evidence: Optional[bytes]  # Encrypted evidence for disputes