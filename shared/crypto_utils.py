"""
Cryptographic Utilities
Shared cryptographic functions for EFMM-Toll system
"""

import hashlib
import secrets
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import json


class CryptoUtils:
    """Utility class for cryptographic operations"""
    
    @staticmethod
    def generate_random_key(length: int = 32) -> bytes:
        """Generate a random cryptographic key"""
        return secrets.token_bytes(length)
    
    @staticmethod
    def hash_data(data: bytes) -> str:
        """Generate SHA-256 hash of data"""
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def hash_dict(data: Dict[str, Any]) -> str:
        """Generate hash of dictionary data"""
        json_str = json.dumps(data, sort_keys=True)
        return CryptoUtils.hash_data(json_str.encode())
    
    @staticmethod
    def generate_keypair() -> Tuple[str, str]:
        """Generate public/private key pair (placeholder)"""
        # TODO: Implement actual key pair generation
        # This would use libraries like cryptography or PyNaCl
        private_key = secrets.token_hex(32)
        public_key = secrets.token_hex(32)
        return public_key, private_key
    
    @staticmethod
    def sign_data(data: bytes, private_key: str) -> str:
        """Sign data with private key (placeholder)"""
        # TODO: Implement actual digital signature
        # This would use ECDSA or similar signing algorithm
        data_hash = CryptoUtils.hash_data(data)
        signature = hashlib.sha256((data_hash + private_key).encode()).hexdigest()
        return signature
    
    @staticmethod
    def verify_signature(data: bytes, signature: str, public_key: str) -> bool:
        """Verify digital signature (placeholder)"""
        # TODO: Implement actual signature verification
        data_hash = CryptoUtils.hash_data(data)
        expected_signature = hashlib.sha256((data_hash + public_key).encode()).hexdigest()
        return signature == expected_signature
    
    @staticmethod
    def encrypt_data(data: bytes, key: bytes) -> bytes:
        """Encrypt data with symmetric key (placeholder)"""
        # TODO: Implement actual encryption (AES-GCM recommended)
        # This is a placeholder XOR implementation
        encrypted = bytearray()
        key_len = len(key)
        for i, byte in enumerate(data):
            encrypted.append(byte ^ key[i % key_len])
        return bytes(encrypted)
    
    @staticmethod
    def decrypt_data(encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt data with symmetric key (placeholder)"""
        # TODO: Implement actual decryption
        # For XOR, decryption is the same as encryption
        return CryptoUtils.encrypt_data(encrypted_data, key)


class ZKProofUtils:
    """Utility class for zero-knowledge proof operations"""
    
    @staticmethod
    def generate_proof(statement: Dict[str, Any], witness: Dict[str, Any]) -> str:
        """Generate zero-knowledge proof (placeholder)"""
        # TODO: Implement actual ZK proof generation
        # This would use libraries like libsnark, circom, or arkworks
        proof_data = {
            'statement_hash': CryptoUtils.hash_dict(statement),
            'witness_hash': CryptoUtils.hash_dict(witness),
            'timestamp': datetime.now().isoformat()
        }
        return CryptoUtils.hash_dict(proof_data)
    
    @staticmethod
    def verify_proof(statement: Dict[str, Any], proof: str) -> bool:
        """Verify zero-knowledge proof (placeholder)"""
        # TODO: Implement actual ZK proof verification
        # This would verify the cryptographic proof against the statement
        return len(proof) == 64  # Simple placeholder check
    
    @staticmethod
    def generate_payment_proof(amount: float, balance: float, 
                             vehicle_id: str) -> Tuple[str, Dict[str, Any]]:
        """Generate zero-knowledge payment proof"""
        # Statement: "I have sufficient balance to pay the toll"
        statement = {
            'has_sufficient_balance': True,
            'toll_amount': amount,
            'vehicle_authenticated': True
        }
        
        # Witness: Private information (actual balance)
        witness = {
            'actual_balance': balance,
            'vehicle_id': vehicle_id,
            'timestamp': datetime.now().isoformat()
        }
        
        proof = ZKProofUtils.generate_proof(statement, witness)
        return proof, statement


class PrivacyUtils:
    """Utility class for privacy-preserving operations"""
    
    @staticmethod
    def generate_pseudonym(entity_id: str, timestamp: str) -> str:
        """Generate pseudonym for privacy protection"""
        pseudonym_data = f"{entity_id}:{timestamp}:{secrets.token_hex(8)}"
        return CryptoUtils.hash_data(pseudonym_data.encode())[:16]
    
    @staticmethod
    def add_differential_privacy_noise(value: float, epsilon: float = 1.0) -> float:
        """Add Laplace noise for differential privacy"""
        # TODO: Implement proper differential privacy mechanism
        # This is a simplified version
        import random
        sensitivity = 1.0  # Assuming sensitivity of 1
        scale = sensitivity / epsilon
        noise = random.uniform(-scale, scale)
        return value + noise
    
    @staticmethod
    def anonymize_location(lat: float, lon: float, radius: float = 100.0) -> Tuple[float, float]:
        """Anonymize location coordinates within radius"""
        import random
        import math
        
        # Add random offset within radius
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, radius)
        
        # Convert to lat/lon offsets (approximate)
        lat_offset = (distance * math.cos(angle)) / 111000  # ~111km per degree
        lon_offset = (distance * math.sin(angle)) / (111000 * math.cos(math.radians(lat)))
        
        return lat + lat_offset, lon + lon_offset
