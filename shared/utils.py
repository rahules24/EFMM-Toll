"""
Utility Functions
Common utility functions for EFMM-Toll system
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import hashlib
import secrets


class IDGenerator:
    """Utility class for generating unique identifiers"""
    
    @staticmethod
    def generate_uuid() -> str:
        """Generate UUID4 string"""
        return str(uuid.uuid4())
    
    @staticmethod
    def generate_short_id(length: int = 8) -> str:
        """Generate short alphanumeric ID"""
        return secrets.token_urlsafe(length)[:length]
    
    @staticmethod
    def generate_vehicle_id() -> str:
        """Generate vehicle ID with prefix"""
        return f"vehicle_{IDGenerator.generate_short_id()}"
    
    @staticmethod
    def generate_rsu_id() -> str:
        """Generate RSU ID with prefix"""
        return f"rsu_{IDGenerator.generate_short_id()}"
    
    @staticmethod
    def generate_token_id() -> str:
        """Generate token ID with prefix"""
        return f"token_{IDGenerator.generate_short_id(16)}"
    
    @staticmethod
    def generate_session_id() -> str:
        """Generate session ID"""
        return f"session_{IDGenerator.generate_short_id(12)}"


class TimeUtils:
    """Utility class for time operations"""
    
    @staticmethod
    def get_current_timestamp() -> str:
        """Get current ISO timestamp"""
        return datetime.now().isoformat()
    
    @staticmethod
    def get_expiry_timestamp(duration_seconds: int) -> str:
        """Get expiry timestamp"""
        expiry = datetime.now() + timedelta(seconds=duration_seconds)
        return expiry.isoformat()
    
    @staticmethod
    def is_expired(timestamp_str: str) -> bool:
        """Check if timestamp is expired"""
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return timestamp < datetime.now()
        except ValueError:
            return True
    
    @staticmethod
    def time_until_expiry(timestamp_str: str) -> int:
        """Get seconds until expiry (negative if expired)"""
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            delta = timestamp - datetime.now()
            return int(delta.total_seconds())
        except ValueError:
            return -1


class DataValidator:
    """Utility class for data validation"""
    
    @staticmethod
    def validate_location(location: Dict[str, float]) -> bool:
        """Validate location coordinates"""
        if not isinstance(location, dict):
            return False
        
        if 'lat' not in location or 'lon' not in location:
            return False
        
        lat = location['lat']
        lon = location['lon']
        
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            return False
        
        return -90 <= lat <= 90 and -180 <= lon <= 180
    
    @staticmethod
    def validate_amount(amount: float) -> bool:
        """Validate monetary amount"""
        return isinstance(amount, (int, float)) and amount >= 0
    
    @staticmethod
    def validate_entity_id(entity_id: str) -> bool:
        """Validate entity ID format"""
        return isinstance(entity_id, str) and len(entity_id) >= 3
    
    @staticmethod
    def validate_token(token: str) -> bool:
        """Validate token format"""
        return isinstance(token, str) and len(token) >= 16
    
    @staticmethod
    def validate_message_structure(message: Dict[str, Any]) -> bool:
        """Validate message structure"""
        required_fields = ['message_id', 'message_type', 'sender_id', 'timestamp', 'payload']
        return all(field in message for field in required_fields)


class JsonUtils:
    """Utility class for JSON operations"""
    
    @staticmethod
    def serialize_with_datetime(obj: Any) -> str:
        """Serialize object to JSON with datetime handling"""
        def datetime_handler(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        return json.dumps(obj, default=datetime_handler, sort_keys=True)
    
    @staticmethod
    def safe_json_loads(json_str: str) -> Optional[Dict[str, Any]]:
        """Safely load JSON string"""
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return None
    
    @staticmethod
    def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = JsonUtils.deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result


class AsyncUtils:
    """Utility class for async operations"""
    
    @staticmethod
    async def run_with_timeout(coro, timeout: float):
        """Run coroutine with timeout"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logging.warning(f"Operation timed out after {timeout} seconds")
            return None
    
    @staticmethod
    async def gather_with_error_handling(coroutines: List, return_exceptions: bool = True):
        """Gather coroutines with error handling"""
        try:
            results = await asyncio.gather(*coroutines, return_exceptions=return_exceptions)
            return results
        except Exception as e:
            logging.error(f"Error in gather operation: {e}")
            return [None] * len(coroutines)
    
    @staticmethod
    async def retry_with_backoff(coro, max_retries: int = 3, initial_delay: float = 1.0):
        """Retry coroutine with exponential backoff"""
        for attempt in range(max_retries):
            try:
                return await coro
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                delay = initial_delay * (2 ** attempt)
                logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)


class LocationUtils:
    """Utility class for location operations"""
    
    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula (in meters)"""
        import math
        
        # Earth radius in meters
        R = 6371000
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        # Haversine formula
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    @staticmethod
    def is_within_radius(center_lat: float, center_lon: float, 
                        point_lat: float, point_lon: float, radius: float) -> bool:
        """Check if point is within radius of center"""
        distance = LocationUtils.calculate_distance(center_lat, center_lon, point_lat, point_lon)
        return distance <= radius
    
    @staticmethod
    def generate_geofence(center_lat: float, center_lon: float, 
                         radius: float, num_points: int = 8) -> List[Tuple[float, float]]:
        """Generate geofence polygon points"""
        import math
        
        points = []
        angle_step = 2 * math.pi / num_points
        
        for i in range(num_points):
            angle = i * angle_step
            # Approximate conversion (works for small radii)
            lat_offset = (radius * math.cos(angle)) / 111000  # ~111km per degree latitude
            lon_offset = (radius * math.sin(angle)) / (111000 * math.cos(math.radians(center_lat)))
            
            point_lat = center_lat + lat_offset
            point_lon = center_lon + lon_offset
            points.append((point_lat, point_lon))
        
        return points


class LoggingUtils:
    """Utility class for logging operations"""
    
    @staticmethod
    def setup_logger(name: str, level: str = "INFO", 
                    format_string: Optional[str] = None) -> logging.Logger:
        """Setup logger with specified configuration"""
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add console handler
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(handler)
        
        return logger
    
    @staticmethod
    def log_function_call(func_name: str, args: List[Any], kwargs: Dict[str, Any]):
        """Log function call with parameters"""
        logger = logging.getLogger(__name__)
        logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
    
    @staticmethod
    def log_performance(func_name: str, duration: float):
        """Log function performance"""
        logger = logging.getLogger(__name__)
        logger.info(f"{func_name} completed in {duration:.4f} seconds")
