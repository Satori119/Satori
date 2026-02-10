from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import deque
import asyncio
from satori.common.utils.logging import setup_logger

logger = setup_logger(__name__)

class RateLimiter:
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        
        self._request_history: Dict[str, deque] = {}
        self._lock = asyncio.Lock()
    
    async def check_rate_limit(self, hotkey: str) -> bool:
        now = datetime.now()
        
        async with self._lock:
            if hotkey not in self._request_history:
                self._request_history[hotkey] = deque()
            
            history = self._request_history[hotkey]
            
            cutoff_time = now - timedelta(hours=1)
            while history and history[0] < cutoff_time:
                history.popleft()
            
            minute_cutoff = now - timedelta(minutes=1)
            recent_requests = [ts for ts in history if ts >= minute_cutoff]
            
            if len(recent_requests) >= self.requests_per_minute:
                logger.warning(
                    f"Rate limit exceeded for {hotkey[:16]}...: "
                    f"{len(recent_requests)} requests in last minute"
                )
                return False
            
            if len(history) >= self.requests_per_hour:
                logger.warning(
                    f"Hourly rate limit exceeded for {hotkey[:16]}...: "
                    f"{len(history)} requests in last hour"
                )
                return False
            
            burst_cutoff = now - timedelta(seconds=10)
            burst_requests = [ts for ts in history if ts >= burst_cutoff]
            
            if len(burst_requests) >= self.burst_size:
                logger.warning(
                    f"Burst rate limit exceeded for {hotkey[:16]}...: "
                    f"{len(burst_requests)} requests in last 10 seconds"
                )
                return False
            
            history.append(now)
            
            return True
    
    async def get_rate_limit_info(self, hotkey: str) -> Dict:
        now = datetime.now()
        
        async with self._lock:
            if hotkey not in self._request_history:
                return {
                    "requests_last_minute": 0,
                    "requests_last_hour": 0,
                    "limit_per_minute": self.requests_per_minute,
                    "limit_per_hour": self.requests_per_hour
                }
            
            history = self._request_history[hotkey]
            
            minute_cutoff = now - timedelta(minutes=1)
            hour_cutoff = now - timedelta(hours=1)
            
            requests_last_minute = len([ts for ts in history if ts >= minute_cutoff])
            requests_last_hour = len([ts for ts in history if ts >= hour_cutoff])
            
            return {
                "requests_last_minute": requests_last_minute,
                "requests_last_hour": requests_last_hour,
                "limit_per_minute": self.requests_per_minute,
                "limit_per_hour": self.requests_per_hour,
                "remaining_per_minute": max(0, self.requests_per_minute - requests_last_minute),
                "remaining_per_hour": max(0, self.requests_per_hour - requests_last_hour)
            }
    
    async def reset_rate_limit(self, hotkey: str):
        async with self._lock:
            if hotkey in self._request_history:
                del self._request_history[hotkey]
