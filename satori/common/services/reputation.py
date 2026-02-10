from typing import Dict, Optional
from datetime import datetime, timezone
from satori.common.utils.logging import setup_logger

logger = setup_logger(__name__)

class ReputationService:

    def __init__(self, alpha: float = 0.9):

        self.alpha = alpha
    
    def calculate_reputation(
        self,
        previous_reputation: float,
        current_score: float
    ) -> float:

        if current_score < 0 or current_score > 10:
            logger.warning(f"Invalid current_score: {current_score}, clamping to [0, 10]")
            current_score = max(0.0, min(10.0, current_score))
        
        reputation = self.alpha * previous_reputation + (1 - self.alpha) * current_score
        
        return max(0.0, min(10.0, reputation))
    
    def calculate_cooldown_hours(
        self,
        consecutive_failures: int
    ) -> int:

        if consecutive_failures <= 1:
            return 0
        elif consecutive_failures == 2:
            return 0
        elif consecutive_failures == 3:
            return 24
        else:
            return 48
    
    def apply_reputation_penalty(
        self,
        current_reputation: float,
        consecutive_failures: int
    ) -> float:

        if consecutive_failures >= 2:
            return current_reputation * 0.9
        return current_reputation
    
    def get_priority_level(self, reputation: float) -> str:

        if reputation > 7.0:
            return "high"
        elif reputation >= 4.0:
            return "normal"
        else:
            return "low"
    
    def should_allow_submission(
        self,
        reputation: float,
        consecutive_failures: int,
        last_failure_time: Optional[datetime] = None
    ) -> bool:

        cooldown_hours = self.calculate_cooldown_hours(consecutive_failures)
        
        if cooldown_hours == 0:
            return True
        
        if last_failure_time is None:
            return True
        
        now = datetime.now(timezone.utc)
        time_since_failure = (now - last_failure_time).total_seconds() / 3600
        
        return time_since_failure >= cooldown_hours

