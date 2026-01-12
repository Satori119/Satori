from datetime import datetime, timezone
from typing import Optional


def get_current_time() -> datetime:
    return datetime.now(timezone.utc)


def calculate_time_coefficient(
    submit_time: datetime,
    execution_start: datetime,
    execution_end: datetime
) -> float:
    hours_since_start = (submit_time - execution_start).total_seconds() / 3600
    
    if hours_since_start < 6:
        return 0.8
    
    best_window_start = 24.0
    best_window_end = 48.0
    
    if best_window_start <= hours_since_start <= best_window_end:
        return 1.0
    
    if hours_since_start > best_window_end:
        delay_hours = hours_since_start - best_window_end
        decay_rate = 0.005
        coefficient = 1.0 - (delay_hours * decay_rate)
        return max(0.5, coefficient)
    
    return 1.0

