from sqlalchemy.orm import Session
from typing import Dict, List, Optional
from satori.task_center.services.score_archive import ScoreArchive
from satori.common.config.yaml_config import YamlConfig
from satori.common.utils.logging import setup_logger
from satori.task_center import shared
import statistics
import httpx
from satori.common.config import settings

logger = setup_logger(__name__)

class ConsensusSync:
    def __init__(self, db: Session):
        self.db = db
        wallet = shared.wallet if hasattr(shared, 'wallet') else None
        wallet_name = shared.wallet_name if hasattr(shared, 'wallet_name') else None
        hotkey_name = shared.hotkey_name if hasattr(shared, 'hotkey_name') else None
        yaml_config = shared.yaml_config
        self.score_archive = ScoreArchive(db, wallet=wallet, wallet_name=wallet_name, hotkey_name=hotkey_name, yaml_config=yaml_config)
    
    def aggregate_scores(self, task_id: str) -> Dict[str, float]:
        all_scores = self.score_archive.get_all_scores_for_task(task_id)
        
        max_validators = settings.CONSENSUS_MAX_VALIDATORS
        min_validators = settings.CONSENSUS_MIN_VALIDATORS
        
        miner_aggregated = {}
        for miner_data in all_scores:
            miner_hotkey = miner_data["miner_hotkey"]
            scores = [s["final_score"] for s in miner_data["scores"]]
            
            # Limit to maximum number of validators
            if len(scores) > max_validators:
                logger.warning(
                    f"Task {task_id}, miner {miner_hotkey[:16]}...: "
                    f"Number of validator scores ({len(scores)}) exceeds maximum ({max_validators}). "
                    f"Limiting to {max_validators} scores."
                )
                # Take only the first max_validators scores
                scores = scores[:max_validators]
            
            if len(scores) < min_validators:
                logger.warning(
                    f"Task {task_id}, miner {miner_hotkey[:16]}...: "
                    f"Number of validator scores ({len(scores)}) is below minimum ({min_validators}). "
                    f"Consensus may not be reliable."
                )
            
            # With max 2 validators, we use all scores (no need to filter outliers)
            avg_score = statistics.mean(scores) if scores else 0.0
            
            miner_aggregated[miner_hotkey] = avg_score
        
        return miner_aggregated
    
    def sync_consensus_data(self, task_id: str) -> Dict[str, Dict]:
        aggregated_scores = self.aggregate_scores(task_id)

        return {
            "task_id": task_id,
            "miner_scores": aggregated_scores,
            "consensus_status": "completed"
        }

    async def notify_validators(self, task_id: str, consensus_data: Dict):
        from satori.common.models.validator import Validator
        validators = self.db.query(Validator).filter(Validator.is_active == True).all()
        
        for validator in validators:
            try:
                validator_url = f"http://{validator.hotkey}:8000"
                
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(
                        f"{validator_url}/v1/consensus/sync",
                        json={
                            "task_id": task_id,
                            "consensus_data": consensus_data
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to notify validator {validator.hotkey}: {e}")

