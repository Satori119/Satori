from typing import Dict, List, Optional
from satori.common.services.scoring import ScoringService
from satori.common.config import settings
from satori.common.utils.logging import setup_logger

logger = setup_logger(__name__)


class RewardService:

    def __init__(self):
        self.scoring_service = ScoringService()

    def calculate_normalized_weights(
        self,
        miner_scores: Dict[str, float],
        miner_weights: Optional[Dict[str, float]] = None,
        task_type: str = "text_lora_creation",
        has_text_tasks: bool = False,
        has_image_tasks: bool = False
    ) -> Dict[str, float]:
        if miner_weights:
            weights = miner_weights.copy()
        else:
            weights = {}
            for miner_hotkey, score in miner_scores.items():
                if score < 3.5:
                    weights[miner_hotkey] = 0.0
                else:
                    weights[miner_hotkey] = self.scoring_service.calculate_quality_score(score)

        if has_text_tasks and has_image_tasks:
            if task_type in ["text_lora_creation", "text_lora"]:
                pool_ratio = settings.TEXT_POOL_RATIO
            elif task_type in ["image_lora_creation", "image_lora"]:
                pool_ratio = settings.IMAGE_POOL_RATIO
            else:
                pool_ratio = 0.50

            weights = {k: v * pool_ratio for k, v in weights.items()}

        total_weight = sum(weights.values())

        if total_weight == 0:
            logger.warning("Total weight is 0, no weights to distribute")
            return {hotkey: 0.0 for hotkey in miner_scores.keys()}

        normalized_weights = {}
        for miner_hotkey, weight in weights.items():
            if weight == 0.0:
                normalized_weights[miner_hotkey] = 0.0
            else:
                normalized_weights[miner_hotkey] = weight / total_weight

        pool_ratio_str = "100%"
        if has_text_tasks and has_image_tasks:
            if task_type in ["text_lora_creation", "text_lora"]:
                pool_ratio_str = f"{settings.TEXT_POOL_RATIO * 100:.0f}%"
            else:
                pool_ratio_str = f"{settings.IMAGE_POOL_RATIO * 100:.0f}%"

        logger.info(
            f"Weight calculation: pool_ratio={pool_ratio_str}, "
            f"total_weight={total_weight:.4f}, "
            f"miners={len([w for w in normalized_weights.values() if w > 0])}, "
            f"task_type={task_type}"
        )

        return normalized_weights

    def calculate_rewards(
        self,
        miner_scores: Dict[str, float],
        miner_weights: Optional[Dict[str, float]] = None,
        task_type: str = "text_lora_creation",
        total_emission: float = 1000.0,
        has_text_tasks: bool = False,
        has_image_tasks: bool = False
    ) -> Dict[str, float]:
        return self.calculate_normalized_weights(
            miner_scores=miner_scores,
            miner_weights=miner_weights,
            task_type=task_type,
            has_text_tasks=has_text_tasks,
            has_image_tasks=has_image_tasks
        )
