from sqlalchemy.orm import Session
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from satori.common.models.reward_distribution import RewardDistribution
from satori.common.models.task import Task, TaskStatus
from satori.common.models.audit_task import AuditTask
from satori.common.models.score import Score
from satori.common.models.workflow_type import WorkflowType
from satori.common.services.reward import RewardService
from satori.common.services.scoring import ScoringService
import bittensor as bt
from satori.common.config.yaml_config import YamlConfig
from satori.task_center.services.task_lifecycle_manager import TaskLifecycleManager
from satori.task_center.services.score_archive import ScoreArchive
from satori.common.utils.logging import setup_logger
import uuid

logger = setup_logger(__name__)


class ContinuousRewardDistributor:

    def __init__(self, db: Session, wallet: Optional[bt.wallet] = None, wallet_name: Optional[str] = None, hotkey_name: Optional[str] = None, yaml_config: Optional[YamlConfig] = None):
        self.db = db
        self.reward_service = RewardService()
        self.scoring_service = ScoringService()
        self.lifecycle_manager = TaskLifecycleManager(db)
        self.score_archive = ScoreArchive(db, wallet=wallet, wallet_name=wallet_name, hotkey_name=hotkey_name, yaml_config=yaml_config)

    def _get_reward_phase_task_types(self) -> Tuple[bool, bool]:
        reward_tasks = self.db.query(Task).filter(
            Task.status == TaskStatus.REWARD
        ).all()

        has_text_tasks = False
        has_image_tasks = False

        for task in reward_tasks:
            task_type = task.workflow_type.value if hasattr(task.workflow_type, 'value') else str(task.workflow_type)
            task_type_lower = task_type.lower()

            if "text" in task_type_lower:
                has_text_tasks = True
            elif "image" in task_type_lower:
                has_image_tasks = True

            if has_text_tasks and has_image_tasks:
                break

        return has_text_tasks, has_image_tasks

    def distribute_rewards_for_completed_audit(
        self,
        audit_task_id: str,
        task_id: str
    ) -> Dict[str, float]:
        can_distribute, reason = self.lifecycle_manager.can_distribute_rewards(task_id)
        if not can_distribute:
            logger.warning(f"Reward distribution not allowed for task {task_id}: {reason}")
            return {}

        audit_task = self.db.query(AuditTask).filter(
            AuditTask.audit_task_id == audit_task_id
        ).first()

        if not audit_task:
            logger.warning(f"Audit task {audit_task_id} not found")
            return {}

        task = self.db.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            logger.warning(f"Task {task_id} not found")
            return {}

        miner_scores = self.score_archive.get_all_scores_for_task(task_id)

        eligible_miners = {}
        miner_submit_times = {}

        for miner_data in miner_scores:
            miner_hotkey = miner_data["miner_hotkey"]
            avg_score = miner_data.get("average_score", 0.0)

            if avg_score < 3.5:
                continue

            eligible_miners[miner_hotkey] = avg_score

            from satori.common.models.miner_submission import MinerSubmission
            submission = self.db.query(MinerSubmission).filter(
                MinerSubmission.task_id == task_id,
                MinerSubmission.miner_hotkey == miner_hotkey
            ).order_by(MinerSubmission.created_at.desc()).first()

            if submission:
                miner_submit_times[miner_hotkey] = submission.created_at

        if not eligible_miners:
            logger.info(f"No eligible miners for task {task_id} (all below baseline)")
            return {}

        miner_weights = {}
        miner_time_coefficients = {}
        for miner_hotkey, score in eligible_miners.items():
            quality_score = self.scoring_service.calculate_quality_score(score)

            time_coefficient = 1.0
            if miner_submit_times.get(miner_hotkey) and task.execution_start and task.review_start:
                submit_time = miner_submit_times[miner_hotkey]
                time_coefficient = self.scoring_service.calculate_time_coefficient(
                    submit_time, task.execution_start, task.review_start
                )

            miner_time_coefficients[miner_hotkey] = time_coefficient

            constraint_coefficient = 1.0

            final_weight = self.scoring_service.calculate_final_weight(
                quality_score,
                time_coefficient,
                constraint_coefficient
            )

            miner_weights[miner_hotkey] = final_weight

        task_type = task.workflow_type.value if hasattr(task.workflow_type, 'value') else str(task.workflow_type)

        has_text_tasks, has_image_tasks = self._get_reward_phase_task_types()

        normalized_weights = self.reward_service.calculate_normalized_weights(
            miner_scores=eligible_miners,
            miner_weights=miner_weights,
            task_type=task_type,
            has_text_tasks=has_text_tasks,
            has_image_tasks=has_image_tasks
        )

        distribution_round = f"audit_{audit_task_id}"
        for miner_hotkey, weight_ratio in normalized_weights.items():
            if weight_ratio > 0:
                distribution_id = str(uuid.uuid4())
                distribution = RewardDistribution(
                    id=distribution_id,
                    task_id=task_id,
                    miner_hotkey=miner_hotkey,
                    reward_amount=weight_ratio,
                    weight=miner_weights.get(miner_hotkey, 0.0),
                    score=eligible_miners.get(miner_hotkey, 0.0),
                    distribution_data={
                        "audit_task_id": audit_task_id,
                        "time_coefficient": miner_time_coefficients.get(miner_hotkey, 1.0),
                        "quality_score": self.scoring_service.calculate_quality_score(
                            eligible_miners.get(miner_hotkey, 0.0)
                        )
                    },
                    distribution_round=distribution_round
                )
                self.db.add(distribution)

        self.db.commit()

        logger.info(
            f"Weight distribution recorded for task {task_id} (audit {audit_task_id}): "
            f"{len([w for w in normalized_weights.values() if w > 0])} miners with positive weight"
        )

        return normalized_weights

    def get_total_rewards_for_miner(
        self,
        task_id: str,
        miner_hotkey: str
    ) -> float:
        distributions = self.db.query(RewardDistribution).filter(
            RewardDistribution.task_id == task_id,
            RewardDistribution.miner_hotkey == miner_hotkey
        ).all()

        return sum(d.reward_amount for d in distributions)

    def get_all_rewards_for_task(self, task_id: str) -> Dict[str, float]:
        distributions = self.db.query(RewardDistribution).filter(
            RewardDistribution.task_id == task_id
        ).all()

        miner_rewards = {}
        for dist in distributions:
            if dist.miner_hotkey not in miner_rewards:
                miner_rewards[dist.miner_hotkey] = 0.0
            miner_rewards[dist.miner_hotkey] += dist.reward_amount

        return miner_rewards
