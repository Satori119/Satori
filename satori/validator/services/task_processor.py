import asyncio
from typing import Dict, Any, Optional
from satori.validator.services.audit_validator import AuditValidator
from satori.validator.services.score_calculator import ScoreCalculator
from satori.validator.schemas.audit import AuditTaskRequest
from satori.common.utils.logging import setup_logger
from satori.common.config.yaml_config import YamlConfig
import httpx
from satori.common.config import settings
import bittensor as bt

logger = setup_logger(__name__)

class TaskProcessor:

    def __init__(self, wallet: bt.wallet, wallet_name: str, hotkey_name: str, yaml_config: Optional[YamlConfig] = None):
        self.wallet = wallet
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.yaml_config = yaml_config
        self.audit_validator = AuditValidator()
        self.score_calculator = ScoreCalculator()
        self.is_running = False
        self.process_interval = 60
        self._process_task = None

        if yaml_config:
            self.task_center_url = yaml_config.get_task_center_url() or settings.TASK_CENTER_URL
            self.api_key = yaml_config.get_task_center_api_key()
        else:
            self.task_center_url = settings.TASK_CENTER_URL
            self.api_key = getattr(settings, 'API_KEY', None)

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with API Key if configured"""
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    async def start(self):
        if self.is_running:
            logger.warning("Task processor is already running")
            return

        self.is_running = True
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info("Task processor started")

    async def stop(self):
        if not self.is_running:
            return

        self.is_running = False

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        logger.info("Task processor stopped")

    async def _process_loop(self):
        while self.is_running:
            try:
                await self._process_pending_tasks()
                await asyncio.sleep(self.process_interval)
            except asyncio.CancelledError:
                logger.info("Process loop cancelled")
                break
            except Exception as e:
                logger.error(f"Task processing loop error: {e}", exc_info=True)
                await asyncio.sleep(self.process_interval)

    async def _process_pending_tasks(self):
        try:
            validator_key = self.wallet.hotkey.ss58_address
            task_center_url = self.task_center_url

            logger.debug(f"Fetching pending audit tasks for validator {validator_key[:20]}...")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{task_center_url}/v1/validators/pending",
                    params={"validator_key": validator_key},
                    headers=self._get_headers()
                )

                if response.status_code == 200:
                    data = response.json()
                    tasks = data.get("tasks", data) if isinstance(data, dict) else data

                    if tasks:
                        logger.info(f"Found {len(tasks)} pending audit tasks")

                    for task in tasks:
                        try:
                            await self._process_audit_task(task)
                        except Exception as e:
                            audit_task_id = task.get("audit_task_id", task.get("id", "unknown"))
                            logger.error(f"Failed to process audit task {audit_task_id}: {e}", exc_info=True)
                            continue
                elif response.status_code == 404:
                    logger.debug("No pending audit tasks found")
                else:
                    logger.warning(f"Unexpected response from task center: {response.status_code}")

        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching pending tasks: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error processing pending tasks: {e}", exc_info=True)

    async def _process_audit_task(self, task: Dict[str, Any]):
        audit_task_id = task.get("audit_task_id", task.get("id", "unknown"))
        miner_hotkey = task.get("miner_hotkey", "")
        task_id = task.get("original_task_id", task.get("task_id", ""))
        audit_type = task.get("audit_type", "lora")
        lora_url = task.get("lora_url", "")
        dataset_url = task.get("dataset_url", "")
        task_info = task.get("task_info", {})

        logger.info(f"Processing audit task {audit_task_id} (type={audit_type}) for miner {miner_hotkey[:20]}...")

        try:
            if audit_type == "dataset":
                await self._process_dataset_audit_task(
                    audit_task_id=audit_task_id,
                    task_id=task_id,
                    miner_hotkey=miner_hotkey,
                    dataset_url=dataset_url,
                    task_info=task_info
                )
            else:
                await self._process_lora_audit_task(
                    audit_task_id=audit_task_id,
                    task_id=task_id,
                    miner_hotkey=miner_hotkey,
                    lora_url=lora_url,
                    task_info=task_info
                )

        except Exception as e:
            logger.error(f"Failed to process audit task {audit_task_id}: {e}", exc_info=True)
            await self._update_audit_task_status(audit_task_id, "failed", {"error": str(e)})
            raise

    async def _process_lora_audit_task(
        self,
        audit_task_id: str,
        task_id: str,
        miner_hotkey: str,
        lora_url: str,
        task_info: Dict[str, Any]
    ):
        audit_request = AuditTaskRequest(
            audit_task_id=audit_task_id,
            miner_hotkey=miner_hotkey,
            lora_url=lora_url,
            task_info=task_info
        )

        result = await self.audit_validator.process_audit_task(audit_request)

        score = result.get("final_score", 0.0)

        logger.info(f"LoRA audit task {audit_task_id} completed: "
                   f"cosine_similarity={result.get('cosine_similarity', 0):.4f}, "
                   f"quality_score={result.get('quality_score', 0):.2f}, "
                   f"final_score={score:.2f}")

        await self._submit_score(
            task_id=task_id,
            miner_hotkey=miner_hotkey,
            audit_task_id=audit_task_id,
            result=result,
            score=score
        )

        await self._update_audit_task_status(audit_task_id, "completed", result)

    async def _process_dataset_audit_task(
        self,
        audit_task_id: str,
        task_id: str,
        miner_hotkey: str,
        dataset_url: str,
        task_info: Dict[str, Any]
    ):
        result = await self.audit_validator.process_audit_task({
            "audit_task_id": audit_task_id,
            "miner_hotkey": miner_hotkey,
            "dataset_url": dataset_url,
            "audit_type": "dataset",
            "task_info": task_info
        })

        is_valid = result.get("is_valid", False)

        logger.info(f"Dataset audit task {audit_task_id} completed: is_valid={is_valid}")

        await self._submit_dataset_validation(
            audit_task_id=audit_task_id,
            task_id=task_id,
            miner_hotkey=miner_hotkey,
            is_approved=is_valid,
            validation_result=result,
            rejection_reason=result.get("rejection_reason")
        )

        await self._update_audit_task_status(audit_task_id, "completed", result)

    async def _submit_score(
        self,
        task_id: str,
        miner_hotkey: str,
        audit_task_id: str,
        result: Dict[str, Any],
        score: float
    ):
        validator_hotkey = self.wallet.hotkey.ss58_address
        task_center_url = self.task_center_url

        score_data = {
            "task_id": task_id,
            "miner_hotkey": miner_hotkey,
            "validator_hotkey": validator_hotkey,
            "audit_task_id": audit_task_id,
            "cosine_similarity": result.get("cosine_similarity", 0.0),
            "quality_score": result.get("quality_score", 0.0),
            "final_score": score,
            "content_safety_score": result.get("content_safety_score", 0.0),
            "rejected": result.get("rejected", False),
            "rejection_reason": result.get("reason", None)
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{task_center_url}/v1/scores/submit",
                    json=score_data,
                    headers=self._get_headers()
                )

                if response.status_code == 200:
                    logger.info(f"Score submitted successfully for audit task {audit_task_id}")
                else:
                    logger.warning(f"Score submission returned status {response.status_code}: {response.text[:200]}")

        except httpx.HTTPError as e:
            logger.error(f"HTTP error submitting score: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error submitting score: {e}", exc_info=True)

    async def _submit_dataset_validation(
        self,
        audit_task_id: str,
        task_id: str,
        miner_hotkey: str,
        is_approved: bool,
        validation_result: Dict[str, Any],
        rejection_reason: Optional[str] = None
    ):
        validator_hotkey = self.wallet.hotkey.ss58_address
        task_center_url = self.task_center_url

        validation_data = {
            "audit_task_id": audit_task_id,
            "validator_hotkey": validator_hotkey,
            "is_approved": is_approved,
            "validation_result": validation_result,
            "rejection_reason": rejection_reason
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{task_center_url}/v1/validators/dataset/validation",
                    json=validation_data,
                    headers=self._get_headers()
                )

                if response.status_code == 200:
                    logger.info(f"Dataset validation submitted: audit_task={audit_task_id}, approved={is_approved}")
                else:
                    logger.warning(f"Dataset validation submission returned {response.status_code}: {response.text[:200]}")

        except httpx.HTTPError as e:
            logger.error(f"HTTP error submitting dataset validation: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error submitting dataset validation: {e}", exc_info=True)

    async def _update_audit_task_status(
        self,
        audit_task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None
    ):
        task_center_url = self.task_center_url

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{task_center_url}/v1/audit/update_status",
                    json={
                        "audit_task_id": audit_task_id,
                        "status": status,
                        "result": result
                    },
                    headers=self._get_headers()
                )

                if response.status_code == 200:
                    logger.debug(f"Audit task {audit_task_id} status updated to {status}")
                elif response.status_code == 404:
                    logger.warning(f"Audit task {audit_task_id} not found for status update")
                else:
                    logger.warning(f"Status update returned {response.status_code}")

        except httpx.HTTPError as e:
            logger.error(f"HTTP error updating audit task status: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error updating audit task status: {e}", exc_info=True)
