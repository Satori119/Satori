from fastapi import APIRouter, Depends, HTTPException, Security, Request
from sqlalchemy.orm import Session
from satori.common.database import get_db
from satori.common.auth.api_key import verify_api_key
from satori.task_center.services.score_archive import ScoreArchive
from satori.task_center.services.consensus_sync import ConsensusSync
from satori.task_center.schemas.score import ScoreSubmit, ScoreQueryResponse
from satori.task_center import shared
from satori.common.utils.logging import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

@router.post("/submit")
async def submit_score(
    score_data: ScoreSubmit,
    db: Session = Depends(get_db)
):
    from satori.task_center.services.task_lifecycle_manager import TaskLifecycleManager
    lifecycle_manager = TaskLifecycleManager(db)
    can_score, reason = lifecycle_manager.can_validator_score(score_data.task_id)

    if not can_score:
        logger.warning(f"Score submission rejected for task {score_data.task_id}: {reason}")
        raise HTTPException(status_code=400, detail=f"Scoring not allowed: {reason}")

    archive = ScoreArchive(db, wallet_manager=shared.wallet_manager, yaml_config=shared.yaml_config)
    archive.submit_score(score_data)
    return {"status": "success", "message": "Score submitted successfully"}


@router.get("/query/{miner_hotkey}", response_model=ScoreQueryResponse)
async def query_score(
    miner_hotkey: str,
    task_id: str = None,
    request: Request = None,
    db: Session = Depends(get_db)
):
    archive = ScoreArchive(db, wallet_manager=shared.wallet_manager, yaml_config=shared.yaml_config)
    scores = archive.get_miner_scores(miner_hotkey, task_id)

    if not scores:
        raise HTTPException(status_code=404, detail="No scores found")

    return ScoreQueryResponse(
        miner_hotkey=miner_hotkey,
        scores=scores,
        ema_score=archive.calculate_ema_score(miner_hotkey, task_id)
    )


@router.get("/all")
async def get_all_scores(
    task_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    archive = ScoreArchive(db, wallet_manager=shared.wallet_manager, yaml_config=shared.yaml_config)
    consensus_sync = ConsensusSync(db)

    scores = archive.get_all_scores_for_task(task_id)
    consensus_data = consensus_sync.sync_consensus_data(task_id)

    return {
        "task_id": task_id,
        "scores": scores,
        "consensus_data": consensus_data
    }


@router.get("/query")
async def query_all_scores(
    task_id: str = None,
    request: Request = None,
    db: Session = Depends(get_db)
):
    archive = ScoreArchive(db, wallet_manager=shared.wallet_manager, yaml_config=shared.yaml_config)
    consensus_sync = ConsensusSync(db)

    if task_id:
        scores = archive.get_all_scores_for_task(task_id)
        consensus_data = consensus_sync.sync_consensus_data(task_id)

        return {
            "task_id": task_id,
            "miner_scores": {
                item["miner_hotkey"]: {
                    "consensus_score": item["consensus_score"],
                    "ema_score": item["ema_score"],
                    "validator_count": item["validator_count"]
                }
                for item in scores
            },
            "consensus_data": consensus_data
        }
    else:
        from satori.common.models.score import Score
        from satori.common.bittensor.client import BittensorClient

        wallet_name = shared.wallet_manager.wallet_name if shared.wallet_manager else "task_center"
        hotkey_name = shared.wallet_manager.hotkey_name if shared.wallet_manager else "default"
        bittensor_client = BittensorClient(wallet_name, hotkey_name, yaml_config=shared.yaml_config)

        registered_miners = bittensor_client.get_all_miners()
        registered_hotkeys = {m.get("hotkey") for m in registered_miners if m.get("hotkey")}

        logger.info(f"Found {len(registered_hotkeys)} registered miners on subnet")

        miners = db.query(Score.miner_hotkey).distinct().all()

        all_miner_scores = {}
        excluded_count = 0
        for (miner_hotkey,) in miners:
            if miner_hotkey not in registered_hotkeys:
                excluded_count += 1
                continue

            ema_score = archive.calculate_ema_score(miner_hotkey)
            history = archive.get_miner_history_scores(miner_hotkey, limit=100)

            all_miner_scores[miner_hotkey] = {
                "ema_score": ema_score,
                "history_count": len(history),
                "latest_score": history[0]["final_score"] if history else 0.0
            }

        logger.info(f"Returning scores for {len(all_miner_scores)} registered miners, excluded {excluded_count} deregistered miners")

        return {
            "all_miners": all_miner_scores,
            "total_miners": len(all_miner_scores)
        }

