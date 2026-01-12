from fastapi import APIRouter, HTTPException
from satori.common.utils.logging import setup_logger
from satori.miner import shared

router = APIRouter()
logger = setup_logger(__name__)

@router.get("/status")
async def get_queue_status():
    if shared.queue_manager is None:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")
    stats = shared.queue_manager.get_queue_stats()
    return stats

