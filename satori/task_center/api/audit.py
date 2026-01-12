from fastapi import APIRouter, Depends, Security, Request, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Dict, Any
from satori.common.database import get_db
from satori.common.auth.api_key import verify_api_key
from satori.task_center.services.audit_task_creator import AuditTaskCreator
from satori.common.utils.logging import setup_logger

router = APIRouter()
logger = setup_logger(__name__)


class AuditStatusUpdate(BaseModel):
    audit_task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None


@router.post("/create")
async def create_audit_task(
    task_id: str,
    miner_hotkey: str,
    lora_url: str,
    request: Request,
    db: Session = Depends(get_db)
):
    creator = AuditTaskCreator(db)
    audit_task = creator.create_audit_task(task_id, miner_hotkey, lora_url)

    creator.auto_assign_audit_tasks(task_id)
    logger.info(f"Audit task created and auto-assigned: {audit_task.audit_task_id}")

    return {"audit_task_id": audit_task.audit_task_id, "status": "created"}


@router.post("/update_status")
async def update_audit_task_status(
    update_data: AuditStatusUpdate,
    db: Session = Depends(get_db)
):

    creator = AuditTaskCreator(db)
    creator.update_audit_task_status(
        audit_task_id=update_data.audit_task_id,
        status=update_data.status,
        result=update_data.result
    )

    return {
        "status": "success",
        "audit_task_id": update_data.audit_task_id,
        "new_status": update_data.status
    }


@router.get("/status/{task_id}")
async def get_audit_task_status(
    task_id: str,
    db: Session = Depends(get_db)
):
    creator = AuditTaskCreator(db)
    status = creator.get_audit_task_status(task_id)
    return status


@router.get("/pending")
async def get_pending_audit_tasks(
    validator_key: str,
    db: Session = Depends(get_db)
):

    creator = AuditTaskCreator(db)
    tasks = creator.get_pending_tasks_for_validator(validator_key)

    return {
        "tasks": [
            {
                "audit_task_id": t.audit_task_id,
                "original_task_id": t.original_task_id,
                "miner_hotkey": t.miner_hotkey,
                "lora_url": t.lora_url,
                "task_info": t.task_info,
                "created_at": t.created_at.isoformat() if t.created_at else None
            }
            for t in tasks
        ]
    }

