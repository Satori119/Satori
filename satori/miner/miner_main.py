import os
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI

from satori.common.utils.logging import set_module_prefix, set_global_log_level, setup_logger, reinitialize_all_loggers
from satori.common.config import load_yaml_config

set_module_prefix("MINER")

_config_path = os.getenv("MINER_CONFIG")
if not _config_path:
    import satori.miner
    _miner_dir = Path(satori.miner.__file__).parent
    _config_path = str(_miner_dir / "config.yml")
_yaml_config = load_yaml_config(_config_path)

if _yaml_config:
    _log_level = _yaml_config.get('logging.level', 'INFO')
    set_global_log_level(_log_level)

from satori.miner.api import router
from satori.miner.services.queue_manager import QueueManager
from satori.miner.services.gpu_manager import GPUManager
from satori.miner.services.bittensor_sync import BittensorSyncService
from satori.miner.services.task_monitor_service import TaskMonitorService
from satori.common.services.auto_update import AutoUpdateService
import bittensor as bt
from satori.common.config import settings
from satori.common.utils.logging import setup_logger as setup_logger_base
from satori.common.utils.thread_pool import get_thread_pool
from satori.miner import shared

config_path = _config_path
yaml_config = _yaml_config

log_file = yaml_config.get('logging.file') if yaml_config else None
logger = setup_logger(__name__, log_file=log_file)

if yaml_config:
    wallet_name = yaml_config.get_wallet_name()
    hotkey_name = yaml_config.get_hotkey_name()
    task_center_url = yaml_config.get_task_center_url()
    gpu_count = yaml_config.get_gpu_count()
    auto_update_config = yaml_config.get_auto_update_config()
else:
    wallet_name = "miner"
    hotkey_name = "default"
    task_center_url = settings.TASK_CENTER_URL
    gpu_count = 1
    auto_update_config = {}

queue_manager = QueueManager(
    max_queue_size=yaml_config.get('miner.max_queue_size', 100) if yaml_config else 100,
    max_training_jobs=yaml_config.get('miner.max_training_jobs', 2) if yaml_config else 2,
    max_test_jobs=yaml_config.get('miner.max_test_jobs', 4) if yaml_config else 4
)

gpu_manager = GPUManager(gpu_count)
queue_manager.gpu_manager = gpu_manager

if yaml_config:
    from satori.miner.services.training_service import TrainingService
    training_service = TrainingService(yaml_config)
    queue_manager.training_service = training_service

shared.queue_manager = queue_manager
wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
shared.wallet = wallet
shared.wallet_name = wallet_name
shared.hotkey_name = hotkey_name
shared.yaml_config = yaml_config
bittensor_sync = BittensorSyncService(wallet, wallet_name, hotkey_name, yaml_config=yaml_config)

if yaml_config:
    github_repo = yaml_config.get_github_repo()
    auto_update_enabled = yaml_config.get_auto_update_enabled()
    check_interval = yaml_config.get_auto_update_interval()
else:
    github_repo = settings.GITHUB_REPO
    auto_update_enabled = settings.AUTO_UPDATE_ENABLED
    check_interval = 300

auto_update = AutoUpdateService(
    github_repo=github_repo or "satori/miner",
    branch=auto_update_config.get('branch', 'main'),
    check_interval=check_interval,
    restart_delay=auto_update_config.get('restart_delay', 10)
)

task_monitor = TaskMonitorService(wallet, wallet_name, hotkey_name, yaml_config)

@asynccontextmanager
async def lifespan(app: FastAPI):
    log_file = yaml_config.get('logging.file') if yaml_config else None
    reinitialize_all_loggers(log_file)

    logger.info("Miner service starting up")
    logger.info(f"Miner hotkey: {wallet.hotkey.ss58_address}")
    try:
        subtensor = bt.subtensor(network=yaml_config.get_chain_endpoint() if yaml_config and yaml_config.get_chain_endpoint() else "finney")
        balance = float(subtensor.get_balance(wallet.coldkeypub.ss58_address))
        logger.info(f"Miner balance: {balance} TAO")
    except Exception as e:
        logger.warning(f"Failed to get balance: {e}")
        logger.info("Miner balance: unavailable")
    logger.info(f"Config loaded from: {config_path if yaml_config else 'default'}")

    # Register axon on Bittensor chain
    if yaml_config and yaml_config.get_axon_enabled():
        axon_ip = yaml_config.get_axon_ip()
        axon_port = yaml_config.get_axon_port()
        axon_external_ip = yaml_config.get_axon_external_ip()
        netuid = yaml_config.get_netuid()
        chain_endpoint = yaml_config.get_chain_endpoint()

        logger.info(f"Registering axon on chain: ip={axon_ip}, port={axon_port}, netuid={netuid}")
        try:
            if chain_endpoint:
                subtensor = bt.subtensor(network=chain_endpoint)
            else:
                subtensor = bt.subtensor(network="finney")
            
            axon = bt.axon(
                wallet=wallet,
                ip=axon_ip,
                port=axon_port,
                external_ip=axon_external_ip
            )
            
            success = subtensor.serve_axon(netuid=netuid, axon=axon)
            if success:
                logger.info("Axon registration successful")
            else:
                logger.warning("Axon registration failed, continuing without chain registration")
        except Exception as e:
            logger.error(f"Failed to register axon: {e}", exc_info=True)
    elif yaml_config:
        logger.info("Axon registration disabled in config")

    logger.info("Starting queue manager scheduler...")
    try:
        await queue_manager.start_scheduler()
        logger.info("Queue manager scheduler started successfully")
    except Exception as e:
        logger.error(f"Failed to start queue manager: {e}", exc_info=True)

    logger.info("Starting bittensor sync service...")
    try:
        await bittensor_sync.start_sync()
        logger.info("Bittensor sync service started successfully")
    except Exception as e:
        logger.error(f"Failed to start bittensor sync: {e}", exc_info=True)

    if auto_update_enabled:
        logger.info("Starting auto-update service...")
        try:
            await auto_update.start()
            logger.info("Auto-update service started successfully")
        except Exception as e:
            logger.error(f"Failed to start auto-update: {e}", exc_info=True)

    logger.info("Starting task monitor service...")
    try:
        await task_monitor.start()
        logger.info("Task monitor service started successfully")
    except Exception as e:
        logger.error(f"Failed to start task monitor: {e}", exc_info=True)

    logger.info("All startup tasks completed, yielding to FastAPI...")

    try:
        yield
    finally:
        logger.info("Miner service shutting down")
    
    try:
        await queue_manager.stop_scheduler()
    except Exception as e:
        logger.error(f"Error stopping queue manager: {e}", exc_info=True)
    
    try:
        await bittensor_sync.stop_sync()
    except Exception as e:
        logger.error(f"Error stopping bittensor sync: {e}", exc_info=True)
    
    try:
        await auto_update.stop()
    except Exception as e:
        logger.error(f"Error stopping auto-update: {e}", exc_info=True)

    try:
        await task_monitor.stop()
    except Exception as e:
        logger.error(f"Error stopping task monitor: {e}", exc_info=True)

    try:
        thread_pool = get_thread_pool()
        thread_pool.shutdown(wait=True)
    except Exception as e:
        logger.error(f"Error shutting down thread pool: {e}", exc_info=True)

app = FastAPI(title="SATORI Miner", version="1.0.0", lifespan=lifespan)

from satori.common.middleware import add_request_logging
add_request_logging(app, exclude_paths=["/health", "/docs", "/openapi.json", "/redoc"])

app.include_router(router, prefix="/v1")

if __name__ == "__main__":
    import uvicorn

    if yaml_config:
        default_host = yaml_config.get_axon_ip()
        default_port = yaml_config.get_axon_port()
    else:
        default_host = "0.0.0.0"
        default_port = 8001

    host = os.getenv("MINER_HOST", default_host)
    port = int(os.getenv("MINER_PORT", str(default_port)))

    uvicorn_log_level = "debug" if _log_level and _log_level.upper() == "DEBUG" else "info"

    logger.info(f"Starting Miner service on {host}:{port}")
    logger.info("Using asyncio event loop (required for bittensor compatibility)")

    uvicorn.run(
        app,
        host=host,
        port=port,
        loop="asyncio",
        log_level=uvicorn_log_level,
        log_config=None
    )
