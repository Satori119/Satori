import asyncio
import os
import sys
from pathlib import Path

if 'uvloop' in sys.modules:
    pass
else:
    os.environ.setdefault("UVLOOP_DISABLED", "1")

try:
    import pandas as pd
    if not hasattr(pd.io.json, 'json_normalize'):
        try:
            from pandas import json_normalize
            pd.io.json.json_normalize = json_normalize
        except ImportError:
            pass
except (ImportError, AttributeError):
    pass

from satori.common.utils.logging import set_module_prefix, set_global_log_level, setup_logger, reinitialize_all_loggers
from satori.common.config import load_yaml_config

set_module_prefix("TASK_CENTER")

_config_path = os.getenv("TASK_CENTER_CONFIG")
if not _config_path:
    import satori.task_center
    _task_center_dir = Path(satori.task_center.__file__).parent
    _config_path = str(_task_center_dir / "config.yml")
_yaml_config = load_yaml_config(_config_path)

if _yaml_config:
    _log_level = _yaml_config.get('logging.level', 'INFO')
    set_global_log_level(_log_level)
    _db_url = _yaml_config.get('database.url')
    if _db_url:
        from satori.common.database.session import recreate_engine_and_session
        recreate_engine_and_session(_db_url)

from fastapi import FastAPI
from contextlib import asynccontextmanager
from satori.task_center.api import router
from satori.common.database.base import Base
from satori.common.database import engine
from satori.common.services.auto_update import AutoUpdateService
from satori.common.bittensor.client import BittensorClient
from satori.common.config import settings
from satori.common.database import SessionLocal
from satori.task_center.services.task_lifecycle_manager import TaskLifecycleManager
from satori.task_center.services.miner_health_checker import MinerHealthChecker
from satori.task_center.services.idle_reward_distributor import IdleRewardDistributor
from satori.common.bittensor.wallet import WalletManager
from satori.common.utils.thread_pool import get_thread_pool
from satori.task_center.shared import miner_cache, bittensor_client
import satori.task_center.shared as shared_instances

logger = setup_logger(__name__)

Base.metadata.create_all(bind=engine)

config_path = _config_path
yaml_config = _yaml_config

if yaml_config:
    wallet_name = yaml_config.get('wallet.name', 'task_center')
    hotkey_name = yaml_config.get('wallet.hotkey', 'default')
    auto_update_config = yaml_config.get_auto_update_config()
    database_url = yaml_config.get('database.url', settings.DATABASE_URL)
    heartbeat_timeout = yaml_config.get('task_center.miner_heartbeat_timeout', 120)
    miner_cache.heartbeat_timeout = heartbeat_timeout
else:
    wallet_name = "task_center"
    hotkey_name = "default"
    auto_update_config = {}
    database_url = settings.DATABASE_URL

if database_url != settings.DATABASE_URL:
    from satori.common.database.session import recreate_engine_and_session
    recreate_engine_and_session(database_url)
    logger.info(f"Database URL updated to: {database_url}")

try:
    shared_instances.bittensor_client = BittensorClient(wallet_name, hotkey_name, yaml_config=yaml_config)
except Exception as e:
    logger.warning(f"Bittensor client initialization failed (service will continue): {e}")
    shared_instances.bittensor_client = None

task_center_wallet_manager = WalletManager(wallet_name, hotkey_name)
shared_instances.wallet_manager = task_center_wallet_manager
shared_instances.yaml_config = yaml_config

if yaml_config:
    github_repo = yaml_config.get_github_repo()
    auto_update_enabled = yaml_config.get_auto_update_enabled()
    check_interval = yaml_config.get_auto_update_interval()
else:
    github_repo = settings.GITHUB_REPO
    auto_update_enabled = settings.AUTO_UPDATE_ENABLED
    check_interval = 300

auto_update = AutoUpdateService(
    github_repo=github_repo or "satori/task_center",
    branch=auto_update_config.get('branch', 'main'),
    check_interval=check_interval,
    restart_delay=auto_update_config.get('restart_delay', 10)
)

lifecycle_manager = None
miner_health_checker = None
idle_reward_distributor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global lifecycle_manager, miner_health_checker, idle_reward_distributor

    log_file = yaml_config.get('logging.file') if yaml_config else None
    reinitialize_all_loggers(log_file)

    logger.info("Task Center service starting up")
    logger.info(f"Config loaded from: {config_path if yaml_config else 'default'}")

    if shared_instances.bittensor_client:
        try:
            shared_instances.bittensor_client.sync_metagraph()
        except Exception as e:
            logger.warning(f"Failed to sync metagraph (network may be unavailable): {e}")
    else:
        logger.warning("Bittensor client not available - some features may be limited")

    try:
        db = SessionLocal()
        lifecycle_manager = TaskLifecycleManager(
            db,
            wallet_manager=task_center_wallet_manager,
            yaml_config=yaml_config
        )
        await lifecycle_manager.start()
        logger.info("Task lifecycle manager started")
    except Exception as e:
        logger.error(f"Failed to start task lifecycle manager: {e}", exc_info=True)

    try:
        db = SessionLocal()
        check_interval = yaml_config.get('task_center.miner_health_check_interval', 600) if yaml_config else 600
        heartbeat_timeout = yaml_config.get('task_center.miner_heartbeat_timeout', 120) if yaml_config else 120
        miner_health_checker = MinerHealthChecker(
            db,
            task_center_wallet_manager,
            miner_cache,
            check_interval=check_interval,
            heartbeat_timeout=heartbeat_timeout,
            yaml_config=yaml_config
        )
        await miner_health_checker.start()
        logger.info("Miner health checker started")
    except Exception as e:
        logger.error(f"Failed to start miner health checker: {e}", exc_info=True)

    try:
        db = SessionLocal()
        idle_score_interval = yaml_config.get('task_center.idle_score_interval', 360) if yaml_config else 360
        idle_reward_distributor = IdleRewardDistributor(
            db,
            wallet_manager=task_center_wallet_manager,
            yaml_config=yaml_config,
            score_interval=idle_score_interval
        )
        await idle_reward_distributor.start()
        logger.info("Idle reward distributor started")
    except Exception as e:
        logger.error(f"Failed to start idle reward distributor: {e}", exc_info=True)

    if auto_update_enabled:
        try:
            await auto_update.start()
        except Exception as e:
            logger.error(f"Failed to start auto-update: {e}", exc_info=True)

    try:
        yield
    finally:
        logger.info("Task Center service shutting down")

        if lifecycle_manager:
            try:
                await lifecycle_manager.stop()
            except Exception as e:
                logger.error(f"Error stopping lifecycle manager: {e}", exc_info=True)

        if miner_health_checker:
            try:
                await miner_health_checker.stop()
            except Exception as e:
                logger.error(f"Error stopping miner health checker: {e}", exc_info=True)

        if idle_reward_distributor:
            try:
                await idle_reward_distributor.stop()
            except Exception as e:
                logger.error(f"Error stopping idle reward distributor: {e}", exc_info=True)

        try:
            await auto_update.stop()
        except Exception as e:
            logger.error(f"Error stopping auto-update: {e}", exc_info=True)

        try:
            thread_pool = get_thread_pool()
            thread_pool.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error shutting down thread pool: {e}", exc_info=True)


app = FastAPI(title="SATORI Task Center", version="1.0.0", lifespan=lifespan)

from satori.common.middleware import add_request_logging
add_request_logging(app, exclude_paths=["/health", "/docs", "/openapi.json", "/redoc"])

app.include_router(router, prefix="/v1")

@app.get("/health")
async def health_check():
    try:
        miners_count = 0
        if shared_instances.bittensor_client and shared_instances.bittensor_client.metagraph:
            try:
                miners_count = len(shared_instances.bittensor_client.get_all_miners())
            except Exception:
                pass

        is_idle = False
        last_idle_distribution = None
        if idle_reward_distributor:
            is_idle = idle_reward_distributor.is_system_currently_idle()
            last_dist_time = idle_reward_distributor.get_last_distribution_time()
            if last_dist_time:
                last_idle_distribution = last_dist_time.isoformat()

        return {
            "status": "ok",
            "miners_count": miners_count,
            "online_miners": miner_cache.get_online_count(),
            "cache_size": miner_cache.get_cache_size(),
            "last_update": miner_cache.get_last_update().isoformat() if miner_cache.get_last_update() else None,
            "bittensor_connected": shared_instances.bittensor_client is not None and shared_instances.bittensor_client.metagraph is not None,
            "system_idle": is_idle,
            "last_idle_distribution": last_idle_distribution
        }
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("TASK_CENTER_HOST", "0.0.0.0")
    port = int(os.getenv("TASK_CENTER_PORT", "8000"))

    uvicorn_log_level = "debug" if _log_level and _log_level.upper() == "DEBUG" else "info"

    logger.info(f"Starting Task Center service on {host}:{port}")
    logger.info("Using asyncio event loop (required for bittensor compatibility)")

    uvicorn.run(
        app,
        host=host,
        port=port,
        loop="asyncio",
        log_level=uvicorn_log_level,
        log_config=None
    )

