from satori.common.bittensor.client import BittensorClient
from satori.common.utils.logging import setup_logger
from typing import Optional, Set
import asyncio
from datetime import datetime, timedelta

logger = setup_logger(__name__)

class HotkeyValidator:
    def __init__(self, bittensor_client: BittensorClient):
        self.bittensor_client = bittensor_client
        self._registered_miners: Set[str] = set()
        self._registered_validators: Set[str] = set()
        self._last_sync_time: Optional[datetime] = None
        self._sync_interval = timedelta(minutes=5)
        self._lock = asyncio.Lock()
    
    async def verify_hotkey_registered(
        self,
        hotkey: str,
        node_type: str = "miner"
    ) -> bool:
        await self._sync_if_needed()
        
        if node_type == "miner":
            return hotkey in self._registered_miners
        elif node_type == "validator":
            return hotkey in self._registered_validators
        else:
            return hotkey in self._registered_miners or hotkey in self._registered_validators
    
    async def _sync_if_needed(self):
        async with self._lock:
            now = datetime.now()
            if (self._last_sync_time is None or 
                now - self._last_sync_time > self._sync_interval):
                await self._sync_from_chain()
                self._last_sync_time = now
    
    async def _sync_from_chain(self):
        try:
            if not self.bittensor_client or not self.bittensor_client.metagraph:
                logger.warning("Bittensor client not available, skipping sync")
                return
            
            miners = self.bittensor_client.get_all_miners()
            self._registered_miners = {
                m.get("hotkey") for m in miners 
                if m.get("hotkey")
            }
            
            all_nodes = self.bittensor_client.get_all_miners()
            self._registered_validators = {
                n.get("hotkey") for n in all_nodes 
                if n.get("hotkey")
            }
            
            logger.info(
                f"Synced registered nodes: {len(self._registered_miners)} miners, "
                f"{len(self._registered_validators)} validators"
            )
        except Exception as e:
            logger.error(f"Failed to sync registered nodes: {e}", exc_info=True)
    
    async def force_sync(self):
        async with self._lock:
            await self._sync_from_chain()