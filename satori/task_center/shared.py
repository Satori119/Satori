from typing import Optional, Any
from satori.task_center.services.miner_cache import MinerCache
from satori.common.bittensor.wallet import WalletManager
from satori.common.config.yaml_config import YamlConfig

miner_cache = MinerCache()

bittensor_client: Optional[Any] = None
wallet_manager: Optional[WalletManager] = None
yaml_config: Optional[YamlConfig] = None

