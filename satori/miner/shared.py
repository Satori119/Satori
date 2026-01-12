from satori.miner.services.queue_manager import QueueManager
from satori.common.bittensor.wallet import WalletManager
from satori.common.config.yaml_config import YamlConfig
from typing import Optional

queue_manager: QueueManager = None
wallet_manager: Optional[WalletManager] = None
yaml_config: Optional[YamlConfig] = None

