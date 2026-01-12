import bittensor as bt
from typing import Optional
from satori.common.config import settings
from satori.common.utils.logging import setup_logger

logger = setup_logger(__name__)

class WalletManager:
    def __init__(self, wallet_name: str, hotkey_name: str):
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
        self.axon: Optional[bt.axon] = None

    def get_hotkey(self) -> str:
        return self.wallet.hotkey.ss58_address

    def get_coldkey(self) -> str:
        return self.wallet.coldkeypub.ss58_address

    def get_balance(self) -> float:
        try:
            subtensor = bt.subtensor(network="finney")
            balance = subtensor.get_balance(self.wallet.coldkeypub.ss58_address)
            return float(balance)
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

    def serve_axon(
        self,
        netuid: int,
        ip: str = "0.0.0.0",
        port: int = 8001,
        external_ip: Optional[str] = None,
        chain_endpoint: Optional[str] = None
    ) -> bool:

        try:
            if chain_endpoint:
                subtensor = bt.subtensor(network=chain_endpoint)
            else:
                subtensor = bt.subtensor(network="finney")

            self.axon = bt.axon(
                wallet=self.wallet,
                ip=ip,
                port=port,
                external_ip=external_ip
            )

            success = subtensor.serve_axon(
                netuid=netuid,
                axon=self.axon
            )

            if success:
                logger.info(
                    f"Successfully registered axon on chain: "
                    f"ip={external_ip or ip}, port={port}, netuid={netuid}"
                )
            else:
                logger.error(f"Failed to register axon on chain")

            return success

        except Exception as e:
            logger.error(f"Error serving axon: {e}", exc_info=True)
            return False

