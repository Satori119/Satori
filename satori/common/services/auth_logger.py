from typing import Optional, Dict
from datetime import datetime, timezone
from fastapi import Request
from satori.common.utils.logging import setup_logger
import json

logger = setup_logger(__name__)

class AuthLogger:
    async def log_auth_failure(
        self,
        request: Request,
        hotkey: str,
        failure_reason: str,
        additional_info: Optional[Dict] = None
    ):
        try:
            client_ip = request.client.host if request.client else "unknown"
            method = request.method
            path = request.url.path
            query_params = dict(request.query_params)
            
            log_data = {
                "event": "auth_failure",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "hotkey": hotkey,
                "hotkey_short": hotkey[:16] + "..." if len(hotkey) > 16 else hotkey,
                "client_ip": client_ip,
                "method": method,
                "path": path,
                "query_params": query_params if query_params else None,
                "failure_reason": failure_reason,
                "additional_info": additional_info
            }
            
            logger.warning(
                f"AUTH_FAILURE | hotkey={log_data['hotkey_short']} | "
                f"reason={failure_reason} | ip={client_ip} | "
                f"method={method} | path={path} | "
                f"details={json.dumps(log_data, ensure_ascii=False)}"
            )
            
        except Exception as e:
            logger.error(f"Error in log_auth_failure: {e}", exc_info=True)
    
    async def log_auth_success(
        self,
        request: Request,
        hotkey: str
    ):
        try:
            client_ip = request.client.host if request.client else "unknown"
            method = request.method
            path = request.url.path
            
            logger.debug(
                f"AUTH_SUCCESS | hotkey={hotkey[:16]}... | "
                f"ip={client_ip} | method={method} | path={path}"
            )
        except Exception as e:
            logger.error(f"Error in log_auth_success: {e}")
