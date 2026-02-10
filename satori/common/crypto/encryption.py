import json
import base64
import os
from typing import Dict, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from satori.common.utils.logging import setup_logger

logger = setup_logger(__name__)

try:
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC as PBKDF2
    USE_BACKEND = False
except ImportError:
    try:
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
        from cryptography.hazmat.backends import default_backend
        USE_BACKEND = True
    except ImportError:
        raise ImportError("Could not import PBKDF2 or PBKDF2HMAC from cryptography")

class EncryptionService:
    def __init__(self, secret_key: Optional[str] = None, salt: Optional[bytes] = None):
        if secret_key:
            self.secret_key = secret_key.encode()
        else:
            self.secret_key = Fernet.generate_key()
        
        self.salt = salt or os.urandom(16)
        self.fernet = Fernet(self._derive_key(self.secret_key, self.salt))
    
    def _derive_key(self, password: bytes, salt: bytes) -> bytes:
        if USE_BACKEND:
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
        else:
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000
            )
        return base64.urlsafe_b64encode(kdf.derive(password))
    
    def encrypt(self, data: Dict) -> str:
        try:
            json_data = json.dumps(data)
            encrypted = self.fernet.encrypt(json_data.encode())
            return base64.urlsafe_b64encode(self.salt + encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}", exc_info=True)
            raise
    
    def decrypt(self, encrypted_data: str) -> Dict:
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            
            if len(encrypted_bytes) > 16:
                salt = encrypted_bytes[:16]
                ciphertext = encrypted_bytes[16:]
                fernet = Fernet(self._derive_key(self.secret_key, salt))
                decrypted = fernet.decrypt(ciphertext)
            else:
                legacy_salt = b'satori_salt_2024'
                fernet = Fernet(self._derive_key(self.secret_key, legacy_salt))
                decrypted = fernet.decrypt(encrypted_bytes)
            
            return json.loads(decrypted.decode())
        except Exception as e:
            logger.error(f"Decryption failed: {e}", exc_info=True)
            raise
    
    @staticmethod
    def generate_key() -> str:
        return Fernet.generate_key().decode()

