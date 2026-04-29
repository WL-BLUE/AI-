from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from aida.core.config import settings
from aida.core.exceptions import AuthenticationError, AuthorizationError
from aida.core.logger import get_logger

logger = get_logger("security")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin",
    },
    "analyst": {
        "username": "analyst",
        "hashed_password": pwd_context.hash("analyst123"),
        "role": "analyst",
    },
    "viewer": {
        "username": "viewer",
        "hashed_password": pwd_context.hash("viewer123"),
        "role": "viewer",
    },
}

ROLE_PERMISSIONS = {
    "admin": ["read", "write", "delete", "manage_users", "run_analysis", "generate_report"],
    "analyst": ["read", "write", "run_analysis", "generate_report"],
    "viewer": ["read"],
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.get("security.access_token_expire_minutes", 30)))
    to_encode.update({"exp": expire})
    secret_key = settings.get("security.secret_key", "default-secret-key")
    algorithm = settings.get("security.algorithm", "HS256")
    return jwt.encode(to_encode, secret_key, algorithm=algorithm)


def decode_access_token(token: str) -> dict:
    try:
        secret_key = settings.get("security.secret_key", "default-secret-key")
        algorithm = settings.get("security.algorithm", "HS256")
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        return payload
    except JWTError as e:
        logger.error(f"Token decode failed: {e}")
        raise AuthenticationError("无效的访问令牌")


def check_permission(token: str, required_permission: str) -> bool:
    payload = decode_access_token(token)
    role = payload.get("role", "viewer")
    permissions = ROLE_PERMISSIONS.get(role, [])
    if required_permission not in permissions:
        raise AuthorizationError(f"角色 '{role}' 无 '{required_permission}' 权限")
    return True


def mask_sensitive_data(data: dict, sensitive_keys: list = None) -> dict:
    if sensitive_keys is None:
        sensitive_keys = ["password", "token", "secret", "api_key", "credit_card", "phone", "email"]
    masked = data.copy()
    for key in masked:
        if any(sk in key.lower() for sk in sensitive_keys):
            masked[key] = "***MASKED***"
    return masked
