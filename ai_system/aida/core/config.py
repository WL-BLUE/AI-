import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigManager:
    _instance: Optional["ConfigManager"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._config:
            self._load_config()

    def _load_config(self) -> None:
        config_path = os.environ.get("AIDA_CONFIG_PATH", "config.yaml")
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = self._default_config()

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "app": {"name": "AIDA", "version": "1.0.0", "debug": False, "host": "0.0.0.0", "port": 8000},
            "database": {"driver": "sqlite", "sqlite_path": "./data/aida.db"},
            "logging": {"level": "INFO", "log_dir": "./logs"},
            "security": {"secret_key": "default-secret-key", "algorithm": "HS256", "access_token_expire_minutes": 30},
        }

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, key_path: str, value: Any) -> None:
        keys = key_path.split(".")
        config = self._config
        for key in keys[:-1]:
            if key not in config or not isinstance(config[key], dict):
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

    @property
    def full_config(self) -> Dict[str, Any]:
        return self._config.copy()

    def reload(self) -> None:
        self._config.clear()
        self._load_config()


settings = ConfigManager()
