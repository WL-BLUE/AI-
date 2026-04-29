from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from aida.core.config import settings
from aida.core.exceptions import DataCollectionError, DataSourceNotFoundError
from aida.core.logger import get_logger

logger = get_logger("data_collection")


class BaseCollector(ABC):
    def __init__(self, source_name: str, source_config: Optional[Dict[str, Any]] = None):
        self.source_name = source_name
        self.config = source_config or settings.get(f"data_sources.{source_name}", {})
        self._last_collected: Optional[datetime] = None
        self._data: Optional[pd.DataFrame] = None

    @abstractmethod
    def fetch(self, **kwargs) -> pd.DataFrame:
        pass

    def collect(self, **kwargs) -> pd.DataFrame:
        logger.info(f"开始采集数据源: {self.source_name}")
        try:
            self._data = self.fetch(**kwargs)
            self._last_collected = datetime.now()
            logger.info(f"数据采集完成: {self.source_name}, 记录数: {len(self._data)}")
            return self._data
        except DataSourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"数据采集失败 [{self.source_name}]: {e}")
            raise DataCollectionError(f"采集 {self.source_name} 失败: {e}")

    @property
    def last_collected(self) -> Optional[datetime]:
        return self._last_collected

    @property
    def data(self) -> Optional[pd.DataFrame]:
        return self._data

    def _read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        file_path = Path(path)
        if not file_path.exists():
            raise DataSourceNotFoundError(path)
        encoding = kwargs.pop("encoding", self.config.get("encoding", "utf-8"))
        date_columns = kwargs.pop("date_columns", self.config.get("date_columns", []))
        try:
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            return df
        except Exception as e:
            raise DataCollectionError(f"读取CSV失败 [{path}]: {e}")

    def _read_excel(self, path: str, **kwargs) -> pd.DataFrame:
        file_path = Path(path)
        if not file_path.exists():
            raise DataSourceNotFoundError(path)
        try:
            return pd.read_excel(file_path, **kwargs)
        except Exception as e:
            raise DataCollectionError(f"读取Excel失败 [{path}]: {e}")

    def _read_from_database(self, connection_string: str, query: str, **kwargs) -> pd.DataFrame:
        try:
            from sqlalchemy import create_engine

            engine = create_engine(connection_string)
            return pd.read_sql(query, engine, **kwargs)
        except Exception as e:
            raise DataCollectionError(f"数据库查询失败: {e}")

    def _read_from_api(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None, **kwargs) -> pd.DataFrame:
        try:
            import httpx

            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict) and "data" in data:
                return pd.DataFrame(data["data"])
            else:
                return pd.DataFrame([data])
        except Exception as e:
            raise DataCollectionError(f"API请求失败 [{url}]: {e}")

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "last_collected": self._last_collected.isoformat() if self._last_collected else None,
            "record_count": len(self._data) if self._data is not None else 0,
            "columns": list(self._data.columns) if self._data is not None else [],
        }
