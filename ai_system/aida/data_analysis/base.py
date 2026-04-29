from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd

from aida.core.exceptions import DataAnalysisError
from aida.core.logger import get_logger


class BaseAnalyzer(ABC):
    def __init__(self, name: str):
        self.name = name
        self._results: Dict[str, Any] = {}

    @abstractmethod
    def analyze(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        pass

    def run(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        logger = get_logger(self.name)
        logger.info(f"开始分析: {self.name}")
        try:
            self._results = self.analyze(df, **kwargs)
            logger.info(f"分析完成: {self.name}")
            return self._results
        except DataAnalysisError:
            raise
        except Exception as e:
            logger.error(f"分析失败 [{self.name}]: {e}")
            raise DataAnalysisError(f"{self.name} 分析失败: {e}")

    @property
    def results(self) -> Dict[str, Any]:
        return self._results.copy()

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, required_columns: Optional[list] = None) -> None:
        if df is None or df.empty:
            raise DataAnalysisError("数据为空")
        if required_columns:
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                raise DataAnalysisError(f"缺少必要列: {missing}")
