from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from aida.core.config import settings
from aida.core.exceptions import DataCleaningError
from aida.core.logger import get_logger

logger = get_logger("data_cleaning")


class DataCleaner:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or settings.get("data_cleaning", {})
        self._cleaning_report: Dict[str, Any] = {}

    def clean(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        logger.info(f"开始数据清洗, 原始记录数: {len(df)}")
        original_count = len(df)
        self._cleaning_report = {"original_count": original_count}

        df = df.copy()

        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        df = self.handle_outliers(df)
        df = self.fix_data_types(df)
        df = self.normalize_text(df)

        self._cleaning_report["final_count"] = len(df)
        self._cleaning_report["records_removed"] = original_count - len(df)
        logger.info(
            f"数据清洗完成, 最终记录数: {len(df)}, 移除: {original_count - len(df)}"
        )
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        check_cols = self.config.get("duplicate_check_columns")
        before = len(df)
        if check_cols:
            df = df.drop_duplicates(subset=check_cols, keep="first")
        else:
            df = df.drop_duplicates(keep="first")
        removed = before - len(df)
        self._cleaning_report["duplicates_removed"] = removed
        if removed > 0:
            logger.info(f"移除重复记录: {removed}")
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        strategy = self.config.get("missing_value_strategy", "auto")
        missing_info = df.isnull().sum()
        total_missing = missing_info.sum()
        self._cleaning_report["missing_values"] = missing_info[missing_info > 0].to_dict()

        if total_missing == 0:
            return df

        logger.info(f"处理缺失值, 总计: {total_missing}, 策略: {strategy}")

        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue

            missing_pct = df[col].isnull().sum() / len(df)

            if missing_pct > 0.5:
                logger.warning(f"列 '{col}' 缺失率超过50% ({missing_pct:.1%}), 将删除该列")
                df = df.drop(columns=[col])
                continue

            if strategy == "auto":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].fillna(method="ffill")
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "未知")
            elif strategy == "drop":
                df = df.dropna(subset=[col])
            elif strategy == "mean":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val.iloc[0])
            elif strategy == "zero":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(0)

        return df

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        method = self.config.get("outlier_method", "iqr")
        threshold = self.config.get("outlier_threshold", 1.5)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}

        for col in numeric_cols:
            if method == "iqr":
                lower, upper = self._iqr_bounds(df[col], threshold)
            elif method == "zscore":
                lower, upper = self._zscore_bounds(df[col], threshold)
            else:
                continue

            outliers = (df[col] < lower) | (df[col] > upper)
            outlier_count = outliers.sum()
            if outlier_count > 0:
                outlier_counts[col] = int(outlier_count)
                df.loc[outliers, col] = df[col].median()
                logger.info(f"列 '{col}' 处理异常值: {outlier_count} 个")

        self._cleaning_report["outliers_handled"] = outlier_counts
        return df

    @staticmethod
    def _iqr_bounds(series: pd.Series, threshold: float = 1.5) -> Tuple[float, float]:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return q1 - threshold * iqr, q3 + threshold * iqr

    @staticmethod
    def _zscore_bounds(series: pd.Series, threshold: float = 3.0) -> Tuple[float, float]:
        mean = series.mean()
        std = series.std()
        return mean - threshold * std, mean + threshold * std

    def fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        date_format = self.config.get("date_format", "%Y-%m-%d")
        type_fixes = {}

        for col in df.columns:
            original_dtype = str(df[col].dtype)

            if "date" in col.lower() or "time" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], format=date_format, errors="coerce")
                    if str(df[col].dtype) != original_dtype:
                        type_fixes[col] = f"{original_dtype} -> {df[col].dtype}"
                except Exception:
                    pass

            if df[col].dtype == "object":
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.05 and df[col].nunique() <= self.config.get("categorical_max_unique", 50):
                    df[col] = df[col].astype("category")
                    type_fixes[col] = f"{original_dtype} -> category"

        self._cleaning_report["type_fixes"] = type_fixes
        return df

    def normalize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        text_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in text_cols:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(r"\s+", " ", regex=True)
        return df

    @property
    def cleaning_report(self) -> Dict[str, Any]:
        return self._cleaning_report.copy()
