from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from aida.core.config import settings
from aida.core.exceptions import DataCleaningError
from aida.core.logger import get_logger

logger = get_logger("ai_cleaner")


class AIDataCleaner:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or settings.get("data_cleaning", {})
        self._isolation_forest: Optional[IsolationForest] = None
        self._scaler = StandardScaler()

    def ai_detect_anomalies(self, df: pd.DataFrame, columns: Optional[List[str]] = None, contamination: float = 0.05) -> pd.DataFrame:
        logger.info("使用AI算法检测异常数据")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if columns:
            numeric_cols = [c for c in columns if c in numeric_cols]

        if not numeric_cols:
            logger.warning("无数值列可用于异常检测")
            return df

        try:
            X = df[numeric_cols].fillna(df[numeric_cols].median())
            X_scaled = self._scaler.fit_transform(X)

            self._isolation_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
            predictions = self._isolation_forest.fit_predict(X_scaled)

            anomaly_mask = predictions == -1
            anomaly_count = anomaly_mask.sum()
            logger.info(f"AI异常检测完成: 发现 {anomaly_count} 个异常 ({anomaly_count / len(df):.2%})")

            df_result = df.copy()
            df_result["_is_anomaly"] = anomaly_mask
            df_result["_anomaly_score"] = self._isolation_forest.decision_function(X_scaled)
            return df_result

        except Exception as e:
            logger.error(f"AI异常检测失败: {e}")
            raise DataCleaningError(f"AI异常检测失败: {e}")

    def ai_impute_missing(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        logger.info("使用AI算法智能填充缺失值")
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if columns:
            numeric_cols = [c for c in columns if c in numeric_cols]

        for col in numeric_cols:
            missing_mask = df[col].isnull()
            if not missing_mask.any():
                continue

            other_cols = [c for c in numeric_cols if c != col and df[c].notna().sum() > 0]
            if not other_cols:
                df[col] = df[col].fillna(df[col].median())
                continue

            train_mask = df[col].notna() & df[other_cols].notna().all(axis=1)
            if train_mask.sum() < 10:
                df[col] = df[col].fillna(df[col].median())
                continue

            try:
                from sklearn.ensemble import GradientBoostingRegressor

                X_train = df.loc[train_mask, other_cols]
                y_train = df.loc[train_mask, col]
                predict_mask = missing_mask & df[other_cols].notna().all(axis=1)

                if predict_mask.sum() == 0:
                    df[col] = df[col].fillna(df[col].median())
                    continue

                X_predict = df.loc[predict_mask, other_cols]

                model = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
                model.fit(X_train, y_train)
                predictions = model.predict(X_predict)
                df.loc[predict_mask, col] = predictions

                remaining_missing = df[col].isnull().sum()
                if remaining_missing > 0:
                    df[col] = df[col].fillna(df[col].median())

                logger.info(f"列 '{col}' AI填充完成, 填充 {predict_mask.sum()} 个缺失值")

            except Exception as e:
                logger.warning(f"列 '{col}' AI填充失败, 回退到中位数: {e}")
                df[col] = df[col].fillna(df[col].median())

        return df

    def ai_deduplicate(self, df: pd.DataFrame, key_columns: Optional[List[str]] = None, similarity_threshold: float = 0.9) -> pd.DataFrame:
        logger.info("使用AI算法智能去重")
        if key_columns is None:
            key_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()[:3]

        if not key_columns:
            return df.drop_duplicates(keep="first")

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            text_data = df[key_columns].fillna("").astype(str).agg(" ".join, axis=1)

            if len(text_data) > 10000:
                logger.info("数据量较大, 使用精确去重")
                return df.drop_duplicates(subset=key_columns, keep="first")

            vectorizer = TfidfVectorizer(max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(text_data)
            sim_matrix = cosine_similarity(tfidf_matrix)

            to_drop = set()
            for i in range(len(sim_matrix)):
                if i in to_drop:
                    continue
                for j in range(i + 1, len(sim_matrix)):
                    if j in to_drop:
                        continue
                    if sim_matrix[i, j] >= similarity_threshold:
                        to_drop.add(j)

            df_result = df.drop(index=list(to_drop)).reset_index(drop=True)
            logger.info(f"AI去重完成: 移除 {len(to_drop)} 条相似记录")
            return df_result

        except Exception as e:
            logger.warning(f"AI去重失败, 回退到精确去重: {e}")
            return df.drop_duplicates(subset=key_columns, keep="first")

    def ai_clean_pipeline(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        logger.info("启动AI数据清洗流水线")
        report = {"original_count": len(df)}

        df = self.ai_deduplicate(df, kwargs.get("key_columns"), kwargs.get("similarity_threshold", 0.9))
        report["after_dedup"] = len(df)

        df = self.ai_impute_missing(df, kwargs.get("impute_columns"))
        report["after_imputation"] = len(df)

        df = self.ai_detect_anomalies(df, kwargs.get("anomaly_columns"), kwargs.get("contamination", 0.05))
        anomaly_count = df["_is_anomaly"].sum()
        report["anomalies_detected"] = int(anomaly_count)

        if kwargs.get("remove_anomalies", False):
            df = df[~df["_is_anomaly"]].drop(columns=["_is_anomaly", "_anomaly_score"])
            report["after_anomaly_removal"] = len(df)
        else:
            df = df.drop(columns=["_is_anomaly", "_anomaly_score"])

        report["final_count"] = len(df)
        logger.info(f"AI数据清洗流水线完成: {report}")
        return df, report
