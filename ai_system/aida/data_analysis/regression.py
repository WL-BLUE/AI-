from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from aida.core.exceptions import DataAnalysisError
from aida.core.logger import get_logger
from aida.data_analysis.base import BaseAnalyzer

logger = get_logger("regression")

MODEL_REGISTRY = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


class RegressionAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("regression")
        self._model = None
        self._scaler = StandardScaler()

    def analyze(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        target_col = kwargs.pop("target", None)
        feature_cols = kwargs.pop("features", None)
        model_type = kwargs.pop("model_type", "linear")

        if not target_col or target_col not in df.columns:
            raise DataAnalysisError(f"目标列 '{target_col}' 不存在")

        if feature_cols is None:
            feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]

        if not feature_cols:
            raise DataAnalysisError("无可用特征列")

        return self._fit_and_evaluate(df, target_col, feature_cols, model_type, **kwargs)

    def _fit_and_evaluate(self, df: pd.DataFrame, target_col: str, feature_cols: List[str], model_type: str, **kwargs) -> Dict[str, Any]:
        df_clean = df[feature_cols + [target_col]].dropna()
        if len(df_clean) < 10:
            raise DataAnalysisError("有效数据不足(少于10条)")

        X = df_clean[feature_cols].values
        y = df_clean[target_col].values

        test_size = kwargs.get("test_size", 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        model_class = MODEL_REGISTRY.get(model_type, LinearRegression)
        model_params = kwargs.get("model_params", {})
        self._model = model_class(**model_params)
        self._model.fit(X_train_scaled, y_train)

        y_pred = self._model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        cv_scores = cross_val_score(self._model, X_train_scaled, y_train, cv=min(5, len(X_train) // 2), scoring="r2")

        feature_importance = {}
        if hasattr(self._model, "coef_"):
            for name, coef in zip(feature_cols, self._model.coef_):
                feature_importance[name] = round(float(coef), 4)
        elif hasattr(self._model, "feature_importances_"):
            for name, imp in zip(feature_cols, self._model.feature_importances_):
                feature_importance[name] = round(float(imp), 4)

        result = {
            "model_type": model_type,
            "features": feature_cols,
            "target": target_col,
            "sample_size": len(df_clean),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "metrics": {
                "r2": round(float(r2), 4),
                "rmse": round(float(rmse), 4),
                "mae": round(float(mae), 4),
                "cv_r2_mean": round(float(cv_scores.mean()), 4),
                "cv_r2_std": round(float(cv_scores.std()), 4),
            },
            "feature_importance": feature_importance,
        }

        logger.info(f"回归分析完成: R²={r2:.4f}, RMSE={rmse:.4f}")
        return result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise DataAnalysisError("模型未训练")
        X_scaled = self._scaler.transform(X.values)
        return self._model.predict(X_scaled)
