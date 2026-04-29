from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from aida.core.config import settings
from aida.core.exceptions import DataAnalysisError
from aida.core.logger import get_logger
from aida.data_analysis.base import BaseAnalyzer

logger = get_logger("sales_predictor")


class SalesPredictor(BaseAnalyzer):
    def __init__(self):
        super().__init__("sales_predictor")
        self._model = None
        self._scaler = StandardScaler()

    def analyze(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        self._validate_dataframe(df, ["order_date", "total_amount"])

        date_col = kwargs.get("date_col", "order_date")
        amount_col = kwargs.get("amount_col", "total_amount")
        forecast_days = kwargs.get("forecast_days", settings.get("analysis.sales_prediction.forecast_days", 30))

        daily_sales = self._aggregate_daily(df, date_col, amount_col)
        trend_result = self._analyze_trend(daily_sales)
        seasonality_result = self._analyze_seasonality(daily_sales)
        forecast_result = self._forecast(daily_sales, forecast_days)

        return {
            "trend": trend_result,
            "seasonality": seasonality_result,
            "forecast": forecast_result,
            "summary": self._generate_summary(trend_result, seasonality_result, forecast_result),
        }

    def _aggregate_daily(self, df: pd.DataFrame, date_col: str, amount_col: str) -> pd.DataFrame:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        daily = df.groupby(df[date_col].dt.date)[amount_col].agg(["sum", "count"]).reset_index()
        daily.columns = ["date", "total_sales", "order_count"]
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date").reset_index(drop=True)

        full_range = pd.date_range(start=daily["date"].min(), end=daily["date"].max(), freq="D")
        daily = daily.set_index("date").reindex(full_range).fillna(0).reset_index()
        daily.columns = ["date", "total_sales", "order_count"]
        return daily

    def _analyze_trend(self, daily: pd.DataFrame) -> Dict[str, Any]:
        if len(daily) < 7:
            return {"direction": "unknown", "growth_rate": 0}

        x = np.arange(len(daily))
        y = daily["total_sales"].values

        mask = y > 0
        if mask.sum() < 3:
            return {"direction": "stable", "growth_rate": 0}

        model = LinearRegression()
        model.fit(x[mask].reshape(-1, 1), y[mask])
        slope = model.coef_[0]
        mean_val = y[mask].mean()
        growth_rate = (slope / mean_val * 100) if mean_val > 0 else 0

        if growth_rate > 1:
            direction = "上升"
        elif growth_rate < -1:
            direction = "下降"
        else:
            direction = "平稳"

        return {
            "direction": direction,
            "growth_rate": round(float(growth_rate), 2),
            "slope": round(float(slope), 2),
            "r2": round(float(model.score(x[mask].reshape(-1, 1), y[mask])), 4),
        }

    def _analyze_seasonality(self, daily: pd.DataFrame) -> Dict[str, Any]:
        if len(daily) < 14:
            return {"has_seasonality": False}

        daily = daily.copy()
        daily["day_of_week"] = daily["date"].dt.dayofweek
        daily["month"] = daily["date"].dt.month

        dow_avg = daily.groupby("day_of_week")["total_sales"].mean()
        month_avg = daily.groupby("month")["total_sales"].mean()

        dow_names = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"}
        month_names = {i: f"{i}月" for i in range(1, 13)}

        peak_dow = dow_avg.idxmax()
        peak_month = month_avg.idxmax()

        overall_avg = daily["total_sales"].mean()
        has_seasonality = (dow_avg.max() / overall_avg > 1.3) if overall_avg > 0 else False

        return {
            "has_seasonality": has_seasonality,
            "day_of_week_pattern": {dow_names.get(k, str(k)): round(float(v), 2) for k, v in dow_avg.items()},
            "monthly_pattern": {month_names.get(k, str(k)): round(float(v), 2) for k, v in month_avg.items()},
            "peak_day_of_week": dow_names.get(peak_dow, str(peak_dow)),
            "peak_month": month_names.get(peak_month, str(peak_month)),
        }

    def _forecast(self, daily: pd.DataFrame, forecast_days: int) -> Dict[str, Any]:
        if len(daily) < 14:
            return {"forecast_values": [], "method": "insufficient_data"}

        df = daily.copy()
        df["day_index"] = np.arange(len(df))
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month

        for dow in range(7):
            df[f"dow_{dow}"] = (df["day_of_week"] == dow).astype(int)

        feature_cols = ["day_index"] + [f"dow_{i}" for i in range(7)]
        X = df[feature_cols].values
        y = df["total_sales"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        self._model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        self._model.fit(X_train, y_train)

        y_pred_test = self._model.predict(X_test)
        r2 = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)

        last_index = df["day_index"].max()
        future_dates = pd.date_range(start=daily["date"].max() + pd.Timedelta(days=1), periods=forecast_days, freq="D")

        future_features = []
        for i, date in enumerate(future_dates):
            dow = date.dayofweek
            features = [last_index + i + 1] + [1 if d == dow else 0 for d in range(7)]
            future_features.append(features)

        X_future = np.array(future_features)
        forecast_values = self._model.predict(X_future)

        std_residual = np.std(y_test - y_pred_test)
        confidence_level = settings.get("analysis.sales_prediction.confidence_level", 0.95)
        from scipy import stats

        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        forecast_result = []
        for i, (date, val) in enumerate(zip(future_dates, forecast_values)):
            forecast_result.append({
                "date": date.strftime("%Y-%m-%d"),
                "predicted_sales": round(float(max(0, val)), 2),
                "lower_bound": round(float(max(0, val - z_score * std_residual)), 2),
                "upper_bound": round(float(val + z_score * std_residual), 2),
            })

        return {
            "method": "gradient_boosting",
            "forecast_days": forecast_days,
            "model_r2": round(float(r2), 4),
            "model_mae": round(float(mae), 2),
            "confidence_level": confidence_level,
            "forecast_values": forecast_result,
        }

    def _generate_summary(self, trend: Dict, seasonality: Dict, forecast: Dict) -> str:
        parts = []
        direction = trend.get("direction", "未知")
        growth = trend.get("growth_rate", 0)
        parts.append(f"销售趋势呈{direction}态势，日均增长率{growth:.2f}%")

        if seasonality.get("has_seasonality"):
            peak_dow = seasonality.get("peak_day_of_week", "未知")
            peak_month = seasonality.get("peak_month", "未知")
            parts.append(f"存在明显季节性，{peak_dow}和{peak_month}为销售高峰")

        if forecast.get("forecast_values"):
            total_forecast = sum(f["predicted_sales"] for f in forecast["forecast_values"])
            parts.append(f"未来{forecast.get('forecast_days', 30)}天预测总销售额约{total_forecast:,.0f}元")

        return "；".join(parts)
