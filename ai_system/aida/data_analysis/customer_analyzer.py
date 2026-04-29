from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from aida.core.config import settings
from aida.core.exceptions import DataAnalysisError
from aida.core.logger import get_logger
from aida.data_analysis.base import BaseAnalyzer

logger = get_logger("customer_analyzer")


class CustomerAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("customer_analyzer")
        self._scaler = StandardScaler()

    def analyze(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        self._validate_dataframe(df)

        rfm_result = self._rfm_analysis(df, **kwargs)
        churn_result = self._churn_analysis(df, **kwargs)
        segment_result = self._segment_analysis(df, **kwargs)
        behavior_result = self._behavior_analysis(df, **kwargs)

        return {
            "rfm_analysis": rfm_result,
            "churn_analysis": churn_result,
            "segment_analysis": segment_result,
            "behavior_analysis": behavior_result,
            "summary": self._generate_summary(rfm_result, churn_result, segment_result),
        }

    def _rfm_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        customer_col = kwargs.get("customer_col", "customer_id")
        date_col = kwargs.get("date_col", "order_date")
        amount_col = kwargs.get("amount_col", "total_amount")

        if customer_col not in df.columns:
            return {"status": "skipped", "reason": "缺少客户ID列"}

        df = df.copy()
        reference_date = pd.Timestamp.now()

        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            reference_date = df[date_col].max() + pd.Timedelta(days=1)

            rfm = df.groupby(customer_col).agg({
                date_col: lambda x: (reference_date - x.max()).days,
                customer_col: "count",
                amount_col: "sum",
            })
            rfm.columns = ["recency", "frequency", "monetary"]
        else:
            if "days_since_last_purchase" in df.columns and "purchase_frequency" in df.columns and "total_spent" in df.columns:
                rfm = pd.DataFrame({
                    "recency": df["days_since_last_purchase"],
                    "frequency": df["purchase_frequency"],
                    "monetary": df["total_spent"],
                }, index=df["customer_id"] if "customer_id" in df.columns else df.index)
            else:
                return {"status": "skipped", "reason": "缺少必要的RFM数据列"}

        for col in ["recency", "frequency", "monetary"]:
            try:
                rfm[f"{col}_score"] = pd.qcut(rfm[col], 5, labels=False, duplicates="drop") + 1
            except (ValueError, TypeError):
                n_unique = rfm[col].nunique()
                n_bins = min(5, n_unique)
                if n_bins < 2:
                    rfm[f"{col}_score"] = 3
                else:
                    rfm[f"{col}_score"] = pd.cut(rfm[col], bins=n_bins, labels=False, duplicates="drop") + 1
            rfm[f"{col}_score"] = rfm[f"{col}_score"].fillna(3).astype(int)
        rfm["recency_score"] = 6 - rfm["recency_score"]
        rfm["rfm_score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str) + rfm["monetary_score"].astype(str)
        rfm["rfm_total"] = rfm["recency_score"] + rfm["frequency_score"] + rfm["monetary_score"]

        def rfm_segment(score):
            if score >= 13:
                return "重要价值客户"
            elif score >= 10:
                return "重要发展客户"
            elif score >= 7:
                return "一般价值客户"
            elif score >= 4:
                return "一般保持客户"
            else:
                return "流失预警客户"

        rfm["segment"] = rfm["rfm_total"].apply(rfm_segment)

        segment_counts = rfm["segment"].value_counts()
        return {
            "status": "completed",
            "total_customers": len(rfm),
            "avg_recency": round(float(rfm["recency"].mean()), 1),
            "avg_frequency": round(float(rfm["frequency"].mean()), 1),
            "avg_monetary": round(float(rfm["monetary"].mean()), 2),
            "segments": {k: int(v) for k, v in segment_counts.items()},
        }

    def _churn_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        if "churn_probability" not in df.columns and "days_since_last_purchase" not in df.columns:
            return {"status": "skipped", "reason": "缺少流失相关数据"}

        if "churn_probability" in df.columns:
            churn_prob = df["churn_probability"]
        else:
            days = df["days_since_last_purchase"]
            churn_prob = 1 - np.exp(-days / 180)

        high_churn = (churn_prob > 0.5).sum()
        medium_churn = ((churn_prob > 0.2) & (churn_prob <= 0.5)).sum()
        low_churn = (churn_prob <= 0.2).sum()

        return {
            "status": "completed",
            "avg_churn_probability": round(float(churn_prob.mean()), 4),
            "high_churn_count": int(high_churn),
            "medium_churn_count": int(medium_churn),
            "low_churn_count": int(low_churn),
            "high_churn_pct": round(float(high_churn / len(df) * 100), 2),
        }

    def _segment_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        feature_names = kwargs.get("features", settings.get("analysis.customer_clustering.features", []))

        available_features = [f for f in feature_names if f in df.columns]
        if not available_features:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            available_features = numeric_cols[:4]

        if len(available_features) < 2:
            return {"status": "skipped", "reason": "特征不足"}

        X = df[available_features].fillna(0).values
        X_scaled = self._scaler.fit_transform(X)

        max_clusters = kwargs.get("max_clusters", settings.get("analysis.customer_clustering.max_clusters", 5))
        n_clusters = min(max_clusters, len(df) // 10, 8)
        n_clusters = max(n_clusters, 2)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        from sklearn.metrics import silhouette_score

        sil_score = silhouette_score(X_scaled, labels) if n_clusters >= 2 else -1

        cluster_profiles = {}
        for i in range(n_clusters):
            mask = labels == i
            profile = {col: round(float(df.loc[mask, col].mean()), 2) for col in available_features}
            profile["count"] = int(mask.sum())
            profile["percentage"] = round(float(mask.sum() / len(df) * 100), 2)
            cluster_profiles[f"cluster_{i}"] = profile

        return {
            "status": "completed",
            "n_clusters": n_clusters,
            "silhouette_score": round(float(sil_score), 4),
            "features_used": available_features,
            "cluster_profiles": cluster_profiles,
        }

    def _behavior_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        result = {}

        if "segment" in df.columns:
            seg_stats = df.groupby("segment").agg({
                "total_spent": "mean",
                "purchase_frequency": "mean",
                "satisfaction_score": "mean",
            }).round(2) if all(c in df.columns for c in ["total_spent", "purchase_frequency", "satisfaction_score"]) else None
            if seg_stats is not None:
                result["segment_behavior"] = seg_stats.to_dict()

        if "region" in df.columns and "total_spent" in df.columns:
            region_spend = df.groupby("region")["total_spent"].agg(["mean", "sum", "count"]).round(2)
            result["regional_analysis"] = region_spend.to_dict()

        if "age" in df.columns:
            bins = [0, 25, 35, 45, 55, 100]
            labels = ["18-25", "26-35", "36-45", "46-55", "55+"]
            df_copy = df.copy()
            df_copy["age_group"] = pd.cut(df_copy["age"], bins=bins, labels=labels)
            age_stats = df_copy.groupby("age_group", observed=True).size()
            result["age_distribution"] = {str(k): int(v) for k, v in age_stats.items()}

        return result

    def _generate_summary(self, rfm: Dict, churn: Dict, segment: Dict) -> str:
        parts = []

        if rfm.get("status") == "completed":
            top_seg = max(rfm.get("segments", {}), key=rfm.get("segments", {}).get, default="未知")
            parts.append(f"客户RFM分析完成，最大客户群为'{top_seg}'")

        if churn.get("status") == "completed":
            high_pct = churn.get("high_churn_pct", 0)
            parts.append(f"高流失风险客户占比{high_pct:.1f}%")

        if segment.get("status") == "completed":
            n = segment.get("n_clusters", 0)
            sil = segment.get("silhouette_score", 0)
            parts.append(f"客户分为{n}个群体，轮廓系数{sil:.3f}")

        return "；".join(parts) if parts else "客户分析完成"
