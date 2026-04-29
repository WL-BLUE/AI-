from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from aida.core.exceptions import DataAnalysisError
from aida.core.logger import get_logger
from aida.data_analysis.base import BaseAnalyzer

logger = get_logger("clustering")

CLUSTER_REGISTRY = {
    "kmeans": KMeans,
    "dbscan": DBSCAN,
    "agglomerative": AgglomerativeClustering,
}


class ClusteringAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("clustering")
        self._model = None
        self._scaler = StandardScaler()
        self._pca = None

    def analyze(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        feature_cols = kwargs.pop("features", None)
        algorithm = kwargs.pop("algorithm", "kmeans")
        max_clusters = kwargs.pop("max_clusters", 10)

        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not feature_cols:
            raise DataAnalysisError("无可用特征列")

        return self._cluster(df, feature_cols, algorithm, max_clusters, **kwargs)

    def _cluster(self, df: pd.DataFrame, feature_cols: List[str], algorithm: str, max_clusters: int, **kwargs) -> Dict[str, Any]:
        df_clean = df[feature_cols].dropna()
        if len(df_clean) < 10:
            raise DataAnalysisError("有效数据不足")

        X = df_clean.values
        X_scaled = self._scaler.fit_transform(X)

        if algorithm == "kmeans":
            optimal_k, scores = self._find_optimal_k(X_scaled, max_clusters)
            n_clusters = kwargs.get("n_clusters", optimal_k)
            self._model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = self._model.fit_predict(X_scaled)
        elif algorithm == "dbscan":
            eps = kwargs.get("eps", 0.5)
            min_samples = kwargs.get("min_samples", 5)
            self._model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = self._model.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        elif algorithm == "agglomerative":
            n_clusters = kwargs.get("n_clusters", 3)
            self._model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = self._model.fit_predict(X_scaled)
        else:
            raise DataAnalysisError(f"不支持的聚类算法: {algorithm}")

        n_clusters_actual = len(set(labels)) - (1 if -1 in labels else 0)

        sil_score = -1
        if n_clusters_actual >= 2 and len(set(labels)) > 1:
            try:
                sil_score = silhouette_score(X_scaled, labels)
            except Exception:
                pass

        ch_score = -1
        if n_clusters_actual >= 2:
            try:
                ch_score = calinski_harabasz_score(X_scaled, labels)
            except Exception:
                pass

        pca = PCA(n_components=min(2, len(feature_cols)))
        X_pca = pca.fit_transform(X_scaled)

        cluster_stats = {}
        for label in sorted(set(labels)):
            if label == -1:
                continue
            mask = labels == label
            cluster_stats[f"cluster_{label}"] = {
                "count": int(mask.sum()),
                "percentage": round(float(mask.sum() / len(labels) * 100), 2),
                "center": {col: round(float(df_clean.loc[mask, col].mean()), 4) for col in feature_cols},
            }

        result = {
            "algorithm": algorithm,
            "n_clusters": n_clusters_actual,
            "features": feature_cols,
            "sample_size": len(df_clean),
            "metrics": {
                "silhouette_score": round(float(sil_score), 4),
                "calinski_harabasz_score": round(float(ch_score), 4),
            },
            "cluster_stats": cluster_stats,
            "labels": labels.tolist(),
            "pca_coordinates": X_pca.tolist(),
        }

        if algorithm == "kmeans":
            result["elbow_scores"] = scores

        logger.info(f"聚类分析完成: 算法={algorithm}, 簇数={n_clusters_actual}, 轮廓系数={sil_score:.4f}")
        return result

    def _find_optimal_k(self, X: np.ndarray, max_k: int) -> tuple:
        scores = {}
        k_range = range(2, min(max_k + 1, len(X) // 2))
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            if len(set(labels)) > 1:
                scores[int(k)] = round(float(kmeans.inertia_), 4)

        optimal_k = 3
        if len(scores) >= 3:
            k_list = list(scores.keys())
            inertia_list = list(scores.values())
            diffs = [inertia_list[i] - inertia_list[i + 1] for i in range(len(inertia_list) - 1)]
            if diffs:
                max_diff_idx = diffs.index(max(diffs))
                optimal_k = k_list[max_diff_idx + 1]

        return optimal_k, scores
