from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from aida.core.config import settings
from aida.core.exceptions import VisualizationError
from aida.core.logger import get_logger

logger = get_logger("heatmap_builder")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class HeatmapBuilder:
    def __init__(self):
        self.fig_size = tuple(settings.get("visualization.figure_size", [12, 8]))
        self.dpi = settings.get("visualization.dpi", 150)
        self.output_dir = Path(settings.get("report.output_dir", "./data/reports")) / "charts"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_fig(self, fig: plt.Figure, name: str) -> str:
        filepath = self.output_dir / f"{name}.png"
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"热力图已保存: {filepath}")
        return str(filepath)

    def correlation_heatmap(self, df: pd.DataFrame, columns: Optional[List[str]] = None, title: str = "特征相关性热力图") -> str:
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if columns:
                numeric_df = numeric_df[[c for c in columns if c in numeric_df.columns]]

            corr = numeric_df.corr()

            fig, ax = plt.subplots(figsize=self.fig_size)
            mask = np.triu(np.ones_like(corr, dtype=bool))

            sns.heatmap(
                corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title(title, fontsize=16, fontweight="bold")
            plt.tight_layout()
            return self._save_fig(fig, "correlation_heatmap")
        except Exception as e:
            logger.error(f"相关性热力图生成失败: {e}")
            raise VisualizationError(f"相关性热力图生成失败: {e}")

    def sales_heatmap(self, df: pd.DataFrame, date_col: str = "order_date", value_col: str = "total_amount", title: str = "销售热力图") -> str:
        try:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df["weekday"] = df[date_col].dt.dayofweek
            df["hour"] = df[date_col].dt.hour if df[date_col].dt.hour.nunique() > 1 else 0
            df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)

            if df["hour"].nunique() > 1:
                pivot = df.groupby(["weekday", "hour"])[value_col].sum().reset_index()
                pivot_table = pivot.pivot(index="weekday", columns="hour", values=value_col).fillna(0)
                ylabel = "星期"
                xlabel = "小时"
                yticklabels = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            else:
                pivot = df.groupby(["week_of_year", "weekday"])[value_col].sum().reset_index()
                pivot_table = pivot.pivot(index="week_of_year", columns="weekday", values=value_col).fillna(0)
                ylabel = "周次"
                xlabel = "星期"
                yticklabels = None

            fig, ax = plt.subplots(figsize=self.fig_size)
            sns.heatmap(pivot_table, cmap="YlOrRd", annot=False, fmt=".0f", ax=ax, linewidths=0.5)

            ax.set_title(title, fontsize=16, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

            if yticklabels and len(pivot_table.index) <= 7:
                ax.set_yticklabels(yticklabels[: len(pivot_table.index)], rotation=0)

            plt.tight_layout()
            return self._save_fig(fig, "sales_heatmap")
        except Exception as e:
            logger.error(f"销售热力图生成失败: {e}")
            raise VisualizationError(f"销售热力图生成失败: {e}")

    def inventory_heatmap(self, df: pd.DataFrame, category_col: str = "category", warehouse_col: str = "warehouse", value_col: str = "current_stock", title: str = "库存分布热力图") -> str:
        try:
            pivot = df.groupby([category_col, warehouse_col])[value_col].sum().reset_index()
            pivot_table = pivot.pivot(index=category_col, columns=warehouse_col, values=value_col).fillna(0)

            fig, ax = plt.subplots(figsize=self.fig_size)
            sns.heatmap(pivot_table, cmap="Blues", annot=True, fmt=".0f", ax=ax, linewidths=0.5)

            ax.set_title(title, fontsize=16, fontweight="bold")
            ax.set_xlabel("仓库", fontsize=12)
            ax.set_ylabel("品类", fontsize=12)
            plt.tight_layout()
            return self._save_fig(fig, "inventory_heatmap")
        except Exception as e:
            logger.error(f"库存热力图生成失败: {e}")
            raise VisualizationError(f"库存热力图生成失败: {e}")

    def customer_feature_heatmap(self, df: pd.DataFrame, features: Optional[List[str]] = None, segment_col: str = "segment", title: str = "客户特征热力图") -> str:
        try:
            if features is None:
                features = [c for c in df.select_dtypes(include=[np.number]).columns if c != "customer_id"]
                features = features[:8]

            if segment_col not in df.columns:
                return self.correlation_heatmap(df, features, title)

            segment_avg = df.groupby(segment_col)[features].mean()
            segment_avg = (segment_avg - segment_avg.min()) / (segment_avg.max() - segment_avg.min() + 1e-8)

            fig, ax = plt.subplots(figsize=self.fig_size)
            sns.heatmap(segment_avg, cmap="YlGnBu", annot=True, fmt=".2f", ax=ax, linewidths=0.5)

            ax.set_title(title, fontsize=16, fontweight="bold")
            ax.set_xlabel("特征", fontsize=12)
            ax.set_ylabel("客户群体", fontsize=12)
            plt.tight_layout()
            return self._save_fig(fig, "customer_feature_heatmap")
        except Exception as e:
            logger.error(f"客户特征热力图生成失败: {e}")
            raise VisualizationError(f"客户特征热力图生成失败: {e}")
