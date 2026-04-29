import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from aida.core.config import settings
from aida.core.exceptions import VisualizationError
from aida.core.logger import get_logger

logger = get_logger("chart_builder")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class ChartBuilder:
    def __init__(self):
        self.theme = settings.get("visualization.theme", "whitegrid")
        self.palette = settings.get("visualization.color_palette", "Set2")
        self.fig_size = tuple(settings.get("visualization.figure_size", [12, 8]))
        self.dpi = settings.get("visualization.dpi", 150)
        self.output_dir = Path(settings.get("report.output_dir", "./data/reports")) / "charts"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sns.set_style(self.theme)
        sns.set_palette(self.palette)

    def _save_fig(self, fig: plt.Figure, name: str) -> str:
        filepath = self.output_dir / f"{name}.png"
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"图表已保存: {filepath}")
        return str(filepath)

    def sales_trend_chart(self, daily_sales: pd.DataFrame, title: str = "销售趋势") -> str:
        try:
            fig, ax = plt.subplots(figsize=self.fig_size)
            ax.plot(daily_sales["date"], daily_sales["total_sales"], linewidth=1.5, alpha=0.8, color="#667eea")
            ax.fill_between(daily_sales["date"], daily_sales["total_sales"], alpha=0.2, color="#667eea")

            rolling = daily_sales["total_sales"].rolling(window=7, min_periods=1).mean()
            ax.plot(daily_sales["date"], rolling, linewidth=2.5, color="#764ba2", label="7日移动平均")

            ax.set_title(title, fontsize=16, fontweight="bold")
            ax.set_xlabel("日期", fontsize=12)
            ax.set_ylabel("销售额", fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            return self._save_fig(fig, "sales_trend")
        except Exception as e:
            logger.error(f"销售趋势图生成失败: {e}")
            raise VisualizationError(f"销售趋势图生成失败: {e}")

    def sales_forecast_chart(self, historical: pd.DataFrame, forecast_data: List[Dict], title: str = "销售预测") -> str:
        try:
            fig, ax = plt.subplots(figsize=self.fig_size)

            ax.plot(historical["date"], historical["total_sales"], linewidth=1.5, color="#667eea", label="历史数据", alpha=0.8)

            forecast_dates = [pd.Timestamp(f["date"]) for f in forecast_data]
            forecast_values = [f["predicted_sales"] for f in forecast_data]
            lower = [f["lower_bound"] for f in forecast_data]
            upper = [f["upper_bound"] for f in forecast_data]

            ax.plot(forecast_dates, forecast_values, linewidth=2, color="#e74c3c", label="预测值", linestyle="--")
            ax.fill_between(forecast_dates, lower, upper, alpha=0.2, color="#e74c3c", label="置信区间")

            ax.axvline(x=historical["date"].max(), color="gray", linestyle=":", alpha=0.5, label="预测起点")
            ax.set_title(title, fontsize=16, fontweight="bold")
            ax.set_xlabel("日期", fontsize=12)
            ax.set_ylabel("销售额", fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            return self._save_fig(fig, "sales_forecast")
        except Exception as e:
            logger.error(f"销售预测图生成失败: {e}")
            raise VisualizationError(f"销售预测图生成失败: {e}")

    def category_pie_chart(self, df: pd.DataFrame, category_col: str = "category", value_col: str = "total_amount", title: str = "品类销售占比") -> str:
        try:
            cat_data = df.groupby(category_col)[value_col].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = sns.color_palette(self.palette, len(cat_data))
            wedges, texts, autotexts = ax.pie(
                cat_data.values, labels=cat_data.index, autopct="%1.1f%%",
                colors=colors, startangle=90, pctdistance=0.85,
            )
            for text in autotexts:
                text.set_fontsize(10)
            ax.set_title(title, fontsize=16, fontweight="bold")
            return self._save_fig(fig, "category_pie")
        except Exception as e:
            logger.error(f"品类饼图生成失败: {e}")
            raise VisualizationError(f"品类饼图生成失败: {e}")

    def channel_bar_chart(self, df: pd.DataFrame, channel_col: str = "channel", value_col: str = "total_amount", title: str = "渠道销售对比") -> str:
        try:
            ch_data = df.groupby(channel_col)[value_col].agg(["sum", "count", "mean"]).reset_index()
            ch_data.columns = [channel_col, "total_sales", "order_count", "avg_order"]

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            sns.barplot(data=ch_data, x=channel_col, y="total_sales", ax=axes[0], palette=self.palette)
            axes[0].set_title("总销售额", fontsize=13)
            axes[0].set_xlabel("")

            sns.barplot(data=ch_data, x=channel_col, y="order_count", ax=axes[1], palette=self.palette)
            axes[1].set_title("订单数", fontsize=13)
            axes[1].set_xlabel("")

            sns.barplot(data=ch_data, x=channel_col, y="avg_order", ax=axes[2], palette=self.palette)
            axes[2].set_title("平均订单额", fontsize=13)
            axes[2].set_xlabel("")

            fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
            plt.tight_layout()
            return self._save_fig(fig, "channel_bar")
        except Exception as e:
            logger.error(f"渠道柱状图生成失败: {e}")
            raise VisualizationError(f"渠道柱状图生成失败: {e}")

    def customer_segment_chart(self, segment_data: Dict, title: str = "客户分群") -> str:
        try:
            profiles = segment_data.get("cluster_profiles", {})
            if not profiles:
                raise VisualizationError("无分群数据")

            names = list(profiles.keys())
            counts = [profiles[n].get("count", 0) for n in names]
            percentages = [profiles[n].get("percentage", 0) for n in names]

            fig, axes = plt.subplots(1, 2, figsize=(16, 7))

            colors = sns.color_palette(self.palette, len(names))
            axes[0].bar(names, counts, color=colors)
            axes[0].set_title("各群体人数", fontsize=13)
            axes[0].set_ylabel("人数")

            axes[1].pie(percentages, labels=names, autopct="%1.1f%%", colors=colors, startangle=90)
            axes[1].set_title("各群体占比", fontsize=13)

            fig.suptitle(title, fontsize=16, fontweight="bold")
            plt.tight_layout()
            return self._save_fig(fig, "customer_segment")
        except Exception as e:
            logger.error(f"客户分群图生成失败: {e}")
            raise VisualizationError(f"客户分群图生成失败: {e}")

    def inventory_status_chart(self, status_dist: Dict, title: str = "库存状态分布") -> str:
        try:
            labels = list(status_dist.keys())
            sizes = list(status_dist.values())
            colors_map = {"充足": "#27ae60", "正常": "#2ecc71", "低库存": "#f39c12", "缺货": "#e74c3c"}
            colors = [colors_map.get(l, "#95a5a6") for l in labels]

            fig, ax = plt.subplots(figsize=(10, 8))
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90,
            )
            ax.set_title(title, fontsize=16, fontweight="bold")
            return self._save_fig(fig, "inventory_status")
        except Exception as e:
            logger.error(f"库存状态图生成失败: {e}")
            raise VisualizationError(f"库存状态图生成失败: {e}")

    def rfm_scatter_chart(self, rfm_data: pd.DataFrame, title: str = "RFM分析散点图") -> str:
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            if "recency" in rfm_data.columns and "frequency" in rfm_data.columns:
                axes[0].scatter(rfm_data["recency"], rfm_data["frequency"], alpha=0.5, s=20, c="#667eea")
                axes[0].set_xlabel("Recency (最近购买天数)")
                axes[0].set_ylabel("Frequency (购买频次)")
                axes[0].set_title("R vs F")

            if "recency" in rfm_data.columns and "monetary" in rfm_data.columns:
                axes[1].scatter(rfm_data["recency"], rfm_data["monetary"], alpha=0.5, s=20, c="#e74c3c")
                axes[1].set_xlabel("Recency (最近购买天数)")
                axes[1].set_ylabel("Monetary (消费金额)")
                axes[1].set_title("R vs M")

            if "frequency" in rfm_data.columns and "monetary" in rfm_data.columns:
                axes[2].scatter(rfm_data["frequency"], rfm_data["monetary"], alpha=0.5, s=20, c="#27ae60")
                axes[2].set_xlabel("Frequency (购买频次)")
                axes[2].set_ylabel("Monetary (消费金额)")
                axes[2].set_title("F vs M")

            fig.suptitle(title, fontsize=16, fontweight="bold")
            plt.tight_layout()
            return self._save_fig(fig, "rfm_scatter")
        except Exception as e:
            logger.error(f"RFM散点图生成失败: {e}")
            raise VisualizationError(f"RFM散点图生成失败: {e}")
