from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from aida.core.config import settings
from aida.core.exceptions import VisualizationError
from aida.core.logger import get_logger
from aida.visualization.chart_builder import ChartBuilder
from aida.visualization.heatmap import HeatmapBuilder

logger = get_logger("dashboard_builder")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class DashboardBuilder:
    def __init__(self):
        self.chart_builder = ChartBuilder()
        self.heatmap_builder = HeatmapBuilder()
        self.output_dir = Path(settings.get("report.output_dir", "./data/reports")) / "dashboards"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_sales_dashboard(self, sales_df: pd.DataFrame, analysis_result: Dict[str, Any]) -> Dict[str, str]:
        logger.info("构建销售仪表盘")
        charts = {}

        try:
            if "order_date" in sales_df.columns and "total_amount" in sales_df.columns:
                daily = sales_df.groupby(sales_df["order_date"].dt.date)["total_amount"].sum().reset_index()
                daily.columns = ["date", "total_sales"]
                daily["date"] = pd.to_datetime(daily["date"])
                charts["sales_trend"] = self.chart_builder.sales_trend_chart(daily)
        except Exception as e:
            logger.warning(f"销售趋势图生成失败: {e}")

        try:
            if "category" in sales_df.columns:
                charts["category_pie"] = self.chart_builder.category_pie_chart(sales_df)
        except Exception as e:
            logger.warning(f"品类饼图生成失败: {e}")

        try:
            if "channel" in sales_df.columns:
                charts["channel_bar"] = self.chart_builder.channel_bar_chart(sales_df)
        except Exception as e:
            logger.warning(f"渠道柱状图生成失败: {e}")

        try:
            charts["sales_heatmap"] = self.heatmap_builder.sales_heatmap(sales_df)
        except Exception as e:
            logger.warning(f"销售热力图生成失败: {e}")

        try:
            forecast = analysis_result.get("forecast", {})
            if forecast.get("forecast_values") and "order_date" in sales_df.columns:
                daily = sales_df.groupby(sales_df["order_date"].dt.date)["total_amount"].sum().reset_index()
                daily.columns = ["date", "total_sales"]
                daily["date"] = pd.to_datetime(daily["date"])
                charts["sales_forecast"] = self.chart_builder.sales_forecast_chart(daily, forecast["forecast_values"])
        except Exception as e:
            logger.warning(f"销售预测图生成失败: {e}")

        try:
            numeric_cols = sales_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                charts["correlation"] = self.heatmap_builder.correlation_heatmap(sales_df, numeric_cols[:8])
        except Exception as e:
            logger.warning(f"相关性热力图生成失败: {e}")

        logger.info(f"销售仪表盘构建完成, 生成{len(charts)}个图表")
        return charts

    def build_customer_dashboard(self, customer_df: pd.DataFrame, analysis_result: Dict[str, Any]) -> Dict[str, str]:
        logger.info("构建客户仪表盘")
        charts = {}

        try:
            segment = analysis_result.get("segment_analysis", {})
            if segment.get("status") == "completed":
                charts["customer_segment"] = self.chart_builder.customer_segment_chart(segment)
        except Exception as e:
            logger.warning(f"客户分群图生成失败: {e}")

        try:
            numeric_cols = customer_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                charts["customer_feature_heatmap"] = self.heatmap_builder.customer_feature_heatmap(customer_df, numeric_cols[:8])
        except Exception as e:
            logger.warning(f"客户特征热力图生成失败: {e}")

        try:
            if "segment" in customer_df.columns:
                charts["correlation"] = self.heatmap_builder.customer_feature_heatmap(customer_df, segment_col="segment")
        except Exception as e:
            logger.warning(f"客户分群热力图生成失败: {e}")

        logger.info(f"客户仪表盘构建完成, 生成{len(charts)}个图表")
        return charts

    def build_inventory_dashboard(self, inventory_df: pd.DataFrame, analysis_result: Dict[str, Any]) -> Dict[str, str]:
        logger.info("构建库存仪表盘")
        charts = {}

        try:
            health = analysis_result.get("health_check", {})
            status_dist = health.get("status_distribution", {})
            if status_dist:
                charts["inventory_status"] = self.chart_builder.inventory_status_chart(status_dist)
        except Exception as e:
            logger.warning(f"库存状态图生成失败: {e}")

        try:
            if "category" in inventory_df.columns and "warehouse" in inventory_df.columns:
                charts["inventory_heatmap"] = self.heatmap_builder.inventory_heatmap(inventory_df)
        except Exception as e:
            logger.warning(f"库存热力图生成失败: {e}")

        try:
            numeric_cols = inventory_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                charts["correlation"] = self.heatmap_builder.correlation_heatmap(inventory_df, numeric_cols[:8])
        except Exception as e:
            logger.warning(f"库存相关性热力图生成失败: {e}")

        logger.info(f"库存仪表盘构建完成, 生成{len(charts)}个图表")
        return charts

    def build_comprehensive_dashboard(self, sales_df: pd.DataFrame, customer_df: pd.DataFrame, inventory_df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        logger.info("构建综合仪表盘")
        all_charts = {}

        if sales_df is not None and not sales_df.empty:
            sales_analysis = analysis_results.get("sales_analysis", {})
            all_charts.update(self.build_sales_dashboard(sales_df, sales_analysis))

        if customer_df is not None and not customer_df.empty:
            customer_analysis = analysis_results.get("customer_analysis", {})
            all_charts.update(self.build_customer_dashboard(customer_df, customer_analysis))

        if inventory_df is not None and not inventory_df.empty:
            inventory_analysis = analysis_results.get("inventory_analysis", {})
            all_charts.update(self.build_inventory_dashboard(inventory_df, inventory_analysis))

        logger.info(f"综合仪表盘构建完成, 共生成{len(all_charts)}个图表")
        return all_charts
