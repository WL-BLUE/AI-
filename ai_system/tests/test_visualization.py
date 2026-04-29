import pytest
import pandas as pd
import numpy as np

from aida.visualization.chart_builder import ChartBuilder
from aida.visualization.heatmap import HeatmapBuilder


@pytest.fixture
def sample_sales_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "order_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "total_amount": np.random.uniform(100, 5000, n),
        "category": np.random.choice(["电子产品", "服装", "食品"], n),
        "channel": np.random.choice(["线上", "线下", "直播"], n),
        "quantity": np.random.randint(1, 50, n),
        "unit_price": np.random.uniform(10, 1000, n),
    })


class TestChartBuilder:
    def test_sales_trend_chart(self, sample_sales_df, tmp_path):
        builder = ChartBuilder()
        builder.output_dir = tmp_path
        builder._save_fig = lambda fig, name: str(tmp_path / f"{name}.png")

        daily = sample_sales_df.groupby("order_date")["total_amount"].sum().reset_index()
        daily.columns = ["date", "total_sales"]
        path = builder.sales_trend_chart(daily)
        assert path.endswith(".png")

    def test_category_pie_chart(self, sample_sales_df, tmp_path):
        builder = ChartBuilder()
        builder.output_dir = tmp_path
        builder._save_fig = lambda fig, name: str(tmp_path / f"{name}.png")

        path = builder.category_pie_chart(sample_sales_df)
        assert path.endswith(".png")


class TestHeatmapBuilder:
    def test_correlation_heatmap(self, sample_sales_df, tmp_path):
        builder = HeatmapBuilder()
        builder.output_dir = tmp_path
        builder._save_fig = lambda fig, name: str(tmp_path / f"{name}.png")

        path = builder.correlation_heatmap(sample_sales_df, ["total_amount", "quantity", "unit_price"])
        assert path.endswith(".png")
