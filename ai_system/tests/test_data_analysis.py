import pytest
import pandas as pd
import numpy as np

from aida.data_analysis.regression import RegressionAnalyzer
from aida.data_analysis.clustering import ClusteringAnalyzer
from aida.data_analysis.sales_predictor import SalesPredictor
from aida.data_analysis.customer_analyzer import CustomerAnalyzer
from aida.data_analysis.inventory_optimizer import InventoryOptimizer


@pytest.fixture
def sales_df():
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "order_id": [f"ORD{i:06d}" for i in range(n)],
        "order_date": dates,
        "product_id": [f"P{i:03d}" for i in np.random.randint(1, 50, n)],
        "category": np.random.choice(["电子产品", "服装", "食品"], n),
        "channel": np.random.choice(["线上", "线下"], n),
        "quantity": np.random.randint(1, 50, n),
        "unit_price": np.round(np.random.uniform(10, 1000, n), 2),
        "discount": np.round(np.random.choice([0, 0.1, 0.2], n), 2),
        "customer_id": [f"C{i:04d}" for i in np.random.randint(1, 100, n)],
        "total_amount": np.round(np.random.uniform(50, 5000, n), 2),
    })


@pytest.fixture
def customer_df():
    np.random.seed(43)
    n = 100
    return pd.DataFrame({
        "customer_id": [f"CUST{i:04d}" for i in range(n)],
        "segment": np.random.choice(["高价值", "中等价值", "低价值"], n),
        "total_spent": np.round(np.random.exponential(5000, n), 2),
        "purchase_frequency": np.random.randint(1, 50, n),
        "avg_order_value": np.round(np.random.uniform(50, 2000, n), 2),
        "days_since_last_purchase": np.random.randint(0, 365, n),
        "churn_probability": np.round(np.random.beta(2, 5, n), 3),
        "satisfaction_score": np.round(np.random.uniform(1, 5, n), 1),
        "region": np.random.choice(["华东", "华南", "华北"], n),
        "age": np.random.randint(18, 70, n),
    })


@pytest.fixture
def inventory_df():
    np.random.seed(44)
    n = 100
    return pd.DataFrame({
        "product_id": [f"P{i:03d}" for i in range(n)],
        "category": np.random.choice(["电子产品", "服装", "食品"], n),
        "warehouse": np.random.choice(["仓库A", "仓库B"], n),
        "current_stock": np.random.randint(0, 5000, n),
        "safety_stock": np.random.randint(50, 500, n),
        "reorder_point": np.random.randint(100, 800, n),
        "unit_cost": np.round(np.random.uniform(5, 1500, n), 2),
        "stock_value": np.round(np.random.uniform(100, 500000, n), 2),
        "daily_demand_avg": np.round(np.random.uniform(1, 100, n), 1),
        "lead_time_days": np.random.randint(1, 30, n),
    })


class TestRegressionAnalyzer:
    def test_linear_regression(self, sales_df):
        analyzer = RegressionAnalyzer()
        result = analyzer.run(sales_df, target="total_amount", features=["quantity", "unit_price", "discount"], model_type="linear")
        assert "metrics" in result
        assert "r2" in result["metrics"]

    def test_random_forest(self, sales_df):
        analyzer = RegressionAnalyzer()
        result = analyzer.run(sales_df, target="total_amount", features=["quantity", "unit_price"], model_type="random_forest")
        assert "feature_importance" in result


class TestClusteringAnalyzer:
    def test_kmeans(self, customer_df):
        analyzer = ClusteringAnalyzer()
        result = analyzer.run(customer_df, features=["total_spent", "purchase_frequency", "days_since_last_purchase"], algorithm="kmeans", max_clusters=5)
        assert "n_clusters" in result
        assert "cluster_stats" in result


class TestSalesPredictor:
    def test_sales_prediction(self, sales_df):
        predictor = SalesPredictor()
        result = predictor.run(sales_df, forecast_days=7)
        assert "trend" in result
        assert "forecast" in result
        assert "summary" in result


class TestCustomerAnalyzer:
    def test_customer_analysis(self, customer_df):
        analyzer = CustomerAnalyzer()
        result = analyzer.run(customer_df)
        assert "rfm_analysis" in result
        assert "churn_analysis" in result


class TestInventoryOptimizer:
    def test_inventory_analysis(self, inventory_df):
        optimizer = InventoryOptimizer()
        result = optimizer.run(inventory_df)
        assert "health_check" in result
        assert "abc_analysis" in result
