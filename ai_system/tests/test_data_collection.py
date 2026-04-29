import pytest
import pandas as pd
import numpy as np

from aida.data_collection.sales_collector import SalesCollector
from aida.data_collection.inventory_collector import InventoryCollector
from aida.data_collection.customer_collector import CustomerCollector


class TestSalesCollector:
    def test_generate_sample_data(self):
        collector = SalesCollector(source_config={"type": "sample"})
        df = collector.fetch()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "order_id" in df.columns
        assert "total_amount" in df.columns
        assert "order_date" in df.columns

    def test_collect(self):
        collector = SalesCollector(source_config={"type": "sample"})
        df = collector.collect()
        assert collector.last_collected is not None
        assert collector.data is not None

    def test_metadata(self):
        collector = SalesCollector(source_config={"type": "sample"})
        collector.collect()
        meta = collector.get_metadata()
        assert meta["source_name"] == "sales"
        assert meta["record_count"] > 0


class TestInventoryCollector:
    def test_generate_sample_data(self):
        collector = InventoryCollector(source_config={"type": "sample"})
        df = collector.fetch()
        assert isinstance(df, pd.DataFrame)
        assert "current_stock" in df.columns
        assert "stock_status" in df.columns


class TestCustomerCollector:
    def test_generate_sample_data(self):
        collector = CustomerCollector(source_config={"type": "sample"})
        df = collector.fetch()
        assert isinstance(df, pd.DataFrame)
        assert "customer_id" in df.columns
        assert "total_spent" in df.columns
