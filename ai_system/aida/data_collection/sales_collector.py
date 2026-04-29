from typing import Any, Dict, Optional

import pandas as pd

from aida.core.config import settings
from aida.core.logger import get_logger
from aida.data_collection.base import BaseCollector

logger = get_logger("sales_collector")


class SalesCollector(BaseCollector):
    def __init__(self, source_config: Optional[Dict[str, Any]] = None):
        super().__init__("sales", source_config)

    def fetch(self, **kwargs) -> pd.DataFrame:
        source_type = self.config.get("type", "csv")
        try:
            if source_type == "csv":
                return self._fetch_from_csv(**kwargs)
            elif source_type == "excel":
                return self._fetch_from_excel(**kwargs)
            elif source_type == "database":
                return self._fetch_from_database(**kwargs)
            elif source_type == "api":
                return self._fetch_from_api(**kwargs)
        except Exception as e:
            logger.warning(f"数据源获取失败，使用样本数据: {e}")
        return self._generate_sample_data(**kwargs)

    def _fetch_from_csv(self, **kwargs) -> pd.DataFrame:
        path = kwargs.get("path", self.config.get("path", ""))
        return self._read_csv(path)

    def _fetch_from_excel(self, **kwargs) -> pd.DataFrame:
        path = kwargs.get("path", self.config.get("path", ""))
        sheet_name = kwargs.get("sheet_name", 0)
        return self._read_excel(path, sheet_name=sheet_name)

    def _fetch_from_database(self, **kwargs) -> pd.DataFrame:
        conn_str = kwargs.get("connection_string", self.config.get("connection_string", ""))
        query = kwargs.get("query", "SELECT * FROM sales")
        return self._read_from_database(conn_str, query)

    def _fetch_from_api(self, **kwargs) -> pd.DataFrame:
        url = kwargs.get("url", self.config.get("url", ""))
        params = kwargs.get("params", self.config.get("params", {}))
        headers = kwargs.get("headers", self.config.get("headers", {}))
        return self._read_from_api(url, params, headers)

    def _generate_sample_data(self, num_records: int = 1000, **kwargs) -> pd.DataFrame:
        import numpy as np

        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")
        n = min(num_records, len(dates))

        products = [f"产品{i:03d}" for i in range(1, 51)]
        categories = ["电子产品", "服装", "食品", "家居", "运动"]
        regions = ["华东", "华南", "华北", "西南", "东北"]
        channels = ["线上", "线下", "直播", "分销"]

        data = {
            "order_id": [f"ORD{i:06d}" for i in range(1, n + 1)],
            "order_date": np.random.choice(dates, n),
            "product_id": np.random.choice(products, n),
            "category": np.random.choice(categories, n),
            "region": np.random.choice(regions, n),
            "channel": np.random.choice(channels, n),
            "quantity": np.random.randint(1, 50, n),
            "unit_price": np.round(np.random.uniform(10, 2000, n), 2),
            "discount": np.round(np.random.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2], n), 2),
            "customer_id": [f"CUST{np.random.randint(1, 200):04d}" for _ in range(n)],
        }
        df = pd.DataFrame(data)
        df["total_amount"] = np.round(df["quantity"] * df["unit_price"] * (1 - df["discount"]), 2)
        df["delivery_date"] = df["order_date"] + pd.to_timedelta(np.random.randint(1, 10, n), unit="D")
        df["order_date"] = pd.to_datetime(df["order_date"])
        df["delivery_date"] = pd.to_datetime(df["delivery_date"])

        logger.info(f"生成销售样本数据: {n} 条记录")
        return df
