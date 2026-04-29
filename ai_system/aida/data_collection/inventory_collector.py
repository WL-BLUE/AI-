from typing import Any, Dict, Optional

import pandas as pd

from aida.core.config import settings
from aida.core.logger import get_logger
from aida.data_collection.base import BaseCollector

logger = get_logger("inventory_collector")


class InventoryCollector(BaseCollector):
    def __init__(self, source_config: Optional[Dict[str, Any]] = None):
        super().__init__("inventory", source_config)

    def fetch(self, **kwargs) -> pd.DataFrame:
        source_type = self.config.get("type", "csv")
        try:
            if source_type == "csv":
                return self._fetch_from_csv(**kwargs)
            elif source_type == "database":
                return self._fetch_from_database(**kwargs)
        except Exception as e:
            logger.warning(f"数据源获取失败，使用样本数据: {e}")
        return self._generate_sample_data(**kwargs)

    def _fetch_from_csv(self, **kwargs) -> pd.DataFrame:
        path = kwargs.get("path", self.config.get("path", ""))
        return self._read_csv(path)

    def _fetch_from_database(self, **kwargs) -> pd.DataFrame:
        conn_str = kwargs.get("connection_string", self.config.get("connection_string", ""))
        query = kwargs.get("query", "SELECT * FROM inventory")
        return self._read_from_database(conn_str, query)

    def _generate_sample_data(self, num_products: int = 200, **kwargs) -> pd.DataFrame:
        import numpy as np

        np.random.seed(43)
        products = [f"产品{i:03d}" for i in range(1, num_products + 1)]
        categories = ["电子产品", "服装", "食品", "家居", "运动"]
        warehouses = ["仓库A", "仓库B", "仓库C", "仓库D"]

        data = {
            "product_id": products,
            "product_name": [f"商品_{p}" for p in products],
            "category": np.random.choice(categories, num_products),
            "warehouse": np.random.choice(warehouses, num_products),
            "current_stock": np.random.randint(0, 5000, num_products),
            "safety_stock": np.random.randint(50, 500, num_products),
            "reorder_point": np.random.randint(100, 800, num_products),
            "unit_cost": np.round(np.random.uniform(5, 1500, num_products), 2),
            "daily_demand_avg": np.round(np.random.uniform(1, 100, num_products), 1),
            "lead_time_days": np.random.randint(1, 30, num_products),
            "last_restock_date": pd.date_range(start="2025-01-01", periods=num_products, freq="h"),
        }
        df = pd.DataFrame(data)
        df["stock_value"] = np.round(df["current_stock"] * df["unit_cost"], 2)
        df["stock_status"] = df.apply(
            lambda r: "缺货" if r["current_stock"] == 0
            else ("低库存" if r["current_stock"] < r["safety_stock"]
                  else ("正常" if r["current_stock"] < r["reorder_point"]
                        else "充足")),
            axis=1,
        )

        logger.info(f"生成库存样本数据: {num_products} 条记录")
        return df
