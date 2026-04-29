from typing import Any, Dict, Optional

import pandas as pd

from aida.core.config import settings
from aida.core.logger import get_logger
from aida.data_collection.base import BaseCollector

logger = get_logger("customer_collector")


class CustomerCollector(BaseCollector):
    def __init__(self, source_config: Optional[Dict[str, Any]] = None):
        super().__init__("customer", source_config)

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
        query = kwargs.get("query", "SELECT * FROM customers")
        return self._read_from_database(conn_str, query)

    def _generate_sample_data(self, num_customers: int = 500, **kwargs) -> pd.DataFrame:
        import numpy as np

        np.random.seed(44)
        customer_ids = [f"CUST{i:04d}" for i in range(1, num_customers + 1)]
        segments = ["高价值", "中等价值", "低价值", "新客户", "流失风险"]
        regions = ["华东", "华南", "华北", "西南", "东北"]
        genders = ["男", "女"]

        data = {
            "customer_id": customer_ids,
            "segment": np.random.choice(segments, num_customers, p=[0.15, 0.35, 0.25, 0.15, 0.10]),
            "region": np.random.choice(regions, num_customers),
            "gender": np.random.choice(genders, num_customers),
            "age": np.random.randint(18, 70, num_customers),
            "register_date": pd.date_range(start="2022-01-01", periods=num_customers, freq="h"),
            "total_spent": np.round(np.random.exponential(5000, num_customers), 2),
            "purchase_frequency": np.random.randint(1, 100, num_customers),
            "avg_order_value": np.round(np.random.uniform(50, 3000, num_customers), 2),
            "days_since_last_purchase": np.random.randint(0, 365, num_customers),
            "satisfaction_score": np.round(np.random.uniform(1, 5, num_customers), 1),
            "churn_probability": np.round(np.random.beta(2, 5, num_customers), 3),
        }
        df = pd.DataFrame(data)

        logger.info(f"生成客户样本数据: {num_customers} 条记录")
        return df
