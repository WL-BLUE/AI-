from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from aida.core.config import settings
from aida.core.exceptions import DataAnalysisError
from aida.core.logger import get_logger
from aida.data_analysis.base import BaseAnalyzer

logger = get_logger("inventory_optimizer")


class InventoryOptimizer(BaseAnalyzer):
    def __init__(self):
        super().__init__("inventory_optimizer")

    def analyze(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        self._validate_dataframe(df)

        health_result = self._health_check(df)
        abc_result = self._abc_analysis(df, **kwargs)
        reorder_result = self._reorder_analysis(df, **kwargs)
        safety_stock_result = self._safety_stock_analysis(df, **kwargs)

        return {
            "health_check": health_result,
            "abc_analysis": abc_result,
            "reorder_analysis": reorder_result,
            "safety_stock": safety_stock_result,
            "summary": self._generate_summary(health_result, abc_result, reorder_result),
        }

    def _health_check(self, df: pd.DataFrame) -> Dict[str, Any]:
        total_products = len(df)
        total_value = df["stock_value"].sum() if "stock_value" in df.columns else 0

        status_col = "stock_status" if "stock_status" in df.columns else None
        if status_col:
            status_counts = df[status_col].value_counts().to_dict()
        else:
            stock_col = "current_stock" if "current_stock" in df.columns else None
            safety_col = "safety_stock" if "safety_stock" in df.columns else None
            if stock_col and safety_col:
                out_of_stock = (df[stock_col] == 0).sum()
                low_stock = (df[stock_col] < df[safety_col]).sum()
                normal = total_products - out_of_stock - low_stock
                status_counts = {"缺货": int(out_of_stock), "低库存": int(low_stock), "正常": int(normal)}
            else:
                status_counts = {}

        turnover_rate = 0
        if "daily_demand_avg" in df.columns and "current_stock" in df.columns:
            avg_demand = df["daily_demand_avg"].sum()
            avg_stock = df["current_stock"].sum()
            if avg_stock > 0:
                turnover_rate = round(float(avg_demand * 365 / avg_stock), 2)

        return {
            "total_products": total_products,
            "total_stock_value": round(float(total_value), 2),
            "status_distribution": {k: int(v) for k, v in status_counts.items()},
            "inventory_turnover_rate": turnover_rate,
        }

    def _abc_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        value_col = "stock_value" if "stock_value" in df.columns else None
        if not value_col:
            if "current_stock" in df.columns and "unit_cost" in df.columns:
                df = df.copy()
                df["stock_value"] = df["current_stock"] * df["unit_cost"]
                value_col = "stock_value"
            else:
                return {"status": "skipped", "reason": "缺少库存价值数据"}

        df_sorted = df.nlargest(len(df), value_col)
        total_value = df_sorted[value_col].sum()
        df_sorted["cumulative_value"] = df_sorted[value_col].cumsum()
        df_sorted["cumulative_pct"] = df_sorted["cumulative_value"] / total_value

        a_threshold = kwargs.get("a_threshold", 0.8)
        b_threshold = kwargs.get("b_threshold", 0.95)

        df_sorted["abc_class"] = "C"
        df_sorted.loc[df_sorted["cumulative_pct"] <= a_threshold, "abc_class"] = "A"
        df_sorted.loc[(df_sorted["cumulative_pct"] > a_threshold) & (df_sorted["cumulative_pct"] <= b_threshold), "abc_class"] = "B"

        abc_summary = {}
        for cls in ["A", "B", "C"]:
            mask = df_sorted["abc_class"] == cls
            abc_summary[cls] = {
                "product_count": int(mask.sum()),
                "product_pct": round(float(mask.sum() / len(df) * 100), 2),
                "value_pct": round(float(df_sorted.loc[mask, value_col].sum() / total_value * 100), 2),
                "total_value": round(float(df_sorted.loc[mask, value_col].sum()), 2),
            }

        return {"status": "completed", "summary": abc_summary}

    def _reorder_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        if not all(c in df.columns for c in ["current_stock", "reorder_point"]):
            return {"status": "skipped", "reason": "缺少库存/补货点数据"}

        need_reorder = df[df["current_stock"] <= df["reorder_point"]]
        reorder_products = need_reorder.shape[0]

        reorder_list = []
        product_col = "product_id" if "product_id" in df.columns else None
        for _, row in need_reorder.head(20).iterrows():
            item = {"current_stock": int(row["current_stock"]), "reorder_point": int(row["reorder_point"])}
            if product_col:
                item["product_id"] = str(row[product_col])
            if "daily_demand_avg" in df.columns:
                days_supply = row["current_stock"] / row["daily_demand_avg"] if row["daily_demand_avg"] > 0 else 0
                item["days_of_supply"] = round(float(days_supply), 1)
            reorder_list.append(item)

        return {
            "status": "completed",
            "products_needing_reorder": int(reorder_products),
            "reorder_percentage": round(float(reorder_products / len(df) * 100), 2),
            "reorder_list": reorder_list,
        }

    def _safety_stock_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        if not all(c in df.columns for c in ["daily_demand_avg", "lead_time_days"]):
            return {"status": "skipped", "reason": "缺少需求/提前期数据"}

        service_level = kwargs.get("service_level", settings.get("analysis.inventory_optimization.service_level", 0.95))
        z_score = stats.norm.ppf(service_level)

        df = df.copy()
        demand_std = df["daily_demand_avg"] * 0.3
        df["calculated_safety_stock"] = np.round(z_score * demand_std * np.sqrt(df["lead_time_days"]), 0)
        df["recommended_reorder_point"] = np.round(df["daily_demand_avg"] * df["lead_time_days"] + df["calculated_safety_stock"], 0)

        if "safety_stock" in df.columns:
            df["safety_stock_diff"] = df["calculated_safety_stock"] - df["safety_stock"]
            over_stocked = (df["safety_stock_diff"] > 0).sum()
            under_stocked = (df["safety_stock_diff"] < 0).sum()
            adjustment_summary = {
                "over_stocked_products": int(over_stocked),
                "under_stocked_products": int(under_stocked),
                "avg_adjustment": round(float(df["safety_stock_diff"].mean()), 1),
            }
        else:
            adjustment_summary = None

        return {
            "status": "completed",
            "service_level": service_level,
            "z_score": round(float(z_score), 4),
            "adjustment_summary": adjustment_summary,
        }

    def _generate_summary(self, health: Dict, abc: Dict, reorder: Dict) -> str:
        parts = []

        status_dist = health.get("status_distribution", {})
        if status_dist:
            out_of_stock = status_dist.get("缺货", 0)
            low_stock = status_dist.get("低库存", 0)
            if out_of_stock > 0 or low_stock > 0:
                parts.append(f"库存预警：{out_of_stock}个缺货，{low_stock}个低库存")
            else:
                parts.append("库存状态整体健康")

        if abc.get("status") == "completed":
            a_info = abc["summary"].get("A", {})
            parts.append(f"A类商品{a_info.get('product_count', 0)}个，贡献{a_info.get('value_pct', 0)}%价值")

        if reorder.get("status") == "completed":
            pct = reorder.get("reorder_percentage", 0)
            parts.append(f"{reorder.get('products_needing_reorder', 0)}个商品需要补货({pct:.1f}%)")

        return "；".join(parts) if parts else "库存优化分析完成"
