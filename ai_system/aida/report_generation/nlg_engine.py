from typing import Any, Dict, List, Optional

from aida.core.config import settings
from aida.core.exceptions import ReportGenerationError
from aida.core.logger import get_logger

logger = get_logger("nlg_engine")


class NLGEngine:
    def __init__(self, language: str = "zh-CN"):
        self.language = language

    def generate_text(self, analysis_type: str, data: Dict[str, Any]) -> str:
        generators = {
            "sales_trend": self._gen_sales_trend,
            "sales_forecast": self._gen_sales_forecast,
            "customer_rfm": self._gen_customer_rfm,
            "customer_churn": self._gen_customer_churn,
            "customer_segment": self._gen_customer_segment,
            "inventory_health": self._gen_inventory_health,
            "inventory_abc": self._gen_inventory_abc,
            "inventory_reorder": self._gen_inventory_reorder,
            "marketing_effect": self._gen_marketing_effect,
            "data_overview": self._gen_data_overview,
        }
        generator = generators.get(analysis_type, self._gen_generic)
        return generator(data)

    def _gen_sales_trend(self, data: Dict) -> str:
        trend = data.get("trend", {})
        direction = trend.get("direction", "未知")
        growth = trend.get("growth_rate", 0)
        seasonality = data.get("seasonality", {})

        text = f"本期销售趋势整体呈{direction}态势"
        if abs(growth) > 0.01:
            text += f"，日均增长率为{growth:.2f}%"
        else:
            text += "，增长基本持平"

        if seasonality.get("has_seasonality"):
            peak_dow = seasonality.get("peak_day_of_week", "")
            peak_month = seasonality.get("peak_month", "")
            text += f"。销售存在明显季节性特征，{peak_dow}为每周销售高峰，{peak_month}为年度销售旺季"

        return text + "。"

    def _gen_sales_forecast(self, data: Dict) -> str:
        forecast = data.get("forecast", {})
        if not forecast.get("forecast_values"):
            return "暂无足够数据进行销售预测。"

        days = forecast.get("forecast_days", 30)
        total_pred = sum(f["predicted_sales"] for f in forecast["forecast_values"])
        avg_pred = total_pred / days if days > 0 else 0
        r2 = forecast.get("model_r2", 0)

        text = f"基于历史数据的机器学习模型（R²={r2:.3f}）预测，未来{days}天总销售额约为{total_pred:,.0f}元"
        text += f"，日均销售额约{avg_pred:,.0f}元"
        text += f"。预测置信度为{forecast.get('confidence_level', 0.95):.0%}"

        return text + "。"

    def _gen_customer_rfm(self, data: Dict) -> str:
        rfm = data.get("rfm_analysis", {})
        if rfm.get("status") != "completed":
            return "RFM分析未完成。"

        segments = rfm.get("segments", {})
        total = rfm.get("total_customers", 0)
        text = f"对{total}名客户进行RFM分析，客户分层结果如下："

        for seg, count in sorted(segments.items(), key=lambda x: x[1], reverse=True):
            pct = count / total * 100 if total > 0 else 0
            text += f"{seg}{count}人（{pct:.1f}%），"

        text = text.rstrip("，") + "。"
        text += f"平均消费金额{rfm.get('avg_monetary', 0):,.0f}元，平均购买频次{rfm.get('avg_frequency', 0):.1f}次。"
        return text

    def _gen_customer_churn(self, data: Dict) -> str:
        churn = data.get("churn_analysis", {})
        if churn.get("status") != "completed":
            return "流失分析未完成。"

        avg_prob = churn.get("avg_churn_probability", 0)
        high_pct = churn.get("high_churn_pct", 0)
        high_count = churn.get("high_churn_count", 0)

        text = f"客户平均流失概率为{avg_prob:.1%}，其中{high_count}名客户（{high_pct:.1f}%）属于高流失风险"
        if high_pct > 20:
            text += "，建议立即采取客户挽留措施"
        elif high_pct > 10:
            text += "，建议关注并制定预防性挽留策略"
        else:
            text += "，整体流失风险可控"

        return text + "。"

    def _gen_customer_segment(self, data: Dict) -> str:
        seg = data.get("segment_analysis", {})
        if seg.get("status") != "completed":
            return "客户分群分析未完成。"

        n = seg.get("n_clusters", 0)
        profiles = seg.get("cluster_profiles", {})
        text = f"通过聚类分析将客户划分为{n}个群体："

        for name, profile in profiles.items():
            text += f"群体{name[-1]}包含{profile.get('count', 0)}人（{profile.get('percentage', 0):.1f}%），"
            key_features = {k: v for k, v in profile.items() if k not in ["count", "percentage"]}
            if key_features:
                feat_str = "、".join(f"{k}={v}" for k, v in list(key_features.items())[:3])
                text += f"特征为{feat_str}；"

        return text.rstrip("；") + "。"

    def _gen_inventory_health(self, data: Dict) -> str:
        health = data.get("health_check", {})
        total = health.get("total_products", 0)
        status_dist = health.get("status_distribution", {})
        turnover = health.get("inventory_turnover_rate", 0)

        text = f"当前库存共{total}个商品，总价值{health.get('total_stock_value', 0):,.0f}元"
        if turnover > 0:
            text += f"，库存周转率为{turnover:.2f}次/年"

        out_of_stock = status_dist.get("缺货", 0)
        low_stock = status_dist.get("低库存", 0)
        if out_of_stock > 0 or low_stock > 0:
            text += f"。库存预警：{out_of_stock}个缺货、{low_stock}个低库存商品需关注"
        else:
            text += "。库存状态整体健康"

        return text + "。"

    def _gen_inventory_abc(self, data: Dict) -> str:
        abc = data.get("abc_analysis", {})
        if abc.get("status") != "completed":
            return "ABC分析未完成。"

        summary = abc.get("summary", {})
        text = "库存ABC分类分析结果："
        for cls in ["A", "B", "C"]:
            info = summary.get(cls, {})
            text += f"{cls}类商品{info.get('product_count', 0)}个（{info.get('product_pct', 0):.1f}%），贡献{info.get('value_pct', 0):.1f}%的库存价值；"

        return text.rstrip("；") + "。建议对A类商品实施重点管理。"

    def _gen_inventory_reorder(self, data: Dict) -> str:
        reorder = data.get("reorder_analysis", {})
        if reorder.get("status") != "completed":
            return "补货分析未完成。"

        count = reorder.get("products_needing_reorder", 0)
        pct = reorder.get("reorder_percentage", 0)

        text = f"当前有{count}个商品（{pct:.1f}%）需要补货"
        if pct > 30:
            text += "，补货需求较高，建议优先处理"
        elif pct > 15:
            text += "，需按优先级安排补货"
        else:
            text += "，补货压力较小"

        return text + "。"

    def _gen_marketing_effect(self, data: Dict) -> str:
        channel_data = data.get("channel_analysis", {})
        if not channel_data:
            return "暂无营销效果数据。"

        text = "渠道营销效果分析："
        for channel, metrics in channel_data.items():
            text += f"{channel}渠道销售额{metrics.get('total_sales', 0):,.0f}元，占比{metrics.get('sales_pct', 0):.1f}%；"

        return text.rstrip("；") + "。"

    def _gen_data_overview(self, data: Dict) -> str:
        sales_count = data.get("sales_count", 0)
        customer_count = data.get("customer_count", 0)
        product_count = data.get("product_count", 0)
        total_revenue = data.get("total_revenue", 0)

        text = f"本次分析涵盖{sales_count}条销售记录、{customer_count}名客户、{product_count}个商品"
        text += f"，总销售额{total_revenue:,.0f}元。"
        return text

    def _gen_generic(self, data: Dict) -> str:
        return f"分析完成。{'; '.join(f'{k}: {v}' for k, v in list(data.items())[:5])}。"
