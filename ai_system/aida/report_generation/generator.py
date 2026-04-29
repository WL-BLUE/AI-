import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from aida.core.config import settings
from aida.core.exceptions import ReportGenerationError
from aida.core.logger import get_logger
from aida.report_generation.nlg_engine import NLGEngine
from aida.report_generation.templates import ReportTemplateManager

logger = get_logger("report_generator")


class ReportGenerator:
    def __init__(self):
        self._nlg = NLGEngine(language=settings.get("report.language", "zh-CN"))
        self._template_mgr = ReportTemplateManager()
        self._output_dir = Path(settings.get("report.output_dir", "./data/reports"))
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        analysis_results: Dict[str, Any],
        template_name: str = "comprehensive",
        output_format: str = "html",
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        logger.info(f"开始生成报告, 模板: {template_name}, 格式: {output_format}")

        try:
            sections_data = self._build_sections(analysis_results)
            html_content = self._template_mgr.render_html(template_name, sections_data, metadata)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{template_name}_{timestamp}"
            filepath = self._output_dir / f"{filename}.html"

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"报告已生成: {filepath}")
            return {
                "status": "success",
                "filepath": str(filepath),
                "filename": filename,
                "format": "html",
                "sections": list(sections_data.keys()),
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"报告生成失败: {e}")
            raise ReportGenerationError(f"报告生成失败: {e}")

    def _build_sections(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        sections = {}

        sales_result = analysis_results.get("sales_analysis", {})
        customer_result = analysis_results.get("customer_analysis", {})
        inventory_result = analysis_results.get("inventory_analysis", {})

        overview_metrics = {}
        if sales_result:
            sales_data = sales_result.get("raw_data")
            if sales_data is not None and hasattr(sales_data, "shape"):
                overview_metrics["销售记录数"] = f"{len(sales_data):,}"
                if "total_amount" in sales_data.columns:
                    overview_metrics["总销售额"] = f"{sales_data['total_amount'].sum():,.0f}元"
        if customer_result:
            cust_data = customer_result.get("raw_data")
            if cust_data is not None and hasattr(cust_data, "shape"):
                overview_metrics["客户总数"] = f"{len(cust_data):,}"
        if inventory_result:
            inv_data = inventory_result.get("raw_data")
            if inv_data is not None and hasattr(inv_data, "shape"):
                overview_metrics["商品总数"] = f"{len(inv_data):,}"

        sections["overview"] = {
            "metrics": overview_metrics,
            "text": self._nlg.generate_text("data_overview", {
                "sales_count": overview_metrics.get("销售记录数", "0").replace(",", ""),
                "customer_count": overview_metrics.get("客户总数", "0").replace(",", ""),
                "product_count": overview_metrics.get("商品总数", "0").replace(",", ""),
                "total_revenue": float(overview_metrics.get("总销售额", "0元").replace(",", "").replace("元", "")),
            }),
        }

        if sales_result:
            sections["sales_trend"] = self._build_sales_trend_section(sales_result)
            sections["sales_forecast"] = self._build_sales_forecast_section(sales_result)
            sections["marketing_effect"] = self._build_marketing_section(sales_result)

        if customer_result:
            sections["customer_rfm"] = self._build_customer_rfm_section(customer_result)
            sections["customer_churn"] = self._build_customer_churn_section(customer_result)
            sections["customer_segment"] = self._build_customer_segment_section(customer_result)

        if inventory_result:
            sections["inventory_health"] = self._build_inventory_health_section(inventory_result)
            sections["inventory_abc"] = self._build_inventory_abc_section(inventory_result)
            sections["inventory_reorder"] = self._build_inventory_reorder_section(inventory_result)

        return sections

    def _build_sales_trend_section(self, result: Dict) -> Dict:
        trend = result.get("trend", {})
        seasonality = result.get("seasonality", {})
        metrics = {
            "趋势方向": trend.get("direction", "未知"),
            "增长率": f"{trend.get('growth_rate', 0):.2f}%",
            "趋势R²": f"{trend.get('r2', 0):.4f}",
        }
        text = self._nlg.generate_text("sales_trend", {"trend": trend, "seasonality": seasonality})
        section = {"metrics": metrics, "text": text}

        if trend.get("direction") == "下降":
            section["warning"] = "销售趋势下降，建议分析原因并制定提升策略"
        elif trend.get("direction") == "上升":
            section["success"] = "销售趋势向好，建议保持当前策略"

        return section

    def _build_sales_forecast_section(self, result: Dict) -> Dict:
        forecast = result.get("forecast", {})
        if not forecast.get("forecast_values"):
            return {"text": "暂无销售预测数据"}

        metrics = {
            "预测天数": f"{forecast.get('forecast_days', 0)}天",
            "模型R²": f"{forecast.get('model_r2', 0):.4f}",
            "模型MAE": f"{forecast.get('model_mae', 0):,.0f}",
        }
        text = self._nlg.generate_text("sales_forecast", {"forecast": forecast})

        forecast_table = forecast.get("forecast_values", [])[:10]
        if forecast_table:
            for item in forecast_table:
                item["预测销售额"] = f"{item.pop('predicted_sales', 0):,.0f}"
                item["下限"] = f"{item.pop('lower_bound', 0):,.0f}"
                item["上限"] = f"{item.pop('upper_bound', 0):,.0f}"

        section = {"metrics": metrics, "text": text}
        if forecast_table:
            section["table"] = forecast_table

        return section

    def _build_marketing_section(self, result: Dict) -> Dict:
        raw_data = result.get("raw_data")
        if raw_data is None or not hasattr(raw_data, "columns"):
            return {"text": "暂无营销效果数据"}

        channel_col = "channel" if "channel" in raw_data.columns else None
        amount_col = "total_amount" if "total_amount" in raw_data.columns else None
        if not channel_col or not amount_col:
            return {"text": "暂无营销效果数据"}

        channel_stats = raw_data.groupby(channel_col)[amount_col].agg(["sum", "count", "mean"]).round(2)
        total_sales = channel_stats["sum"].sum()

        channel_data = {}
        metrics = {}
        for channel, row in channel_stats.iterrows():
            pct = row["sum"] / total_sales * 100 if total_sales > 0 else 0
            channel_data[channel] = {"total_sales": row["sum"], "count": int(row["count"]), "avg_order": row["mean"], "sales_pct": round(pct, 1)}
            metrics[f"{channel}销售额"] = f"{row['sum']:,.0f}元"

        text = self._nlg.generate_text("marketing_effect", {"channel_analysis": channel_data})
        return {"metrics": metrics, "text": text}

    def _build_customer_rfm_section(self, result: Dict) -> Dict:
        rfm = result.get("rfm_analysis", {})
        if rfm.get("status") != "completed":
            return {"text": "RFM分析未完成"}

        metrics = {
            "客户总数": f"{rfm.get('total_customers', 0):,}",
            "平均消费": f"{rfm.get('avg_monetary', 0):,.0f}元",
            "平均频次": f"{rfm.get('avg_frequency', 0):.1f}次",
        }
        text = self._nlg.generate_text("customer_rfm", {"rfm_analysis": rfm})

        segments = rfm.get("segments", {})
        table = [{"客户分层": k, "人数": v, "占比": f"{v / rfm.get('total_customers', 1) * 100:.1f}%"} for k, v in segments.items()]

        return {"metrics": metrics, "text": text, "table": table}

    def _build_customer_churn_section(self, result: Dict) -> Dict:
        churn = result.get("churn_analysis", {})
        if churn.get("status") != "completed":
            return {"text": "流失分析未完成"}

        metrics = {
            "平均流失概率": f"{churn.get('avg_churn_probability', 0):.1%}",
            "高风险客户": f"{churn.get('high_churn_count', 0)}人",
            "高风险占比": f"{churn.get('high_churn_pct', 0):.1f}%",
        }
        text = self._nlg.generate_text("customer_churn", {"churn_analysis": churn})

        section = {"metrics": metrics, "text": text}
        if churn.get("high_churn_pct", 0) > 20:
            section["warning"] = "高流失风险客户占比较高，建议立即采取挽留措施"

        return section

    def _build_customer_segment_section(self, result: Dict) -> Dict:
        seg = result.get("segment_analysis", {})
        if seg.get("status") != "completed":
            return {"text": "客户分群分析未完成"}

        metrics = {"群体数量": f"{seg.get('n_clusters', 0)}", "轮廓系数": f"{seg.get('silhouette_score', 0):.4f}"}
        text = self._nlg.generate_text("customer_segment", {"segment_analysis": seg})

        profiles = seg.get("cluster_profiles", {})
        table = []
        for name, profile in profiles.items():
            row = {"群体": name, "人数": profile.get("count", 0), "占比": f"{profile.get('percentage', 0):.1f}%"}
            for k, v in profile.items():
                if k not in ["count", "percentage"]:
                    row[k] = v
            table.append(row)

        return {"metrics": metrics, "text": text, "table": table}

    def _build_inventory_health_section(self, result: Dict) -> Dict:
        health = result.get("health_check", {})
        metrics = {
            "商品总数": f"{health.get('total_products', 0):,}",
            "库存总值": f"{health.get('total_stock_value', 0):,.0f}元",
            "周转率": f"{health.get('inventory_turnover_rate', 0):.2f}次/年",
        }
        text = self._nlg.generate_text("inventory_health", {"health_check": health})

        status_dist = health.get("status_distribution", {})
        table = [{"状态": k, "数量": v} for k, v in status_dist.items()]

        section = {"metrics": metrics, "text": text, "table": table}
        if status_dist.get("缺货", 0) > 0 or status_dist.get("低库存", 0) > 0:
            section["warning"] = f"存在{status_dist.get('缺货', 0)}个缺货和{status_dist.get('低库存', 0)}个低库存商品"

        return section

    def _build_inventory_abc_section(self, result: Dict) -> Dict:
        abc = result.get("abc_analysis", {})
        if abc.get("status") != "completed":
            return {"text": "ABC分析未完成"}

        summary = abc.get("summary", {})
        metrics = {}
        for cls in ["A", "B", "C"]:
            info = summary.get(cls, {})
            metrics[f"{cls}类商品数"] = f"{info.get('product_count', 0)}个"

        text = self._nlg.generate_text("inventory_abc", {"abc_analysis": abc})

        table = [{"类别": cls, "商品数": info.get("product_count", 0), "占比": f"{info.get('product_pct', 0):.1f}%", "价值占比": f"{info.get('value_pct', 0):.1f}%"} for cls, info in summary.items()]

        return {"metrics": metrics, "text": text, "table": table}

    def _build_inventory_reorder_section(self, result: Dict) -> Dict:
        reorder = result.get("reorder_analysis", {})
        if reorder.get("status") != "completed":
            return {"text": "补货分析未完成"}

        metrics = {"需补货商品": f"{reorder.get('products_needing_reorder', 0)}个", "占比": f"{reorder.get('reorder_percentage', 0):.1f}%"}
        text = self._nlg.generate_text("inventory_reorder", {"reorder_analysis": reorder})

        section = {"metrics": metrics, "text": text}
        if reorder.get("reorder_percentage", 0) > 30:
            section["warning"] = "补货需求较高，建议优先处理"

        reorder_list = reorder.get("reorder_list", [])
        if reorder_list:
            section["table"] = reorder_list

        return section
