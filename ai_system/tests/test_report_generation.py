import pytest
import pandas as pd
import numpy as np

from aida.report_generation.generator import ReportGenerator
from aida.report_generation.nlg_engine import NLGEngine
from aida.report_generation.templates import ReportTemplateManager


class TestNLGEngine:
    def test_generate_sales_trend(self):
        nlg = NLGEngine()
        text = nlg.generate_text("sales_trend", {
            "trend": {"direction": "上升", "growth_rate": 2.5, "r2": 0.85},
            "seasonality": {"has_seasonality": True, "peak_day_of_week": "周六", "peak_month": "12月"},
        })
        assert isinstance(text, str)
        assert "上升" in text

    def test_generate_inventory_health(self):
        nlg = NLGEngine()
        text = nlg.generate_text("inventory_health", {
            "health_check": {"total_products": 100, "total_stock_value": 500000, "status_distribution": {"缺货": 5, "低库存": 10}},
        })
        assert isinstance(text, str)
        assert "100" in text


class TestTemplateManager:
    def test_list_templates(self):
        mgr = ReportTemplateManager()
        templates = mgr.list_templates()
        assert "comprehensive" in templates
        assert "sales" in templates

    def test_render_html(self):
        mgr = ReportTemplateManager()
        html = mgr.render_html("comprehensive", {"overview": {"metrics": {"测试": "值"}, "text": "测试文本"}})
        assert "<html" in html
        assert "测试" in html


class TestReportGenerator:
    def test_generate_report(self, tmp_path):
        generator = ReportGenerator()
        generator._output_dir = tmp_path

        analysis_results = {
            "sales_analysis": {
                "trend": {"direction": "上升", "growth_rate": 2.5, "r2": 0.85},
                "seasonality": {"has_seasonality": False},
                "forecast": {"forecast_days": 7, "model_r2": 0.8, "model_mae": 100, "forecast_values": []},
            },
        }
        result = generator.generate(analysis_results, template_name="sales")
        assert result["status"] == "success"
        assert "filepath" in result
