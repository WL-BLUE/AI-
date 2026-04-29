from typing import Any, Dict, Optional

import pandas as pd

from aida.core.config import settings
from aida.core.exceptions import AIDAError
from aida.core.logger import get_logger
from aida.data_analysis.customer_analyzer import CustomerAnalyzer
from aida.data_analysis.inventory_optimizer import InventoryOptimizer
from aida.data_analysis.sales_predictor import SalesPredictor
from aida.data_cleaning.ai_cleaner import AIDataCleaner
from aida.data_cleaning.cleaner import DataCleaner
from aida.data_cleaning.validators import DataValidator
from aida.data_collection.customer_collector import CustomerCollector
from aida.data_collection.inventory_collector import InventoryCollector
from aida.data_collection.sales_collector import SalesCollector
from aida.report_generation.generator import ReportGenerator
from aida.visualization.dashboard import DashboardBuilder

logger = get_logger("pipeline")


class AnalysisPipeline:
    def __init__(self):
        self.sales_collector = SalesCollector()
        self.inventory_collector = InventoryCollector()
        self.customer_collector = CustomerCollector()

        self.cleaner = DataCleaner()
        self.ai_cleaner = AIDataCleaner()
        self.validator = DataValidator()

        self.sales_predictor = SalesPredictor()
        self.customer_analyzer = CustomerAnalyzer()
        self.inventory_optimizer = InventoryOptimizer()

        self.report_generator = ReportGenerator()
        self.dashboard_builder = DashboardBuilder()

        self._sales_data: Optional[pd.DataFrame] = None
        self._inventory_data: Optional[pd.DataFrame] = None
        self._customer_data: Optional[pd.DataFrame] = None

        self._sales_clean: Optional[pd.DataFrame] = None
        self._inventory_clean: Optional[pd.DataFrame] = None
        self._customer_clean: Optional[pd.DataFrame] = None

        self._analysis_results: Dict[str, Any] = {}

    def collect_data(self, source_name: str = "all") -> Dict[str, Any]:
        logger.info(f"开始数据采集: {source_name}")
        results = {}

        if source_name in ("all", "sales"):
            self._sales_data = self.sales_collector.collect()
            results["sales"] = {"records": len(self._sales_data), "columns": list(self._sales_data.columns)}

        if source_name in ("all", "inventory"):
            self._inventory_data = self.inventory_collector.collect()
            results["inventory"] = {"records": len(self._inventory_data), "columns": list(self._inventory_data.columns)}

        if source_name in ("all", "customer"):
            self._customer_data = self.customer_collector.collect()
            results["customer"] = {"records": len(self._customer_data), "columns": list(self._customer_data.columns)}

        return results

    def clean_data(self, source_name: str = "all", use_ai: bool = False) -> Dict[str, Any]:
        logger.info(f"开始数据清洗: {source_name}, AI模式: {use_ai}")
        results = {}

        if source_name in ("all", "sales") and self._sales_data is not None:
            if use_ai:
                self._sales_clean, report = self.ai_cleaner.ai_clean_pipeline(self._sales_data)
            else:
                self._sales_clean = self.cleaner.clean(self._sales_data)
                report = self.cleaner.cleaning_report
            validation = self.validator.validate(self._sales_clean)
            results["sales"] = {"cleaning_report": report, "validation": validation}

        if source_name in ("all", "inventory") and self._inventory_data is not None:
            if use_ai:
                self._inventory_clean, report = self.ai_cleaner.ai_clean_pipeline(self._inventory_data)
            else:
                self._inventory_clean = self.cleaner.clean(self._inventory_data)
                report = self.cleaner.cleaning_report
            validation = self.validator.validate(self._inventory_clean)
            results["inventory"] = {"cleaning_report": report, "validation": validation}

        if source_name in ("all", "customer") and self._customer_data is not None:
            if use_ai:
                self._customer_clean, report = self.ai_cleaner.ai_clean_pipeline(self._customer_data)
            else:
                self._customer_clean = self.cleaner.clean(self._customer_data)
                report = self.cleaner.cleaning_report
            validation = self.validator.validate(self._customer_clean)
            results["customer"] = {"cleaning_report": report, "validation": validation}

        return results

    def analyze_sales(self, forecast_days: int = 30) -> Dict[str, Any]:
        if self._sales_clean is None:
            self._ensure_data("sales")

        logger.info("开始销售分析")
        result = self.sales_predictor.run(self._sales_clean, forecast_days=forecast_days)
        self._analysis_results["sales_analysis"] = result
        self._analysis_results["sales_analysis"]["raw_data"] = self._sales_clean
        return {k: v for k, v in result.items() if k != "raw_data"}

    def analyze_customer(self) -> Dict[str, Any]:
        if self._customer_clean is None:
            self._ensure_data("customer")

        logger.info("开始客户分析")
        result = self.customer_analyzer.run(self._customer_clean)
        self._analysis_results["customer_analysis"] = result
        self._analysis_results["customer_analysis"]["raw_data"] = self._customer_clean
        return {k: v for k, v in result.items() if k != "raw_data"}

    def analyze_inventory(self) -> Dict[str, Any]:
        if self._inventory_clean is None:
            self._ensure_data("inventory")

        logger.info("开始库存分析")
        result = self.inventory_optimizer.run(self._inventory_clean)
        self._analysis_results["inventory_analysis"] = result
        self._analysis_results["inventory_analysis"]["raw_data"] = self._inventory_clean
        return {k: v for k, v in result.items() if k != "raw_data"}

    def run_full_analysis(self, forecast_days: int = 30) -> Dict[str, Any]:
        logger.info("开始全量分析流水线")

        if self._sales_data is None:
            self.collect_data("all")

        if self._sales_clean is None:
            self.clean_data("all")

        results = {}

        try:
            results["sales_analysis"] = self.analyze_sales(forecast_days=forecast_days)
        except AIDAError as e:
            logger.error(f"销售分析失败: {e}")
            results["sales_analysis"] = {"error": str(e)}

        try:
            results["customer_analysis"] = self.analyze_customer()
        except AIDAError as e:
            logger.error(f"客户分析失败: {e}")
            results["customer_analysis"] = {"error": str(e)}

        try:
            results["inventory_analysis"] = self.analyze_inventory()
        except AIDAError as e:
            logger.error(f"库存分析失败: {e}")
            results["inventory_analysis"] = {"error": str(e)}

        self._analysis_results["full_results"] = results
        logger.info("全量分析流水线完成")
        return results

    def generate_report(self, template_name: str = "comprehensive") -> Dict[str, Any]:
        if not self._analysis_results:
            self.run_full_analysis()

        report_data = {}
        for key in ["sales_analysis", "customer_analysis", "inventory_analysis"]:
            if key in self._analysis_results:
                report_data[key] = self._analysis_results[key]

        return self.report_generator.generate(report_data, template_name=template_name)

    def generate_visualizations(self) -> Dict[str, Any]:
        if self._sales_clean is None:
            raise AIDAError("请先运行数据采集和清洗")

        sales_analysis = self._analysis_results.get("sales_analysis", {})
        customer_analysis = self._analysis_results.get("customer_analysis", {})
        inventory_analysis = self._analysis_results.get("inventory_analysis", {})

        charts = self.dashboard_builder.build_comprehensive_dashboard(
            self._sales_clean, self._customer_clean, self._inventory_clean,
            {"sales_analysis": sales_analysis, "customer_analysis": customer_analysis, "inventory_analysis": inventory_analysis},
        )
        return {"charts_generated": len(charts), "chart_paths": charts}

    def _ensure_data(self, source: str) -> None:
        if source == "sales" and self._sales_data is None:
            self.collect_data("sales")
            self.clean_data("sales")
        elif source == "inventory" and self._inventory_data is None:
            self.collect_data("inventory")
            self.clean_data("inventory")
        elif source == "customer" and self._customer_data is None:
            self.collect_data("customer")
            self.clean_data("customer")

    def get_data_status(self) -> Dict[str, Any]:
        return {
            "sales": {
                "collected": self._sales_data is not None,
                "cleaned": self._sales_clean is not None,
                "records": len(self._sales_data) if self._sales_data is not None else 0,
            },
            "inventory": {
                "collected": self._inventory_data is not None,
                "cleaned": self._inventory_clean is not None,
                "records": len(self._inventory_data) if self._inventory_data is not None else 0,
            },
            "customer": {
                "collected": self._customer_data is not None,
                "cleaned": self._customer_clean is not None,
                "records": len(self._customer_data) if self._customer_data is not None else 0,
            },
        }
