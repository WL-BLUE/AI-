import argparse
import sys
from typing import Optional

from aida.core.config import settings
from aida.core.logger import get_logger

logger = get_logger("main")


def run_api(host: Optional[str] = None, port: Optional[int] = None):
    import uvicorn

    api_host = host or settings.get("app.host", "0.0.0.0")
    api_port = port or settings.get("app.port", 8000)
    logger.info(f"启动AIDA API服务: {api_host}:{api_port}")
    uvicorn.run("aida.api.app:app", host=api_host, port=api_port, reload=settings.get("app.debug", False))


def run_analysis(template: str = "comprehensive", forecast_days: int = 30, use_ai: bool = False):
    from aida.pipeline import AnalysisPipeline

    logger.info("启动AIDA分析流水线")
    pipeline = AnalysisPipeline()

    logger.info("步骤1: 数据采集")
    collect_result = pipeline.collect_data("all")
    logger.info(f"数据采集完成: {collect_result}")

    logger.info("步骤2: 数据清洗")
    clean_result = pipeline.clean_data("all", use_ai=use_ai)
    logger.info(f"数据清洗完成: {clean_result}")

    logger.info("步骤3: 数据分析")
    analysis_result = pipeline.run_full_analysis(forecast_days=forecast_days)
    logger.info(f"数据分析完成")

    for key, value in analysis_result.items():
        if isinstance(value, dict) and "summary" in value:
            logger.info(f"  {key}: {value['summary']}")

    logger.info("步骤4: 生成可视化")
    try:
        viz_result = pipeline.generate_visualizations()
        logger.info(f"可视化生成完成: {viz_result['charts_generated']}个图表")
    except Exception as e:
        logger.warning(f"可视化生成失败: {e}")

    logger.info("步骤5: 生成报告")
    report_result = pipeline.generate_report(template_name=template)
    logger.info(f"报告已生成: {report_result.get('filepath', 'unknown')}")

    return {
        "collection": collect_result,
        "cleaning": clean_result,
        "analysis": analysis_result,
        "report": report_result,
    }


def main():
    parser = argparse.ArgumentParser(description="AIDA - AI Data Analysis System")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    api_parser = subparsers.add_parser("api", help="启动API服务")
    api_parser.add_argument("--host", type=str, help="服务地址")
    api_parser.add_argument("--port", type=int, help="服务端口")

    analyze_parser = subparsers.add_parser("analyze", help="运行完整分析")
    analyze_parser.add_argument("--template", type=str, default="comprehensive", choices=["comprehensive", "sales", "customer", "inventory"], help="报告模板")
    analyze_parser.add_argument("--forecast-days", type=int, default=30, help="预测天数")
    analyze_parser.add_argument("--use-ai", action="store_true", help="使用AI数据清洗")

    args = parser.parse_args()

    if args.command == "api":
        run_api(host=args.host, port=args.port)
    elif args.command == "analyze":
        run_analysis(template=args.template, forecast_days=args.forecast_days, use_ai=args.use_ai)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
