from datetime import datetime
from typing import Any, Dict, Optional


class ReportTemplateManager:
    TEMPLATES = {
        "comprehensive": {
            "title": "零售数据分析综合报告",
            "sections": [
                {"id": "overview", "title": "数据概览", "type": "data_overview"},
                {"id": "sales_trend", "title": "销售趋势分析", "type": "sales_trend"},
                {"id": "sales_forecast", "title": "销售预测", "type": "sales_forecast"},
                {"id": "customer_rfm", "title": "客户RFM分析", "type": "customer_rfm"},
                {"id": "customer_churn", "title": "客户流失分析", "type": "customer_churn"},
                {"id": "customer_segment", "title": "客户分群分析", "type": "customer_segment"},
                {"id": "inventory_health", "title": "库存健康度", "type": "inventory_health"},
                {"id": "inventory_abc", "title": "库存ABC分析", "type": "inventory_abc"},
                {"id": "inventory_reorder", "title": "补货建议", "type": "inventory_reorder"},
                {"id": "marketing_effect", "title": "营销效果分析", "type": "marketing_effect"},
            ],
        },
        "sales": {
            "title": "销售分析报告",
            "sections": [
                {"id": "overview", "title": "数据概览", "type": "data_overview"},
                {"id": "sales_trend", "title": "销售趋势", "type": "sales_trend"},
                {"id": "sales_forecast", "title": "销售预测", "type": "sales_forecast"},
                {"id": "marketing_effect", "title": "营销效果", "type": "marketing_effect"},
            ],
        },
        "customer": {
            "title": "客户分析报告",
            "sections": [
                {"id": "overview", "title": "数据概览", "type": "data_overview"},
                {"id": "customer_rfm", "title": "RFM分析", "type": "customer_rfm"},
                {"id": "customer_churn", "title": "流失分析", "type": "customer_churn"},
                {"id": "customer_segment", "title": "客户分群", "type": "customer_segment"},
            ],
        },
        "inventory": {
            "title": "库存分析报告",
            "sections": [
                {"id": "overview", "title": "数据概览", "type": "data_overview"},
                {"id": "inventory_health", "title": "库存健康度", "type": "inventory_health"},
                {"id": "inventory_abc", "title": "ABC分析", "type": "inventory_abc"},
                {"id": "inventory_reorder", "title": "补货建议", "type": "inventory_reorder"},
            ],
        },
    }

    def get_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        return self.TEMPLATES.get(template_name)

    def list_templates(self) -> list:
        return list(self.TEMPLATES.keys())

    def render_html(self, template_name: str, sections_data: Dict[str, Any], metadata: Optional[Dict] = None) -> str:
        template = self.get_template(template_name)
        if not template:
            template = self.TEMPLATES["comprehensive"]

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta = metadata or {}

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{template['title']}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; color: #333; background: #f5f7fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 12px; margin-bottom: 30px; }}
        .header h1 {{ font-size: 28px; margin-bottom: 10px; }}
        .header .meta {{ font-size: 14px; opacity: 0.9; }}
        .section {{ background: white; border-radius: 12px; padding: 30px; margin-bottom: 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
        .section h2 {{ font-size: 22px; color: #2c3e50; border-left: 4px solid #667eea; padding-left: 15px; margin-bottom: 20px; }}
        .section .content {{ font-size: 15px; line-height: 1.8; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
        .metric-card {{ background: #f8f9fa; border-radius: 8px; padding: 20px; text-align: center; }}
        .metric-card .value {{ font-size: 28px; font-weight: bold; color: #667eea; }}
        .metric-card .label {{ font-size: 13px; color: #7f8c8d; margin-top: 5px; }}
        .chart-container {{ margin: 20px 0; text-align: center; }}
        .chart-container img {{ max-width: 100%; border-radius: 8px; }}
        .highlight {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; border-radius: 4px; }}
        .warning {{ background: #f8d7da; border-left: 4px solid #dc3545; padding: 15px; margin: 15px 0; border-radius: 4px; }}
        .success {{ background: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 15px 0; border-radius: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background: #667eea; color: white; }}
        tr:hover {{ background: #f8f9fa; }}
        .footer {{ text-align: center; padding: 20px; color: #7f8c8d; font-size: 13px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{template['title']}</h1>
            <div class="meta">
                <span>生成时间：{now}</span>
                {''.join(f'<span> | {k}: {v}</span>' for k, v in meta.items())}
            </div>
        </div>
"""
        for section in template["sections"]:
            section_id = section["id"]
            section_data = sections_data.get(section_id, {})
            html += f"""
        <div class="section">
            <h2>{section['title']}</h2>
            <div class="content">
"""
            if "metrics" in section_data:
                html += '<div class="metric-grid">'
                for label, value in section_data["metrics"].items():
                    html += f'<div class="metric-card"><div class="value">{value}</div><div class="label">{label}</div></div>'
                html += "</div>"

            if "text" in section_data:
                html += f'<p>{section_data["text"]}</p>'

            if "highlight" in section_data:
                html += f'<div class="highlight">{section_data["highlight"]}</div>'

            if "warning" in section_data:
                html += f'<div class="warning">{section_data["warning"]}</div>'

            if "table" in section_data:
                table_data = section_data["table"]
                if table_data:
                    headers = list(table_data[0].keys())
                    html += "<table><thead><tr>"
                    for h in headers:
                        html += f"<th>{h}</th>"
                    html += "</tr></thead><tbody>"
                    for row in table_data[:20]:
                        html += "<tr>"
                        for h in headers:
                            html += f"<td>{row.get(h, '')}</td>"
                        html += "</tr>"
                    html += "</tbody></table>"

            if "chart_path" in section_data:
                html += f'<div class="chart-container"><img src="{section_data["chart_path"]}" alt="{section["title"]}"></div>'

            html += """
            </div>
        </div>
"""

        html += """
        <div class="footer">
            <p>本报告由AIDA智能数据分析系统自动生成</p>
        </div>
    </div>
</body>
</html>"""
        return html
