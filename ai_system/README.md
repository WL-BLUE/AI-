<div align="center">

# 🤖 AIDA

**AI-based Automated Data Analysis & Report Generation System**

基于AI的自动化数据分析与报告生成系统

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[功能特性](#-功能特性) • [系统架构](#-系统架构) • [快速开始](#-快速开始) • [API文档](#-api文档) • [配置说明](#-配置说明)

</div>

---

## 📋 项目简介

AIDA 是一套面向零售行业的智能数据分析平台，集成数据采集、AI清洗、机器学习分析、自然语言报告生成和数据可视化于一体。系统从原始数据输入到最终报告输出实现全流程自动化，帮助零售企业快速洞察销售趋势、客户行为和库存状况。

## ✨ 功能特性

### 📥 数据采集模块
- 多数据源对接：CSV / Excel / 数据库 / REST API
- 内置销售、库存、客户三大采集器
- APScheduler 定时调度，支持实时数据更新
- 数据源不可用时自动回退至样本数据

### 🧹 数据清洗模块
- **标准清洗**：去重、缺失值处理（auto/mean/median/mode/drop）、IQR/Z-score 异常值检测、数据类型修复、文本规范化
- **AI 清洗**：
  - 孤立森林（Isolation Forest）异常检测
  - GradientBoosting 智能缺失值填充
  - TF-IDF + 余弦相似度语义去重
- 数据质量验证器：完整性 / 唯一性 / 一致性 / 范围 / 格式检查

### 📊 数据分析模块
| 分析器 | 算法 | 输出 |
|--------|------|------|
| 销售预测 | 线性回归趋势 + GradientBoosting 预测 | 趋势方向、增长率、季节性、未来N天预测值及置信区间 |
| 客户分析 | RFM 分层 + KMeans 聚类 + 流失概率 | 客户分层、流失风险、行为画像 |
| 库存优化 | ABC 分类 + 安全库存 + 补货模型 | 健康度、分类、补货建议、安全库存 |
| 回归分析 | Linear / Ridge / Lasso / RF / GBR | R²、RMSE、MAE、特征重要性 |
| 聚类分析 | KMeans / DBSCAN / 层次聚类 | 轮廓系数、簇统计、PCA降维坐标 |

### 📝 报告生成模块
- NLG 自然语言生成引擎，自动撰写中文分析报告
- 4 种报告模板：综合 / 销售 / 客户 / 库存
- HTML 格式输出，含指标卡片、数据表格、预警提示
- 响应式设计，支持移动端查看

### 📈 可视化展示模块
- 销售趋势图 / 预测图（含置信区间）
- 品类饼图 / 渠道柱状图
- 客户分群图 / RFM 散点图
- 相关性热力图 / 销售热力图 / 库存热力图 / 客户特征热力图
- 一键生成综合仪表盘

### 🔐 安全与权限
- JWT 令牌认证
- RBAC 角色权限控制（admin / analyst / viewer）
- bcrypt 密码加密
- 敏感数据脱敏

---

## 🏗 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AIDA System Architecture                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                       │
│  │  Sales   │   │Inventory │   │ Customer │    Data Sources        │
│  │  CSV/DB  │   │  CSV/DB  │   │  CSV/DB  │    (CSV/Excel/DB/API) │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘                       │
│       │              │              │                               │
│       ▼              ▼              ▼                               │
│  ┌─────────────────────────────────────────┐                       │
│  │          Data Collection Layer           │                       │
│  │   (BaseCollector + Scheduler + Fallback) │                       │
│  └─────────────────┬───────────────────────┘                       │
│                    │                                                 │
│                    ▼                                                 │
│  ┌─────────────────────────────────────────┐                       │
│  │           Data Cleaning Layer            │                       │
│  │  ┌──────────┐  ┌──────────┐  ┌────────┐ │                       │
│  │  │ Standard │  │AI-Powered│  │Validator│ │                       │
│  │  │ Cleaner  │  │ Cleaner  │  │         │ │                       │
│  │  └──────────┘  └──────────┘  └────────┘ │                       │
│  └─────────────────┬───────────────────────┘                       │
│                    │                                                 │
│                    ▼                                                 │
│  ┌─────────────────────────────────────────┐                       │
│  │          Data Analysis Layer             │                       │
│  │  ┌────────┐ ┌────────┐ ┌─────────────┐  │                       │
│  │  │Regression│ │Cluster│ │SalesPredict │  │                       │
│  │  └────────┘ └────────┘ └─────────────┘  │                       │
│  │  ┌────────────┐ ┌───────────────────┐   │                       │
│  │  │  Customer   │ │    Inventory      │   │                       │
│  │  │  Analyzer   │ │   Optimizer       │   │                       │
│  │  └────────────┘ └───────────────────┘   │                       │
│  └────────┬──────────────┬─────────────────┘                       │
│           │              │                                          │
│           ▼              ▼                                          │
│  ┌──────────────┐ ┌──────────────┐                                 │
│  │    Report     │ │ Visualization│                                 │
│  │  Generation   │ │   Dashboard  │                                 │
│  │  (NLG Engine) │ │  (Charts +   │                                 │
│  │  + Templates  │ │   Heatmaps)  │                                 │
│  └──────┬───────┘ └──────┬───────┘                                 │
│         │                │                                          │
│         ▼                ▼                                          │
│  ┌─────────────────────────────────────────┐                       │
│  │              REST API Layer              │                       │
│  │    (FastAPI + JWT Auth + RBAC)           │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
│  ┌─────────────────────────────────────────┐                       │
│  │           Core Infrastructure            │                       │
│  │  Config │ Logger │ Security │ Exceptions │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 数据流水线

```
原始数据 → 采集 → 清洗 → 分析 → 报告 + 可视化
   │        │      │      │       │
   │        │      │      │       └─ HTML报告 + PNG图表
   │        │      │      └─ 销售预测 / 客户分群 / 库存优化
   │        │      └─ 去重 / 缺失值 / 异常值 / AI清洗
   │        └─ CSV / Excel / DB / API
   └─ 销售 / 库存 / 客户
```

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- pip

### 安装

```bash
# 克隆项目
git clone https://github.com/<your-username>/aida.git
cd aida

# 安装依赖
pip install -r requirements.txt
```

### 运行

```bash
# 方式1：运行完整分析流水线（使用内置样本数据）
python main.py analyze

# 方式2：使用AI清洗模式
python main.py analyze --use-ai --forecast-days 30

# 方式3：启动API服务
python main.py api --host 0.0.0.0 --port 8000

# 方式4：在代码中使用
python -c "
from aida.pipeline import AnalysisPipeline
p = AnalysisPipeline()
p.collect_data('all')
p.clean_data('all')
result = p.run_full_analysis()
report = p.generate_report()
print(f'报告: {report[\"filepath\"]}')
"
```

### 运行测试

```bash
python -m pytest tests/ -v
```

---

## 📡 API文档

启动API服务后访问 `http://localhost:8000/docs` 查看交互式文档。

### 核心接口

| 方法 | 路径 | 说明 | 权限 |
|------|------|------|------|
| `POST` | `/api/auth/login` | 用户登录获取令牌 | 公开 |
| `GET` | `/api/health` | 健康检查 | 公开 |
| `POST` | `/api/data/collect/{source}` | 采集指定数据源 | read |
| `POST` | `/api/data/clean/{source}` | 清洗指定数据源 | write |
| `POST` | `/api/analysis/sales` | 销售分析 | run_analysis |
| `POST` | `/api/analysis/customer` | 客户分析 | run_analysis |
| `POST` | `/api/analysis/inventory` | 库存分析 | run_analysis |
| `POST` | `/api/analysis/full` | 全量分析 | run_analysis |
| `POST` | `/api/report/generate` | 生成报告 | generate_report |
| `POST` | `/api/visualization/dashboard` | 生成可视化仪表盘 | generate_report |
| `GET` | `/api/status` | 系统状态 | read |

### 默认用户

| 用户名 | 密码 | 角色 | 权限 |
|--------|------|------|------|
| `admin` | `admin123` | admin | 全部权限 |
| `analyst` | `analyst123` | analyst | 读写 + 分析 + 报告 |
| `viewer` | `viewer123` | viewer | 只读 |

### 使用示例

```bash
# 1. 登录
TOKEN=$(curl -s "http://localhost:8000/api/auth/login?username=admin&password=admin123" | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

# 2. 采集数据
curl -X POST -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/data/collect/all

# 3. 运行全量分析
curl -X POST -H "Authorization: Bearer $TOKEN" "http://localhost:8000/api/analysis/full?forecast_days=30"

# 4. 生成报告
curl -X POST -H "Authorization: Bearer $TOKEN" "http://localhost:8000/api/report/generate?template_name=comprehensive"

# 5. 生成可视化
curl -X POST -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/visualization/dashboard
```

---

## ⚙ 配置说明

配置文件：`config.yaml`，主要配置项：

```yaml
# 数据源配置 - 指定数据源类型和路径
data_sources:
  sales:
    type: "csv"           # csv / excel / database / api
    path: "./data/sample/sales_data.csv"
    refresh_interval_minutes: 60

# 数据清洗配置
data_cleaning:
  missing_value_strategy: "auto"   # auto / drop / mean / median / mode / zero
  outlier_method: "iqr"            # iqr / zscore
  outlier_threshold: 1.5

# 分析配置
analysis:
  sales_prediction:
    forecast_days: 30
    confidence_level: 0.95
  customer_clustering:
    algorithm: "kmeans"            # kmeans / dbscan / agglomerative
    max_clusters: 10

# 安全配置
security:
  secret_key: "change-this-in-production"   # 生产环境务必修改
  access_token_expire_minutes: 30
```

---

## 📁 项目结构

```
aida/
├── config.yaml                          # 全局配置
├── requirements.txt                     # 依赖清单
├── setup.py                             # 安装配置
├── main.py                              # CLI入口
├── aida/
│   ├── __init__.py
│   ├── pipeline.py                      # 核心流水线
│   ├── core/                            # 核心框架
│   │   ├── config.py                    #   配置管理（单例模式）
│   │   ├── logger.py                    #   日志系统（loguru）
│   │   ├── security.py                  #   安全与权限（JWT + RBAC）
│   │   └── exceptions.py               #   异常体系
│   ├── data_collection/                 # 数据采集
│   │   ├── base.py                      #   采集器基类
│   │   ├── sales_collector.py           #   销售采集器
│   │   ├── inventory_collector.py       #   库存采集器
│   │   ├── customer_collector.py        #   客户采集器
│   │   └── scheduler.py                #   定时调度器
│   ├── data_cleaning/                   # 数据清洗
│   │   ├── cleaner.py                   #   标准清洗器
│   │   ├── validators.py               #   数据验证器
│   │   └── ai_cleaner.py               #   AI清洗器
│   ├── data_analysis/                   # 数据分析
│   │   ├── base.py                      #   分析器基类
│   │   ├── regression.py               #   回归分析
│   │   ├── clustering.py               #   聚类分析
│   │   ├── sales_predictor.py           #   销售预测
│   │   ├── customer_analyzer.py         #   客户分析
│   │   └── inventory_optimizer.py       #   库存优化
│   ├── report_generation/               # 报告生成
│   │   ├── generator.py                #   报告生成器
│   │   ├── nlg_engine.py               #   NLG引擎
│   │   └── templates.py                #   报告模板
│   ├── visualization/                   # 可视化
│   │   ├── chart_builder.py            #   图表构建
│   │   ├── heatmap.py                  #   热力图
│   │   └── dashboard.py                #   仪表盘
│   └── api/                             # API服务
│       └── app.py                       #   FastAPI应用
├── tests/                               # 单元测试
│   ├── test_data_collection.py
│   ├── test_data_cleaning.py
│   ├── test_data_analysis.py
│   ├── test_report_generation.py
│   └── test_visualization.py
└── data/                                # 数据目录
    └── reports/                         # 报告输出
        ├── *.html                       #   HTML报告
        └── charts/                      #   图表文件
```

---

## 🔧 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.9+ |
| Web框架 | FastAPI + Uvicorn |
| 数据处理 | Pandas, NumPy |
| 机器学习 | scikit-learn, scipy, statsmodels |
| 可视化 | Matplotlib, Seaborn, Plotly |
| 认证 | python-jose (JWT), passlib (bcrypt) |
| 调度 | APScheduler |
| 日志 | Loguru |
| 配置 | PyYAML |
| 模板 | Jinja2 |

---

## 📊 分析输出示例

### 销售预测报告
```
销售趋势呈上升态势，日均增长率2.50%。销售存在明显季节性特征，
周六为每周销售高峰，12月为年度销售旺季。基于历史数据的机器学习
模型（R²=0.823）预测，未来30天总销售额约为1,250,000元，
日均销售额约41,667元。预测置信度为95%。
```

### 客户分析报告
```
对500名客户进行RFM分析，客户分层结果如下：中等价值客户175人
（35.0%），低价值客户125人（25.0%），重要价值客户75人（15.0%），
新客户75人（15.0%），流失预警客户50人（10.0%）。平均消费金额
4,997元，平均购买频次50.0次。客户平均流失概率为28.5%，
其中50名客户（10.0%）属于高流失风险，建议关注并制定预防性挽留策略。
```

### 库存优化报告
```
当前库存共200个商品，总价值15,000,000元，库存周转率为8.50次/年。
库存预警：5个缺货、15个低库存商品需关注。A类商品40个（20.0%），
贡献80.0%的库存价值。当前有25个商品（12.5%）需要补货，
补货压力较小。
```

---

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

<div align="center">

**AIDA** - 让数据分析更智能 🚀

</div>
