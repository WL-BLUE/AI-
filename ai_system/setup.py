from setuptools import setup, find_packages

setup(
    name="aida",
    version="1.0.0",
    description="AI-based Automated Data Analysis and Report Generation System",
    author="AIDA Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "sqlalchemy>=2.0.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "apscheduler>=3.10.0",
        "loguru>=0.7.0",
        "openpyxl>=3.1.0",
        "jinja2>=3.1.0",
        "scipy>=1.11.0",
        "statsmodels>=0.14.0",
        "httpx>=0.24.0",
        "aiofiles>=23.1.0",
    ],
    entry_points={
        "console_scripts": [
            "aida=main:main",
        ],
    },
)
