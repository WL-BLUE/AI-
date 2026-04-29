from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from aida.core.config import settings
from aida.core.exceptions import AIDAError, AuthenticationError, AuthorizationError
from aida.core.logger import get_logger
from aida.core.security import (
    authenticate_user,
    check_permission,
    create_access_token,
    decode_access_token,
)

logger = get_logger("api")

app = FastAPI(
    title=settings.get("app.name", "AIDA"),
    version=settings.get("app.version", "1.0.0"),
    description="AI-based Automated Data Analysis and Report Generation System",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from aida.pipeline import AnalysisPipeline
        _pipeline = AnalysisPipeline()
    return _pipeline


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未提供认证令牌")
    try:
        payload = decode_access_token(credentials.credentials)
        return payload
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@app.post("/api/auth/login")
async def login(username: str = Query(...), password: str = Query(...)):
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    return {"access_token": token, "token_type": "bearer", "role": user["role"]}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": settings.get("app.version", "1.0.0")}


@app.post("/api/data/collect/{source_name}")
async def collect_data(source_name: str, user: Dict = Depends(get_current_user)):
    try:
        check_permission(user.get("sub", ""), "read")
        pipeline = get_pipeline()
        result = pipeline.collect_data(source_name)
        return {"status": "success", "data": result}
    except AuthorizationError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except AIDAError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/data/clean/{source_name}")
async def clean_data(source_name: str, use_ai: bool = Query(False), user: Dict = Depends(get_current_user)):
    try:
        check_permission(user.get("sub", ""), "write")
        pipeline = get_pipeline()
        result = pipeline.clean_data(source_name, use_ai=use_ai)
        return {"status": "success", "data": result}
    except AuthorizationError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except AIDAError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/analysis/sales")
async def analyze_sales(forecast_days: int = Query(30, ge=1, le=365), user: Dict = Depends(get_current_user)):
    try:
        check_permission(user.get("sub", ""), "run_analysis")
        pipeline = get_pipeline()
        result = pipeline.analyze_sales(forecast_days=forecast_days)
        return {"status": "success", "data": result}
    except AuthorizationError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except AIDAError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/analysis/customer")
async def analyze_customer(user: Dict = Depends(get_current_user)):
    try:
        check_permission(user.get("sub", ""), "run_analysis")
        pipeline = get_pipeline()
        result = pipeline.analyze_customer()
        return {"status": "success", "data": result}
    except AuthorizationError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except AIDAError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/analysis/inventory")
async def analyze_inventory(user: Dict = Depends(get_current_user)):
    try:
        check_permission(user.get("sub", ""), "run_analysis")
        pipeline = get_pipeline()
        result = pipeline.analyze_inventory()
        return {"status": "success", "data": result}
    except AuthorizationError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except AIDAError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/analysis/full")
async def full_analysis(forecast_days: int = Query(30, ge=1, le=365), user: Dict = Depends(get_current_user)):
    try:
        check_permission(user.get("sub", ""), "run_analysis")
        pipeline = get_pipeline()
        result = pipeline.run_full_analysis(forecast_days=forecast_days)
        return {"status": "success", "data": result}
    except AuthorizationError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except AIDAError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/report/generate")
async def generate_report(
    template_name: str = Query("comprehensive"),
    user: Dict = Depends(get_current_user),
):
    try:
        check_permission(user.get("sub", ""), "generate_report")
        pipeline = get_pipeline()
        result = pipeline.generate_report(template_name=template_name)
        return {"status": "success", "data": result}
    except AuthorizationError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except AIDAError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/visualization/dashboard")
async def generate_dashboard(user: Dict = Depends(get_current_user)):
    try:
        check_permission(user.get("sub", ""), "generate_report")
        pipeline = get_pipeline()
        result = pipeline.generate_visualizations()
        return {"status": "success", "data": result}
    except AuthorizationError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except AIDAError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/status")
async def get_status(user: Dict = Depends(get_current_user)):
    pipeline = get_pipeline()
    return {
        "status": "running",
        "data_sources": pipeline.get_data_status(),
        "timestamp": datetime.now().isoformat(),
    }


@app.exception_handler(AIDAError)
async def aida_error_handler(request, exc: AIDAError):
    return {"error": {"code": exc.code, "message": exc.message}}
