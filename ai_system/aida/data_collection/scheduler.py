from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from aida.core.config import settings
from aida.core.logger import get_logger
from aida.data_collection.base import BaseCollector

logger = get_logger("scheduler")


class DataScheduler:
    def __init__(self):
        self._scheduler = BackgroundScheduler()
        self._collectors: Dict[str, BaseCollector] = {}
        self._callbacks: Dict[str, List[Callable]] = {}

    def register_collector(self, name: str, collector: BaseCollector, callback: Optional[Callable] = None) -> None:
        self._collectors[name] = collector
        if callback:
            if name not in self._callbacks:
                self._callbacks[name] = []
            self._callbacks[name].append(callback)
        logger.info(f"注册数据采集器: {name}")

    def _execute_collection(self, name: str) -> None:
        collector = self._collectors.get(name)
        if not collector:
            logger.warning(f"未找到采集器: {name}")
            return
        try:
            data = collector.collect()
            logger.info(f"定时采集完成: {name}, 记录数: {len(data)}")
            callbacks = self._callbacks.get(name, [])
            for cb in callbacks:
                try:
                    cb(name, data)
                except Exception as e:
                    logger.error(f"回调执行失败 [{name}]: {e}")
        except Exception as e:
            logger.error(f"定时采集失败 [{name}]: {e}")

    def start(self) -> None:
        for name, collector in self._collectors.items():
            interval = collector.config.get("refresh_interval_minutes", 60)
            self._scheduler.add_job(
                self._execute_collection,
                trigger=IntervalTrigger(minutes=interval),
                id=f"collect_{name}",
                name=f"采集 {name}",
                kwargs={"name": name},
                replace_existing=True,
            )
            logger.info(f"调度任务已添加: {name}, 间隔: {interval} 分钟")

        self._scheduler.start()
        logger.info("数据采集调度器已启动")

    def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
        logger.info("数据采集调度器已停止")

    def trigger_now(self, name: str) -> Optional[Any]:
        collector = self._collectors.get(name)
        if not collector:
            logger.warning(f"未找到采集器: {name}")
            return None
        return collector.collect()

    def get_status(self) -> Dict[str, Any]:
        jobs = self._scheduler.get_jobs()
        return {
            "running": self._scheduler.running,
            "collectors": list(self._collectors.keys()),
            "scheduled_jobs": [
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": str(job.next_run_time) if job.next_run_time else None,
                }
                for job in jobs
            ],
        }
