from typing import Any, Dict, List, Optional

import pandas as pd

from aida.core.logger import get_logger

logger = get_logger("validators")


class DataValidator:
    def __init__(self):
        self._validation_results: List[Dict[str, Any]] = []

    def validate(self, df: pd.DataFrame, rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._validation_results = []

        self._check_completeness(df)
        self._check_uniqueness(df)
        self._check_consistency(df)
        self._check_range(df)
        self._check_format(df)

        if rules:
            self._apply_custom_rules(df, rules)

        passed = all(r["passed"] for r in self._validation_results)
        result = {
            "passed": passed,
            "total_checks": len(self._validation_results),
            "passed_checks": sum(1 for r in self._validation_results if r["passed"]),
            "failed_checks": sum(1 for r in self._validation_results if not r["passed"]),
            "details": self._validation_results,
        }
        logger.info(f"数据验证完成: {'通过' if passed else '未通过'}, 检查项: {result['total_checks']}")
        return result

    def _add_result(self, check_name: str, passed: bool, message: str, details: Any = None) -> None:
        self._validation_results.append({
            "check": check_name,
            "passed": passed,
            "message": message,
            "details": details,
        })

    def _check_completeness(self, df: pd.DataFrame) -> None:
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 1

        self._add_result(
            "数据完整性",
            completeness >= 0.95,
            f"数据完整率: {completeness:.2%}",
            {"completeness_rate": round(completeness, 4), "missing_cells": int(missing_cells)},
        )

        for col in df.columns:
            null_pct = df[col].isnull().sum() / len(df)
            if null_pct > 0.1:
                self._add_result(
                    f"列完整性 - {col}",
                    null_pct <= 0.3,
                    f"列 '{col}' 缺失率: {null_pct:.2%}",
                    {"column": col, "null_percentage": round(null_pct, 4)},
                )

    def _check_uniqueness(self, df: pd.DataFrame) -> None:
        dup_count = df.duplicated().sum()
        dup_pct = dup_count / len(df) if len(df) > 0 else 0

        self._add_result(
            "数据唯一性",
            dup_pct < 0.05,
            f"重复记录率: {dup_pct:.2%} ({dup_count} 条)",
            {"duplicate_count": int(dup_count), "duplicate_rate": round(dup_pct, 4)},
        )

    def _check_consistency(self, df: pd.DataFrame) -> None:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if (df[col] < 0).any():
                neg_count = (df[col] < 0).sum()
                if "price" in col.lower() or "amount" in col.lower() or "quantity" in col.lower():
                    self._add_result(
                        f"数据一致性 - {col}",
                        False,
                        f"列 '{col}' 包含 {neg_count} 个负值",
                        {"column": col, "negative_count": int(neg_count)},
                    )

    def _check_range(self, df: pd.DataFrame) -> None:
        import numpy as np

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outlier_count = ((df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)).sum()
                outlier_pct = outlier_count / len(df)
                if outlier_pct > 0.05:
                    self._add_result(
                        f"数据范围 - {col}",
                        False,
                        f"列 '{col}' 极端异常值比例: {outlier_pct:.2%}",
                        {"column": col, "extreme_outlier_pct": round(outlier_pct, 4)},
                    )

    def _check_format(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            if "email" in col.lower():
                valid_email = df[col].astype(str).str.match(r"^[\w.-]+@[\w.-]+\.\w+$").sum()
                total = len(df)
                if valid_email < total * 0.9:
                    self._add_result(
                        f"数据格式 - {col}",
                        False,
                        f"列 '{col}' 邮箱格式有效率: {valid_email / total:.2%}",
                    )
            elif "phone" in col.lower():
                valid_phone = df[col].astype(str).str.match(r"^\d{11}$").sum()
                total = len(df)
                if valid_phone < total * 0.9:
                    self._add_result(
                        f"数据格式 - {col}",
                        False,
                        f"列 '{col}' 手机号格式有效率: {valid_phone / total:.2%}",
                    )

    def _apply_custom_rules(self, df: pd.DataFrame, rules: Dict[str, Any]) -> None:
        for rule_name, rule in rules.items():
            rule_type = rule.get("type")
            col = rule.get("column")
            if not col or col not in df.columns:
                continue

            if rule_type == "not_null":
                null_count = df[col].isnull().sum()
                self._add_result(
                    f"自定义规则 - {rule_name}",
                    null_count == 0,
                    f"列 '{col}' 非空检查: {null_count} 个空值",
                )
            elif rule_type == "range":
                min_val = rule.get("min")
                max_val = rule.get("max")
                in_range = ((df[col] >= min_val) & (df[col] <= max_val)).sum()
                self._add_result(
                    f"自定义规则 - {rule_name}",
                    in_range == len(df),
                    f"列 '{col}' 范围检查 [{min_val}, {max_val}]: {in_range}/{len(df)} 在范围内",
                )
            elif rule_type == "unique":
                dup_count = df[col].duplicated().sum()
                self._add_result(
                    f"自定义规则 - {rule_name}",
                    dup_count == 0,
                    f"列 '{col}' 唯一性检查: {dup_count} 个重复值",
                )
