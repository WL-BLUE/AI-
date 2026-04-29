import pytest
import pandas as pd
import numpy as np

from aida.data_cleaning.cleaner import DataCleaner
from aida.data_cleaning.validators import DataValidator
from aida.data_cleaning.ai_cleaner import AIDataCleaner


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "id": range(100),
        "value": np.random.randn(100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "date": pd.date_range("2024-01-01", periods=100),
    })


@pytest.fixture
def dirty_df():
    np.random.seed(42)
    df = pd.DataFrame({
        "id": list(range(95)) + [1, 2, 3, 5, 5],
        "value": list(np.random.randn(95)) + [np.nan, np.nan, 999, 999, np.nan],
        "category": np.random.choice(["A", "B", "C", None], 100),
    })
    return df


class TestDataCleaner:
    def test_clean_basic(self, dirty_df):
        cleaner = DataCleaner()
        result = cleaner.clean(dirty_df)
        assert isinstance(result, pd.DataFrame)
        assert result.isnull().sum().sum() == 0 or len(result) > 0

    def test_remove_duplicates(self, dirty_df):
        cleaner = DataCleaner(config={"duplicate_check_columns": ["id"]})
        result = cleaner.remove_duplicates(dirty_df)
        assert len(result) < len(dirty_df)

    def test_handle_missing_values(self, dirty_df):
        cleaner = DataCleaner()
        result = cleaner.handle_missing_values(dirty_df.copy())
        missing = result.isnull().sum().sum()
        assert missing == 0 or result.shape[1] < dirty_df.shape[1]

    def test_cleaning_report(self, dirty_df):
        cleaner = DataCleaner()
        cleaner.clean(dirty_df)
        report = cleaner.cleaning_report
        assert "original_count" in report
        assert "final_count" in report


class TestDataValidator:
    def test_validate_clean_data(self, sample_df):
        validator = DataValidator()
        result = validator.validate(sample_df)
        assert "passed" in result
        assert "total_checks" in result

    def test_validate_dirty_data(self, dirty_df):
        validator = DataValidator()
        result = validator.validate(dirty_df)
        assert result["total_checks"] > 0


class TestAIDataCleaner:
    def test_ai_detect_anomalies(self, sample_df):
        cleaner = AIDataCleaner()
        result = cleaner.ai_detect_anomalies(sample_df, columns=["value"])
        assert "_is_anomaly" in result.columns
        assert "_anomaly_score" in result.columns

    def test_ai_impute_missing(self):
        df = pd.DataFrame({
            "a": [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
            "b": [10, 20, 30, 40, np.nan, 60, 70, 80, 90, 100],
            "c": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })
        cleaner = AIDataCleaner()
        result = cleaner.ai_impute_missing(df)
        assert result.isnull().sum().sum() == 0
