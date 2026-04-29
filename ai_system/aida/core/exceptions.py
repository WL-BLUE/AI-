class AIDAError(Exception):
    def __init__(self, message: str = "", code: str = "UNKNOWN_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class DataCollectionError(AIDAError):
    def __init__(self, message: str = "数据采集失败"):
        super().__init__(message, "DATA_COLLECTION_ERROR")


class DataCleaningError(AIDAError):
    def __init__(self, message: str = "数据清洗失败"):
        super().__init__(message, "DATA_CLEANING_ERROR")


class DataAnalysisError(AIDAError):
    def __init__(self, message: str = "数据分析失败"):
        super().__init__(message, "DATA_ANALYSIS_ERROR")


class ReportGenerationError(AIDAError):
    def __init__(self, message: str = "报告生成失败"):
        super().__init__(message, "REPORT_GENERATION_ERROR")


class VisualizationError(AIDAError):
    def __init__(self, message: str = "可视化生成失败"):
        super().__init__(message, "VISUALIZATION_ERROR")


class ConfigError(AIDAError):
    def __init__(self, message: str = "配置错误"):
        super().__init__(message, "CONFIG_ERROR")


class AuthenticationError(AIDAError):
    def __init__(self, message: str = "认证失败"):
        super().__init__(message, "AUTHENTICATION_ERROR")


class AuthorizationError(AIDAError):
    def __init__(self, message: str = "权限不足"):
        super().__init__(message, "AUTHORIZATION_ERROR")


class DataSourceNotFoundError(AIDAError):
    def __init__(self, source: str = ""):
        super().__init__(f"数据源未找到: {source}", "DATA_SOURCE_NOT_FOUND")


class ModelTrainingError(AIDAError):
    def __init__(self, message: str = "模型训练失败"):
        super().__init__(message, "MODEL_TRAINING_ERROR")


class InvalidDataError(AIDAError):
    def __init__(self, message: str = "无效数据"):
        super().__init__(message, "INVALID_DATA_ERROR")
