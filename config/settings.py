"""
Configuration management for the AI Change Impact Analysis Platform.
Uses Pydantic Settings for type-safe configuration with environment variable support.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GitHubSettings(BaseSettings):
    """GitHub integration settings."""
    token: str = Field(default="", description="GitHub Personal Access Token")
    repo: str = Field(default="", description="Repository in format owner/repo")
    webhook_secret: str = Field(default="", description="Webhook secret for validation")
    runner_labels: str = Field(default="ubuntu-latest", description="GitHub Actions runner labels")
    artifact_retention_days: int = Field(default=7, description="Artifact retention in days")

    model_config = SettingsConfigDict(env_prefix="GITHUB_", extra="ignore")


class LLMSettings(BaseSettings):
    """LLM provider settings."""
    provider: str = Field(default="ollama", description="LLM provider: ollama, openai, anthropic, vllm")
    model: str = Field(default="qwen2.5-coder", description="Model name")
    base_url: str = Field(default="http://localhost:11434", description="Base URL for LLM API")
    api_key: str = Field(default="", description="API key for cloud providers")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="Generation temperature")
    max_tokens: int = Field(default=4096, gt=0, description="Maximum tokens in response")

    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")


class DatabaseSettings(BaseSettings):
    """Database settings."""
    url: str = Field(default="sqlite:///data/history.db", description="Database connection URL")

    model_config = SettingsConfigDict(env_prefix="DATABASE_", extra="ignore")


class MLSettings(BaseSettings):
    """Machine learning model settings."""
    model_path: str = Field(default="models/catboost_model.cbm", description="Path to CatBoost model")
    iterations: int = Field(default=1000, gt=0, description="Training iterations")
    learning_rate: float = Field(default=0.1, gt=0.0, le=1.0, description="Learning rate")
    depth: int = Field(default=6, gt=0, description="Tree depth")
    loss_function: str = Field(default="Logloss", description="Loss function")
    eval_metric: str = Field(default="AUC", description="Evaluation metric")
    shap_sample_size: int = Field(default=1000, gt=0, description="SHAP sample size")
    shap_max_display: int = Field(default=20, gt=0, description="SHAP max features to display")

    model_config = SettingsConfigDict(env_prefix="CATBOOST_", extra="ignore")


class AnalysisSettings(BaseSettings):
    """Code analysis settings."""
    max_repo_size_mb: int = Field(default=500, gt=0, description="Maximum repository size in MB")
    analyze_extensions: List[str] = Field(default=[".py"], description="File extensions to analyze")
    exclude_dirs: List[str] = Field(
        default=["__pycache__", ".git", "venv", "env", ".venv", ".env", "node_modules",
                 "dist", "build", ".pytest_cache", ".mypy_cache", "htmlcov", ".coverage"],
        description="Directories to exclude from analysis"
    )
    max_file_size_kb: int = Field(default=500, gt=0, description="Maximum file size to parse in KB")

    model_config = SettingsConfigDict(env_prefix="ANALYSIS_", extra="ignore")

    @field_validator("analyze_extensions", mode="before")
    @classmethod
    def parse_extensions(cls, v):
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v

    @field_validator("exclude_dirs", mode="before")
    @classmethod
    def parse_exclude_dirs(cls, v):
        if isinstance(v, str):
            return [d.strip() for d in v.split(",")]
        return v


class ImpactSettings(BaseSettings):
    """Impact analysis settings."""
    decay_factor: float = Field(default=0.7, ge=0.0, le=1.0, description="Impact propagation decay factor")
    min_impact_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum impact score threshold")
    max_traversal_depth: int = Field(default=10, gt=0, description="Maximum graph traversal depth")
    weight_import: float = Field(default=1.0, ge=0.0, description="Weight for import relationships")
    weight_function_call: float = Field(default=0.8, ge=0.0, description="Weight for function call relationships")
    weight_inheritance: float = Field(default=0.9, ge=0.0, description="Weight for inheritance relationships")
    weight_module_reference: float = Field(default=0.6, ge=0.0, description="Weight for module reference relationships")

    model_config = SettingsConfigDict(env_prefix="IMPACT_", extra="ignore")


class RepairSettings(BaseSettings):
    """Autonomous repair settings."""
    max_iterations: int = Field(default=3, gt=0, description="Maximum repair iterations")
    validation_timeout: int = Field(default=300, gt=0, description="Validation timeout in seconds")
    enabled: bool = Field(default=True, description="Enable autonomous repair")

    model_config = SettingsConfigDict(env_prefix="REPAIR_", extra="ignore")


class LoggingSettings(BaseSettings):
    """Logging settings."""
    level: str = Field(default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR")
    format: str = Field(default="json", description="Log format: json, text")
    file: str = Field(default="logs/analysis.log", description="Log file path")

    model_config = SettingsConfigDict(env_prefix="LOG_", extra="ignore")


class APISettings(BaseSettings):
    """API server settings."""
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, gt=0, lt=65536, description="API port")
    workers: int = Field(default=4, gt=0, description="Number of worker processes")

    model_config = SettingsConfigDict(env_prefix="API_", extra="ignore")


class Settings(BaseSettings):
    """Main settings class combining all configuration sections."""
    github: GitHubSettings = Field(default_factory=GitHubSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)
    impact: ImpactSettings = Field(default_factory=ImpactSettings)
    repair: RepairSettings = Field(default_factory=RepairSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    api: APISettings = Field(default_factory=APISettings)

    # Project paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "models")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Parse nested settings from flat env vars if needed
        self._parse_nested_settings()

    def _parse_nested_settings(self):
        """Parse nested settings from flat environment variables."""
        # This allows env vars like GITHUB_TOKEN to populate github.token
        pass


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global settings
    settings = Settings()
    return settings