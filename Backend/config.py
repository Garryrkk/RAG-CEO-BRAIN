

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Phase3 Memory Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "memory_platform"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    @property
    def POSTGRES_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def POSTGRES_SYNC_URL(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_DOCUMENTS: str = "documents"
    QDRANT_COLLECTION_PERSONS: str = "persons"
    QDRANT_COLLECTION_COMPANIES: str = "companies"
    QDRANT_COLLECTION_PROJECTS: str = "projects"
    QDRANT_COLLECTION_EVENTS: str = "events"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    CELERY_MAX_RETRIES: int = 3
    CELERY_RETRY_BACKOFF: int = 60

    # Ollama / AI
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3:latest"
    EMBEDDING_MODEL: str = "bge-m3"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32

    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    MINIO_BUCKET_DOCUMENTS: str = "documents"
    MINIO_BUCKET_CHUNKS: str = "chunks"

    # Entity Resolution thresholds
    ENTITY_RESOLUTION_NAME_THRESHOLD: float = 0.85
    ENTITY_RESOLUTION_COMPANY_THRESHOLD: float = 0.80
    ENTITY_RESOLUTION_PROJECT_THRESHOLD: float = 0.80
    ENTITY_RESOLUTION_CONTEXT_WEIGHT: float = 0.3

    # Chunking
    CHUNK_SIZE_EMAILS: int = 512
    CHUNK_SIZE_DOCUMENTS: int = 800
    CHUNK_SIZE_MEETING_NOTES: int = 600
    CHUNK_SIZE_CONTRACTS: int = 1000
    CHUNK_OVERLAP: int = 100

    # Prometheus
    METRICS_PORT: int = 8001
    METRICS_ENABLED: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
