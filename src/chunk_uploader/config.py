"""Configuration management for chunk uploader."""
import os
import yaml
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str
    api_key: str
    collection_name: str
    timeout: float = 60.0
    prefer_grpc: bool = False


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str
    model_type: str = 'sentence'
    normalize: bool = True
    max_length: int = 2048


@dataclass
class UploadConfig:
    """Upload process configuration."""
    batch_size: int = 24
    subset_size: int = 96
    download_threads: int = 8
    vector_size: int = 2560
    score_threshold: float = 0.0
    skip_first_chunks: int = 0
    max_retries: int = 3
    retry_delay_base: float = 2.0
    gpu_memory_threshold: float = 0.85


@dataclass
class AppConfig:
    """Main application configuration."""
    database: DatabaseConfig
    embedding: EmbeddingConfig
    upload: UploadConfig
    chunks_folder: str


class ConfigLoader:
    """Load and validate configuration from YAML files."""
    
    @staticmethod
    def load_yaml(path: str) -> dict:
        """Load YAML configuration file."""
        # Resolve relative paths from current working directory
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        with open(path, "r") as f:
            return yaml.safe_load(f)
    
    @classmethod
    def load_config(
        cls, 
        config_path: str,
        chunks_folder: str,
        db_url: Optional[str] = None,
        db_api_key: Optional[str] = None,
        score_threshold: float = 0.0,
        skip_first_chunks: int = 0
    ) -> AppConfig:
        """Load configuration from YAML and environment."""
        config_data = cls.load_yaml(config_path)
        
        # Database config
        database = DatabaseConfig(
            url=db_url or config_data.get("database_url", ""),
            api_key=db_api_key or config_data.get("database_api_key", ""),
            collection_name=config_data["database"]["collection_name"]
        )
        
        # Embedding config
        embedding_data = config_data["embedding"]
        embedding = EmbeddingConfig(
            model_name=embedding_data["model_name"],
            model_type=embedding_data.get("type", "sentence"),
            normalize=embedding_data.get("normalize", True)
        )
        
        # Upload config
        upload_data = config_data["upload_params"]
        upload = UploadConfig(
            batch_size=min(upload_data.get("batch_size", 48), 24),  # Cap at 24 for stability
            subset_size=min(96, 96),  # Cap for memory management
            vector_size=upload_data.get("vector_size", 2560),
            score_threshold=score_threshold,
            skip_first_chunks=skip_first_chunks
        )
        
        return AppConfig(
            database=database,
            embedding=embedding,
            upload=upload,
            chunks_folder=chunks_folder
        )

