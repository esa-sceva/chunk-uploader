"""Chunk Uploader - A modular system for uploading document chunks to vector databases.

This package provides a production-ready solution for:
- Downloading document chunks from S3
- Generating embeddings using various models
- Uploading vectors to Qdrant database
- Tracking statistics and managing GPU memory

Usage:
    from chunk_uploader.config import ConfigLoader
    from chunk_uploader.uploader import ChunkUploader
    
    config = ConfigLoader.load_config("config.yaml", chunks_folder="/data")
    uploader = ChunkUploader(config)
    uploader.upload_all()
"""

__version__ = "2.0.0"
__author__ = "Your Team"

from .config import AppConfig, ConfigLoader
from .uploader import ChunkUploader
from .embeddings import EmbeddingModelFactory
from .qdrant_uploader import QdrantUploader
from .chunk_parser import ChunkParser
from .s3_downloader import S3Downloader
from .s3_handler import S3ChunkHandler
from .stats_tracker import StatsTracker
from .gpu_manager import GPUManager

__all__ = [
    "AppConfig",
    "ConfigLoader",
    "ChunkUploader",
    "EmbeddingModelFactory",
    "QdrantUploader",
    "ChunkParser",
    "S3Downloader",
    "S3ChunkHandler",
    "StatsTracker",
    "GPUManager",
]

