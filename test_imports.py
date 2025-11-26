"""Test that all imports work correctly."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing imports...")

try:
    from chunk_uploader.config import ConfigLoader, AppConfig
    print("✅ Config imports OK")
except Exception as e:
    print(f"❌ Config imports failed: {e}")
    sys.exit(1)

try:
    from chunk_uploader.embeddings import EmbeddingModelFactory
    print("✅ Embeddings imports OK")
except Exception as e:
    print(f"❌ Embeddings imports failed: {e}")
    sys.exit(1)

try:
    from chunk_uploader.uploader import ChunkUploader
    print("✅ Uploader imports OK")
except Exception as e:
    print(f"❌ Uploader imports failed: {e}")
    sys.exit(1)

try:
    from chunk_uploader.s3_handler import S3ChunkHandler
    print("✅ S3 handler imports OK")
except Exception as e:
    print(f"❌ S3 handler imports failed: {e}")
    sys.exit(1)

try:
    from chunk_uploader import (
        ChunkUploader,
        ConfigLoader,
        EmbeddingModelFactory,
        S3ChunkHandler
    )
    print("✅ Package-level imports OK")
except Exception as e:
    print(f"❌ Package-level imports failed: {e}")
    sys.exit(1)

print("\n🎉 All imports successful!")
print("\nYou can now run:")
print("  python main.py")
print("\nMake sure to set environment variables:")
print("  export QDRANT_URL='...'")
print("  export QDRANT_API_KEY='...'")
print("  export S3_CHUNKS_PATH='s3://...' or CHUNKS_FOLDER='/path'")

