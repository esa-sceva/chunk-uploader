"""Main entry point for chunk uploader."""
import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chunk_uploader.config import ConfigLoader
from chunk_uploader.uploader import ChunkUploader


def main():
    """Main function."""
    # Configuration
    config_path = os.getenv("CONFIG_PATH", "config/config_qwen.yaml")
    chunks_folder = os.getenv("CHUNKS_FOLDER")
    s3_chunks_path = os.getenv("S3_CHUNKS_PATH")
    score_threshold = float(os.getenv("SCORE_THRESHOLD", "-11"))
    skip_first_chunks = int(os.getenv("SKIP_FIRST_CHUNKS", "0"))
    
    # Database credentials (preferably from environment variables)
    db_url = os.getenv("QDRANT_URL")
    db_api_key = os.getenv("QDRANT_API_KEY")
    
    # Validate inputs
    if not db_url or not db_api_key:
        print("❌ Error: QDRANT_URL and QDRANT_API_KEY must be set")
        print("   export QDRANT_URL='https://your-instance:6333'")
        print("   export QDRANT_API_KEY='your-api-key'")
        sys.exit(1)
    
    if not chunks_folder and not s3_chunks_path:
        print("❌ Error: Either CHUNKS_FOLDER or S3_CHUNKS_PATH must be set")
        print("   export CHUNKS_FOLDER='/path/to/json/files'")
        print("   OR")
        print("   export S3_CHUNKS_PATH='s3://bucket/path/'")
        sys.exit(1)
    
    # Load configuration
    config = ConfigLoader.load_config(
        config_path=config_path,
        chunks_folder=chunks_folder or "/tmp/dummy",  # Dummy if using S3
        db_url=db_url,
        db_api_key=db_api_key,
        score_threshold=score_threshold,
        skip_first_chunks=skip_first_chunks
    )
    
    print("🚀 Chunk Uploader")
    print("=" * 50)
    print(f"🤖 Model: {config.embedding.model_name}")
    print(f"   Type: {config.embedding.model_type}")
    print(f"📊 Score threshold: {config.upload.score_threshold}")
    print(f"⏭️ Skip first chunks: {config.upload.skip_first_chunks}")
    print(f"📦 Batch size: {config.upload.batch_size}")
    print(f"📦 Subset size: {config.upload.subset_size}")
    
    if s3_chunks_path:
        print(f"📂 S3 path: {s3_chunks_path}")
    else:
        print(f"📂 Chunks folder: {chunks_folder}")
    print()
    
    # Create uploader and run
    uploader = ChunkUploader(config)
    
    if s3_chunks_path:
        uploader.upload_from_s3(s3_chunks_path)
    else:
        uploader.upload_all()


if __name__ == "__main__":
    main()

