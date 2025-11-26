#!/usr/bin/env python3
"""Quick script to upload chunks to Qdrant collection 'try'."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chunk_uploader.config import ConfigLoader
from chunk_uploader.uploader import ChunkUploader


def main():
    """Upload chunks to 'try' collection."""
    
    # Check required environment variables
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    chunks_folder = os.getenv("CHUNKS_FOLDER")
    
    if not qdrant_url:
        print("❌ Error: QDRANT_URL environment variable not set")
        print("   Set it with: export QDRANT_URL='https://your-instance:6333'")
        sys.exit(1)
    
    if not qdrant_api_key:
        print("❌ Error: QDRANT_API_KEY environment variable not set")
        print("   Set it with: export QDRANT_API_KEY='your-api-key'")
        sys.exit(1)
    
    if not chunks_folder:
        print("❌ Error: CHUNKS_FOLDER environment variable not set")
        print("   Set it with: export CHUNKS_FOLDER='/path/to/json/files'")
        sys.exit(1)
    
    # Verify chunks folder exists
    if not os.path.isdir(chunks_folder):
        print(f"❌ Error: Chunks folder not found: {chunks_folder}")
        sys.exit(1)
    
    # Get optional parameters
    score_threshold = float(os.getenv("SCORE_THRESHOLD", "-11"))
    skip_first_chunks = int(os.getenv("SKIP_FIRST_CHUNKS", "0"))
    
    print("=" * 60)
    print("🚀 Upload to Qdrant Collection 'try'")
    print("=" * 60)
    print(f"📊 Qdrant URL: {qdrant_url}")
    print(f"📂 Chunks folder: {chunks_folder}")
    print(f"📈 Score threshold: {score_threshold}")
    print(f"⏭️  Skip first chunks: {skip_first_chunks}")
    print()
    
    # Count JSON files
    json_files = list(Path(chunks_folder).glob("*.json"))
    print(f"📁 Found {len(json_files)} JSON file(s) to process")
    print()
    
    # Confirm
    response = input("Continue with upload? [y/N]: ")
    if response.lower() != 'y':
        print("❌ Upload cancelled")
        sys.exit(0)
    
    print()
    print("=" * 60)
    print("Starting upload...")
    print("=" * 60)
    print()
    
    # Load configuration
    config = ConfigLoader.load_config(
        config_path="config/config_qwen.yaml",
        chunks_folder=chunks_folder,
        db_url=qdrant_url,
        db_api_key=qdrant_api_key,
        score_threshold=score_threshold,
        skip_first_chunks=skip_first_chunks
    )
    
    # Verify collection name is 'try'
    print(f"🎯 Target collection: {config.database.collection_name}")
    if config.database.collection_name != "try":
        print(f"⚠️  Warning: Collection name in config is '{config.database.collection_name}', not 'try'")
        print(f"   Update config/config_qwen.yaml if needed")
        print()
    
    print(f"🤖 Embedding model: {config.embedding.model_name}")
    print(f"📊 Vector dimensions: {config.upload.vector_size}")
    print(f"📦 Batch size: {config.upload.batch_size}")
    print(f"📦 Subset size: {config.upload.subset_size}")
    print()
    
    try:
        # Create uploader and run
        uploader = ChunkUploader(config)
        uploader.upload_all()
        
        print()
        print("=" * 60)
        print("✅ Upload Complete!")
        print("=" * 60)
        print(f"🎯 Collection: {config.database.collection_name}")
        print(f"📊 Check upload_stats_*.json for detailed statistics")
        
    except KeyboardInterrupt:
        print()
        print("⏹️  Upload interrupted by user")
        print("📊 Partial statistics saved in upload_stats_*.json")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"❌ Error during upload: {e}")
        print("📊 Check upload_stats_*.json for what was completed")
        sys.exit(1)


if __name__ == "__main__":
    main()

