"""Example usage of the refactored chunk uploader.

This file demonstrates various ways to use the refactored modules.
"""

import os
from config import ConfigLoader
from uploader import ChunkUploader
from embeddings import EmbeddingModelFactory
from chunk_parser import ChunkParser
from s3_downloader import S3Downloader
from gpu_manager import GPUManager


def example_basic_usage():
    """Basic usage - upload all chunks from a folder."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Load configuration
    config = ConfigLoader.load_config(
        config_path="config_qwen.yaml",
        chunks_folder="/workspace/chunks",
        db_url=os.getenv("QDRANT_URL"),
        db_api_key=os.getenv("QDRANT_API_KEY"),
        score_threshold=-11,
        skip_first_chunks=0
    )
    
    # Create uploader and run
    uploader = ChunkUploader(config)
    uploader.upload_all()


def example_single_file():
    """Upload a single JSON file."""
    print("=" * 60)
    print("Example 2: Single File Upload")
    print("=" * 60)
    
    config = ConfigLoader.load_config(
        config_path="config_qwen.yaml",
        chunks_folder="/workspace/chunks",
        db_url=os.getenv("QDRANT_URL"),
        db_api_key=os.getenv("QDRANT_API_KEY")
    )
    
    uploader = ChunkUploader(config)
    uploader.upload_file("/workspace/chunks/arxiv.json")


def example_custom_embedder():
    """Use a specific embedding model."""
    print("=" * 60)
    print("Example 3: Custom Embedding Model")
    print("=" * 60)
    
    # Create NASA embedder
    embedder, vector_size = EmbeddingModelFactory.create(
        model_name="nasa-impact/nasa-smd-ibm-st-v2",
        model_type="sentence",
        normalize=True
    )
    
    # Generate embeddings
    texts = [
        "This is a test document about space exploration.",
        "Machine learning models can process natural language.",
    ]
    
    embeddings = embedder.embed_documents(texts, batch_size=2)
    
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Vector dimensions: {len(embeddings[0])}")
    print(f"Expected dimensions: {vector_size}")


def example_parse_chunks():
    """Parse chunks from a JSON file without uploading."""
    print("=" * 60)
    print("Example 4: Parse Chunks Only")
    print("=" * 60)
    
    parser = ChunkParser(score_threshold=0.5, skip_first_chunks=100)
    
    chunks, stats = parser.parse_file("/workspace/chunks/arxiv.json")
    
    print(f"Total chunks: {stats['total']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Filtered: {stats['filtered']}")
    print(f"Available: {stats['processed']}")
    
    # Access chunk metadata
    if chunks:
        first_chunk = chunks[0]
        print(f"\nFirst chunk:")
        print(f"  UID: {first_chunk.uid}")
        print(f"  S3 URI: {first_chunk.s3_uri}")
        print(f"  Score: {first_chunk.metadata['score']}")


def example_download_files():
    """Download files from S3 without processing."""
    print("=" * 60)
    print("Example 5: Download Files Only")
    print("=" * 60)
    
    downloader = S3Downloader(max_workers=8)
    
    s3_uris = [
        "s3://esa-satcom-s3/chunks/arxiv/doc1/chunk1.md",
        "s3://esa-satcom-s3/chunks/arxiv/doc1/chunk2.md",
    ]
    
    downloaded = downloader.download_batch(s3_uris, local_dir="temp_downloads")
    
    print(f"Downloaded {len(downloaded)} files:")
    for s3_uri, local_path in downloaded.items():
        print(f"  {s3_uri} -> {local_path}")


def example_gpu_management():
    """Monitor and manage GPU memory."""
    print("=" * 60)
    print("Example 6: GPU Management")
    print("=" * 60)
    
    gpu = GPUManager(memory_threshold=0.80)
    
    # Check if CUDA is available
    if gpu.is_available():
        print("CUDA is available")
        
        # Get device info
        info = gpu.get_device_info()
        print(f"Device count: {info['device_count']}")
        print(f"Device name: {info['device_name']}")
        
        # Get memory stats
        stats = gpu.get_memory_stats()
        print(f"Allocated: {stats['allocated_gb']:.2f}GB")
        print(f"Reserved: {stats['reserved_gb']:.2f}GB")
        print(f"Total: {stats['total_gb']:.2f}GB")
        print(f"Usage: {stats['usage_percent']:.1f}%")
        
        # Clear cache if needed
        if gpu.check_and_clear_if_needed():
            print("Cache cleared due to high memory usage")
    else:
        print("CUDA not available")


def example_environment_config():
    """Use environment variables for configuration."""
    print("=" * 60)
    print("Example 7: Environment-Based Configuration")
    print("=" * 60)
    
    # Set environment variables (normally done in shell)
    os.environ["CONFIG_PATH"] = "/workspace/config_qwen.yaml"
    os.environ["CHUNKS_FOLDER"] = "/workspace/chunks"
    os.environ["SCORE_THRESHOLD"] = "-11"
    os.environ["SKIP_FIRST_CHUNKS"] = "0"
    os.environ["QDRANT_URL"] = "https://your-qdrant-url:6333"
    os.environ["QDRANT_API_KEY"] = "your-api-key"
    
    # Load from environment
    config = ConfigLoader.load_config(
        config_path=os.getenv("CONFIG_PATH"),
        chunks_folder=os.getenv("CHUNKS_FOLDER"),
        db_url=os.getenv("QDRANT_URL"),
        db_api_key=os.getenv("QDRANT_API_KEY"),
        score_threshold=float(os.getenv("SCORE_THRESHOLD", "0")),
        skip_first_chunks=int(os.getenv("SKIP_FIRST_CHUNKS", "0"))
    )
    
    print(f"Collection: {config.database.collection_name}")
    print(f"Model: {config.embedding.model_name}")
    print(f"Batch size: {config.upload.batch_size}")


def example_custom_workflow():
    """Custom workflow with individual components."""
    print("=" * 60)
    print("Example 8: Custom Workflow")
    print("=" * 60)
    
    # Step 1: Parse chunks
    parser = ChunkParser(score_threshold=0.0)
    chunks, stats = parser.parse_file("/workspace/chunks/arxiv.json")
    print(f"Parsed {len(chunks)} chunks")
    
    # Step 2: Download files
    downloader = S3Downloader(max_workers=8)
    s3_uris = [chunk.s3_uri for chunk in chunks[:10]]  # First 10 only
    files = downloader.download_batch(s3_uris)
    print(f"Downloaded {len(files)} files")
    
    # Step 3: Generate embeddings
    embedder, vector_size = EmbeddingModelFactory.create(
        model_name="Qwen/Qwen3-Embedding-4B",
        model_type="sentence"
    )
    
    # Read file contents
    texts = []
    for s3_uri, local_path in files.items():
        with open(local_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    
    embeddings = embedder.embed_documents(texts[:10])
    print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
    
    # Step 4: Upload to Qdrant
    from qdrant_uploader import QdrantUploader
    
    uploader = QdrantUploader(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name="satcom-chunks-collection"
    )
    
    # Prepare data
    batch_ids = [chunk.uid for chunk in chunks[:10]]
    batch_metadata = [chunk.metadata for chunk in chunks[:10]]
    
    success, failed = uploader.upload_batch(batch_ids, embeddings, batch_metadata)
    
    if success:
        print(f"Successfully uploaded {len(batch_ids)} vectors")
    else:
        print(f"Failed to upload {len(failed or [])} vectors")


if __name__ == "__main__":
    """
    Run examples (comment out the ones you don't want to run).
    
    NOTE: Make sure to set proper environment variables before running:
    - QDRANT_URL
    - QDRANT_API_KEY
    - Ensure AWS credentials are configured
    """
    
    # Uncomment the examples you want to run:
    
    # example_basic_usage()
    # example_single_file()
    example_custom_embedder()
    example_parse_chunks()
    # example_download_files()
    example_gpu_management()
    example_environment_config()
    # example_custom_workflow()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)

