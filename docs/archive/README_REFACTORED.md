# Chunk Uploader - Refactored Version

A modular, production-ready Python application for uploading document chunks with embeddings to Qdrant vector database.

## Features

- ✅ **Modular Architecture**: Separated into focused, reusable modules
- ✅ **Multiple Embedding Models**: Support for Qwen, NASA, Indus models
- ✅ **GPU Memory Management**: Automatic monitoring and cleanup
- ✅ **Concurrent Processing**: Parallel S3 downloads and batch uploads
- ✅ **Retry Logic**: Exponential backoff for failed uploads
- ✅ **Statistics Tracking**: Comprehensive upload statistics
- ✅ **Environment Configuration**: Secure credential management
- ✅ **Score Filtering**: Filter chunks by quality score
- ✅ **Resume Support**: Skip already processed chunks

## Architecture

The application is structured into 9 focused modules:

```
chunk-uploader/
├── config.py           # Configuration management
├── embeddings.py       # Embedding model implementations
├── s3_downloader.py    # S3 file download operations
├── chunk_parser.py     # JSON chunk file parsing
├── qdrant_uploader.py  # Qdrant database operations
├── stats_tracker.py    # Statistics tracking and reporting
├── gpu_manager.py      # GPU memory management
├── uploader.py         # Main upload orchestration
└── main.py             # Application entry point
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (optional, but recommended)
- AWS credentials configured
- Qdrant instance

### Install Dependencies

```bash
pip install -r requirements.txt
```

### For GPU acceleration:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Configuration

### 1. YAML Configuration File

Create or edit `config_qwen.yaml`:

```yaml
upload_params:
  batch_size: 10000
  vector_size: 2560  # 2560 for Qwen3-4B, 768 for NASA/Indus

database:
  collection_name: "satcom-chunks-collection"

embedding:
  model_name: "Qwen/Qwen3-Embedding-4B"
  # Options:
  # - "Qwen/Qwen3-Embedding-4B"
  # - "Qwen/Qwen3-Embedding-0.6B"
  # - "nasa-impact/nasa-smd-ibm-st-v2"
  # - "Tulsikumar/indus-sde-st-v0.2"
  type: 'sentence'  # 'sentence' or 'transformer' (Qwen only)
  normalize: true
```

### 2. Environment Variables

Set the following environment variables:

```bash
# Required
export QDRANT_URL="https://your-qdrant-url:6333"
export QDRANT_API_KEY="your-api-key"

# Optional (with defaults)
export CONFIG_PATH="/workspace/config_qwen.yaml"
export CHUNKS_FOLDER="/workspace/pod4"
export SCORE_THRESHOLD="-11"
export SKIP_FIRST_CHUNKS="0"
export POD_ID="pod1"  # For multi-pod scenarios
```

### 3. AWS Credentials

Ensure AWS credentials are configured:

```bash
aws configure
```

Or set environment variables:
```bash
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
```

## Usage

### Basic Usage

```bash
python main.py
```

### Programmatic Usage

```python
from config import ConfigLoader
from uploader import ChunkUploader

# Load configuration
config = ConfigLoader.load_config(
    config_path="config_qwen.yaml",
    chunks_folder="/path/to/chunks",
    db_url="https://your-qdrant-url:6333",
    db_api_key="your-api-key",
    score_threshold=-11,
    skip_first_chunks=0
)

# Create uploader and run
uploader = ChunkUploader(config)
uploader.upload_all()
```

### Advanced Usage

#### Process a Single File

```python
uploader = ChunkUploader(config)
uploader.upload_file("/path/to/chunks/file.json")
```

#### Use Different Embedding Models

```python
from embeddings import EmbeddingModelFactory

# Qwen 4B with sentence-transformers
embedder, vector_size = EmbeddingModelFactory.create(
    model_name="Qwen/Qwen3-Embedding-4B",
    model_type='sentence'
)

# NASA model
embedder, vector_size = EmbeddingModelFactory.create(
    model_name="nasa-impact/nasa-smd-ibm-st-v2"
)
```

#### Custom GPU Management

```python
from gpu_manager import GPUManager

gpu = GPUManager(memory_threshold=0.80)  # 80% threshold
gpu.print_device_info()
gpu.check_and_clear_if_needed()
```

## Module Documentation

### `config.py` - Configuration Management
Handles loading and validation of configuration from YAML files and environment variables.

**Key Classes**:
- `AppConfig`: Main configuration container
- `ConfigLoader`: Load and merge configurations

### `embeddings.py` - Embedding Models
Provides embedding model implementations with a unified interface.

**Key Classes**:
- `BaseEmbedder`: Abstract base for all embedders
- `QwenSentenceTransformerEmbedder`: Qwen with sentence-transformers
- `QwenTransformerEmbedder`: Qwen with direct transformers
- `EmbeddingModelFactory`: Factory for creating embedders

### `s3_downloader.py` - S3 Operations
Handles concurrent downloads from S3.

**Key Classes**:
- `S3Downloader`: Concurrent S3 file downloader

### `chunk_parser.py` - JSON Parsing
Parses various JSON chunk file formats.

**Key Classes**:
- `ChunkParser`: Parse and filter chunks from JSON
- `ChunkMetadata`: Chunk data container

### `qdrant_uploader.py` - Database Operations
Manages uploads to Qdrant with retry logic.

**Key Classes**:
- `QdrantUploader`: Upload vectors to Qdrant

### `stats_tracker.py` - Statistics Tracking
Tracks and persists comprehensive upload statistics.

**Key Classes**:
- `StatsTracker`: Statistics tracking and reporting

### `gpu_manager.py` - GPU Management
Monitors and manages GPU memory.

**Key Classes**:
- `GPUManager`: GPU memory monitoring and cleanup

### `uploader.py` - Upload Orchestration
Coordinates the entire upload process.

**Key Classes**:
- `ChunkUploader`: Main orchestrator

## Output

### Console Output

The application provides detailed progress information:

```
🚀 Chunk Uploader
==================================================
🤖 Model: Qwen/Qwen3-Embedding-4B
   Type: sentence
📊 Score threshold: -11
⏭️ Skip first chunks: 0
📦 Batch size: 24
📦 Subset size: 96
📂 Chunks folder: /workspace/pod4

🔍 GPU Info:
   Device count: 1
   Current device: 0
   Device name: NVIDIA A100-SXM4-80GB

📂 Processing file: arxiv.json
   Found 5000 total chunks in arxiv
   Skipped 0 chunks
   Filtered 100 chunks below score threshold -11
   Remaining 4900 chunks for processing

🔄 Processing subset 1/51: chunks 1-96
⬇️ Downloaded 96 chunks in 2.5s
🧠 Generating embeddings for 96 texts...
✅ Embeddings generated in 12.3s
📤 Uploading batch 1/4 (24 points)
✅ Batch uploaded: 24 chunks

...

🎉 All uploads complete
📊 Final upload summary:
   📈 Total chunks found: 50000
   ⏭️ Skipped chunks: 0
   🚫 Filtered by score threshold: 1000
   ⚡ Available for processing: 49000
   🎯 Successfully uploaded to Qdrant: 48500
   ❌ Failed uploads: 500
   📊 Upload success rate: 99.0%
```

### Statistics File

A JSON statistics file is generated: `upload_stats_pod1_20250126_143022.json`

```json
{
  "timestamp": "2025-01-26T14:35:22Z",
  "started_at": "2025-01-26T14:30:22Z",
  "collection_name": "satcom-chunks-collection",
  "pod_id": "pod1",
  "score_threshold": -11,
  "total_chunks": 50000,
  "total_filtered": 1000,
  "total_uploaded_to_qdrant": 48500,
  "upload_success_rate": 99.0,
  "per_file_stats": {
    "arxiv.json": {
      "total_chunks": 5000,
      "filtered_chunks": 100,
      "processed": 4900,
      "succeeded": 4850,
      "failed": 50
    }
  }
}
```

## Troubleshooting

### GPU Out of Memory

Reduce batch sizes in config:
```python
config.upload.batch_size = 8  # Reduce from 24
```

### Connection Timeouts

Increase timeout and retries:
```python
config.database.timeout = 120.0  # Increase from 60
config.upload.max_retries = 5     # Increase from 3
```

### Slow Downloads

Increase download threads:
```python
config.upload.download_threads = 16  # Increase from 8
```

### Model Not Loading

Check transformers version:
```bash
pip install --upgrade transformers>=4.51.0
pip install --upgrade sentence-transformers>=2.7.0
```

Clear HuggingFace cache:
```bash
rm -rf ~/.cache/huggingface/
```

## Performance Tips

1. **Use GPU**: Embeddings are 10-100x faster on GPU
2. **Batch Size**: Larger batches = faster processing (if memory allows)
3. **Download Threads**: More threads = faster S3 downloads (up to ~16)
4. **Subset Size**: Process 96-192 chunks at once for optimal memory usage
5. **Score Filtering**: Filter low-quality chunks early to save processing

## Comparison with Original

| Aspect | Original | Refactored | Improvement |
|--------|----------|------------|-------------|
| Files | 1 | 9 | Modular ✅ |
| Lines per file | 1377 | ~150 avg | 89% smaller ✅ |
| Hardcoded credentials | Yes ❌ | No ✅ | Secure ✅ |
| Testable | No ❌ | Yes ✅ | Quality ✅ |
| Reusable | No ❌ | Yes ✅ | Flexible ✅ |

## Contributing

To add a new embedding model:

1. Create a new embedder class inheriting from `BaseEmbedder`
2. Implement `embed_documents()` method
3. Register in `EmbeddingModelFactory`
4. Add vector size to `VECTOR_SIZES` dict

Example:
```python
class MyCustomEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        self.model = load_my_model(model_name)
    
    def embed_documents(self, texts, batch_size=8, normalize=True):
        return self.model.encode(texts)

# In EmbeddingModelFactory:
VECTOR_SIZES["my-custom-model"] = 1536

@classmethod
def create(cls, model_name: str, ...):
    if "my-custom-model" in model_name:
        return MyCustomEmbedder(model_name), 1536
```

## License

[Add your license here]

## Support

For issues and questions, please open an issue on the repository.

