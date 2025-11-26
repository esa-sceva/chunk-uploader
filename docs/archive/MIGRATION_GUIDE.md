# Migration Guide: From Monolithic to Modular Architecture

This guide helps you transition from the old `chunk_uploader_main.py` to the new modular architecture.

## Quick Start

### Old Way (Still Works)
```python
from chunk_uploader_main import MinimalQdrantUploader

uploader = MinimalQdrantUploader(
    config_path="/workspace/config_qwen.yaml",
    chunks_folder="/workspace/pod4",
    score_threshold=-11,
    skip_first_chunks=0
)
uploader.upload_to_qdrant()
```

### New Way (Recommended)
```python
from config import ConfigLoader
from uploader import ChunkUploader

config = ConfigLoader.load_config(
    config_path="/workspace/config_qwen.yaml",
    chunks_folder="/workspace/pod4",
    db_url="your-qdrant-url",
    db_api_key="your-api-key",
    score_threshold=-11,
    skip_first_chunks=0
)

uploader = ChunkUploader(config)
uploader.upload_all()
```

## Key Differences

### 1. Configuration

#### Old: Hardcoded Credentials ❌
```python
# In chunk_uploader_main.py (lines 246-247)
self.qdrant_url = "https://ee10c103-8ab1-47dc-a788-341c02741b31..."
self.qdrant_api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

#### New: Environment Variables ✅
```python
# Pass via constructor or environment
db_url = os.getenv("QDRANT_URL")
db_api_key = os.getenv("QDRANT_API_KEY")

config = ConfigLoader.load_config(
    config_path="config.yaml",
    chunks_folder="/data",
    db_url=db_url,
    db_api_key=db_api_key
)
```

### 2. Module Organization

#### Old: Everything in One Class
```python
class MinimalQdrantUploader:
    def __init__(self, ...):
        # Database setup
        # S3 client setup
        # Embedding model loading
        # GPU management
        # Stats tracking
        # ... (1000+ lines)
    
    def upload_to_qdrant(self):
        # 400+ lines of mixed logic
```

#### New: Separated Concerns
```python
# Each has its own file and responsibility
from s3_downloader import S3Downloader      # S3 operations
from embeddings import EmbeddingFactory     # Embedding models
from qdrant_uploader import QdrantUploader  # Database operations
from chunk_parser import ChunkParser        # JSON parsing
from stats_tracker import StatsTracker      # Statistics
from gpu_manager import GPUManager          # GPU management
from uploader import ChunkUploader          # Orchestration
```

### 3. Embedding Model Loading

#### Old: Monolithic Function
```python
def load_hf_embeddings(model_name: str, model_type: str = 'sentence', normalize: bool = True):
    # 50 lines of if-elif chains
    if "nasa-impact" in model_name:
        embedder = HuggingFaceEmbeddings(...)
        return embedder, 768
    elif "Qwen" in model_name:
        if model_type == 'sentence':
            embedder = QwenSentenceTransformerEmbedder(...)
        # ... more conditions
```

#### New: Factory Pattern
```python
from embeddings import EmbeddingModelFactory

embedder, vector_size = EmbeddingModelFactory.create(
    model_name="Qwen/Qwen3-Embedding-4B",
    model_type="sentence",
    normalize=True
)
```

### 4. JSON Parsing

#### Old: 400+ Line Method
```python
def process_json_file(self, file_path: str):
    # Lines 459-763 (400+ lines)
    # Nested if-elif for different JSON formats
    # Repeated filtering logic
    # Mixed parsing and filtering
```

#### New: Dedicated Parser Class
```python
from chunk_parser import ChunkParser

parser = ChunkParser(score_threshold=0.5, skip_first_chunks=100)
chunks, stats = parser.parse_file("data.json")

# Clean separation:
# - ChunkParser: handles all JSON formats
# - Filtering: centralized in one place
# - Stats: returned separately
```

### 5. Statistics Tracking

#### Old: Mixed with Upload Logic
```python
def upload_to_qdrant(self):
    # Stats variables scattered throughout
    total_chunks = 0
    total_filtered = 0
    succeeded = 0
    failed = 0
    per_file_stats = {}
    
    # Update stats inline with upload logic
    succeeded += len(batch_ids)
    per_file_stats[file_name]["succeeded"] += len(batch_ids)
    
    # Stats writing mixed with upload logic
    write_stats()
```

#### New: Dedicated Stats Tracker
```python
from stats_tracker import StatsTracker

stats = StatsTracker(pod_id="pod1", collection_name="my-collection", score_threshold=0.5)

# Update stats
stats.update_global(uploaded=24, succeeded=24)
stats.update_file("arxiv.json", succeeded=24)

# Write stats
stats.write_stats()

# Print summary
stats.print_summary()
```

### 6. GPU Memory Management

#### Old: Inline Checks
```python
# Scattered throughout the code
if torch.cuda.is_available():
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    memory_reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU: {memory_allocated:.1f}GB allocated")
    torch.cuda.empty_cache()
    gc.collect()
```

#### New: Dedicated GPU Manager
```python
from gpu_manager import GPUManager

gpu = GPUManager(memory_threshold=0.85)

# Check and print
gpu.print_memory_stats("Before Embedding")

# Auto cleanup if needed
gpu.check_and_clear_if_needed()

# Manual cleanup
gpu.clear_cache()
```

## Step-by-Step Migration

### Step 1: Install New Modules

The new modules are in the same directory. No installation needed if you have the files.

### Step 2: Update Import Statements

**Old:**
```python
from chunk_uploader_main import MinimalQdrantUploader, load_hf_embeddings, load_config
```

**New:**
```python
from config import ConfigLoader
from uploader import ChunkUploader
from embeddings import EmbeddingModelFactory
```

### Step 3: Update Configuration Loading

**Old:**
```python
uploader = MinimalQdrantUploader(
    config_path="config.yaml",
    chunks_folder="/data",
    score_threshold=-11,
    skip_first_chunks=0
)
```

**New:**
```python
config = ConfigLoader.load_config(
    config_path="config.yaml",
    chunks_folder="/data",
    db_url=os.getenv("QDRANT_URL"),
    db_api_key=os.getenv("QDRANT_API_KEY"),
    score_threshold=-11,
    skip_first_chunks=0
)

uploader = ChunkUploader(config)
```

### Step 4: Set Environment Variables

Create a `.env` file or export variables:

```bash
export QDRANT_URL="https://your-qdrant-url:6333"
export QDRANT_API_KEY="your-api-key"
export AWS_ACCESS_KEY_ID="your-aws-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret"
```

### Step 5: Update Execution

**Old:**
```python
uploader.upload_to_qdrant()
```

**New:**
```python
uploader.upload_all()
```

Or use the main script:
```bash
python main.py
```

## Feature Parity

All features from the old version are available in the new version:

| Feature | Old | New | Notes |
|---------|-----|-----|-------|
| S3 Download | ✅ | ✅ | Concurrent downloads |
| Multiple JSON Formats | ✅ | ✅ | Cleaner parsing |
| Score Filtering | ✅ | ✅ | Centralized |
| Skip Chunks | ✅ | ✅ | Same behavior |
| Batch Upload | ✅ | ✅ | With retry logic |
| Stats Tracking | ✅ | ✅ | Better organized |
| GPU Management | ✅ | ✅ | Dedicated module |
| Multiple Embedders | ✅ | ✅ | Factory pattern |
| Error Handling | ✅ | ✅ | Improved |
| Progress Reporting | ✅ | ✅ | More detailed |

## Testing Your Migration

### 1. Test Configuration Loading

```python
from config import ConfigLoader

config = ConfigLoader.load_config(
    config_path="config_qwen.yaml",
    chunks_folder="/workspace/chunks",
    db_url="test-url",
    db_api_key="test-key"
)

print(f"Collection: {config.database.collection_name}")
print(f"Model: {config.embedding.model_name}")
print(f"Batch size: {config.upload.batch_size}")
```

### 2. Test Embedding Model

```python
from embeddings import EmbeddingModelFactory

embedder, vector_size = EmbeddingModelFactory.create(
    model_name="Qwen/Qwen3-Embedding-4B"
)

test_embeddings = embedder.embed_documents(["Test sentence"])
print(f"Embedding shape: {len(test_embeddings)} x {len(test_embeddings[0])}")
```

### 3. Test Chunk Parsing

```python
from chunk_parser import ChunkParser

parser = ChunkParser(score_threshold=0.0)
chunks, stats = parser.parse_file("test.json")

print(f"Total: {stats['total']}")
print(f"Processed: {stats['processed']}")
```

### 4. Test Full Pipeline (Dry Run)

```python
from config import ConfigLoader
from uploader import ChunkUploader

config = ConfigLoader.load_config(
    config_path="config.yaml",
    chunks_folder="/workspace/test_chunks",  # Small test folder
    db_url=os.getenv("QDRANT_URL"),
    db_api_key=os.getenv("QDRANT_API_KEY"),
    score_threshold=-11,
    skip_first_chunks=0
)

uploader = ChunkUploader(config)
uploader.upload_all()  # Run on small dataset first
```

## Troubleshooting

### "Module not found" errors

**Solution:** Ensure all new module files are in the same directory:
- `config.py`
- `embeddings.py`
- `chunk_parser.py`
- `s3_downloader.py`
- `qdrant_uploader.py`
- `stats_tracker.py`
- `gpu_manager.py`
- `uploader.py`
- `main.py`

### "Credentials not set" errors

**Solution:** Set environment variables:
```bash
export QDRANT_URL="your-url"
export QDRANT_API_KEY="your-key"
```

Or pass directly:
```python
config = ConfigLoader.load_config(
    ...,
    db_url="https://your-url",
    db_api_key="your-key"
)
```

### Different results from old version

**Expected:** Results should be identical. If not:
1. Check score threshold is the same
2. Check skip_first_chunks is the same
3. Verify the same model is being used
4. Check batch sizes are similar

## Rollback Plan

If you need to rollback:

1. The old `chunk_uploader_main.py` is **unchanged**
2. Simply use the old import:
   ```python
   from chunk_uploader_main import MinimalQdrantUploader
   ```
3. Or run the old script directly:
   ```bash
   python chunk_uploader_main.py
   ```

The new modules don't affect the old code at all.

## Benefits of Migration

1. **Security**: No hardcoded credentials
2. **Maintainability**: Smaller, focused modules
3. **Testability**: Each component can be tested independently
4. **Flexibility**: Easy to swap components
5. **Reusability**: Modules can be used in other projects
6. **Debugging**: Easier to locate and fix issues
7. **Extensibility**: Easy to add new features

## Next Steps

After successful migration:

1. Remove hardcoded credentials from old code
2. Add unit tests for new modules
3. Create CI/CD pipeline
4. Add logging framework
5. Consider async operations for better performance

## Support

If you encounter issues during migration:

1. Check this guide
2. Review `REFACTORING.md` for architecture details
3. See `example_usage.py` for code examples
4. Open an issue with details of the problem

## Checklist

Before going to production with new code:

- [ ] All environment variables are set
- [ ] Configuration file is updated
- [ ] AWS credentials are configured
- [ ] Qdrant connection is tested
- [ ] Small test run completed successfully
- [ ] Statistics file is generated correctly
- [ ] GPU memory management works
- [ ] Error handling is verified
- [ ] Backup of old code exists
- [ ] Team is trained on new architecture

