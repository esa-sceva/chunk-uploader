# Chunk Uploader

Upload document chunks with embeddings to Qdrant vector database. Supports direct S3 uploads or local JSON metadata.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set credentials
export QDRANT_URL="https://your-qdrant-instance:6333"
export QDRANT_API_KEY="your-api-key"

# Option 1: Upload from S3 path directly
export S3_CHUNKS_PATH="s3://your-bucket/chunks/"
python main.py

# Option 2: Upload from local JSON metadata files
export CHUNKS_FOLDER="/path/to/json/files"
python main.py
```

## Project Structure

```
chunk-uploader/
├── src/chunk_uploader/      # Core Python modules
│   ├── config.py            # Configuration management
│   ├── embeddings.py        # Embedding models (Qwen, NASA, Indus)
│   ├── s3_downloader.py     # S3 file operations
│   ├── chunk_parser.py      # JSON parsing
│   ├── qdrant_uploader.py   # Database operations
│   ├── stats_tracker.py     # Statistics tracking
│   ├── gpu_manager.py       # GPU memory management
│   └── uploader.py          # Main orchestration
|
├── config/
│   └── config_qwen.yaml     # Main configuration
|
├── scripts/
│   └── recreate_collection.py  # Recreate Qdrant collection
|
├── examples/
│   └── example_usage.py     # Code examples
|
├── main.py                  # Main entry point
└── requirements.txt         # Dependencies
```

## Configuration

Edit `config/config_qwen.yaml`:

```yaml
database:
  collection_name: "your-collection-name"  # Your collection name

embedding:
  model_name: "Qwen/Qwen3-Embedding-4B"  # Embedding model
  # Options: Qwen/Qwen3-Embedding-0.6B, Qwen/Qwen3-Embedding-4B, 
  #          Qwen/Qwen3-Embedding-8B, nasa-impact/nasa-smd-ibm-st-v2
  type: 'sentence'  # 'sentence' or 'transformer' (for Qwen)
  normalize: true

upload_params:
  batch_size: 10000
  vector_size: 2560  # 2560 for Qwen-4B, 1024 for Qwen-0.6B, 768 for NASA
```

## Metadata Fields

Each chunk uploaded to Qdrant is automatically enriched with comprehensive metadata (if available in input `json` file)

### Core Fields (Auto-generated)
- **`id`** (int) - Unique incremental ID (0, 1, 2, ...)
- **`content`** (str) - Full text content of the chunk
- **`url`** (str) - Source URL (renamed from source_url)
- **`publisher`** (str) - Publisher name (extracted from source_json_file)

### Source Fields (From metadata)
- **`chunk_name`** (str) - Name of the chunk file (e.g., "chunk_001.md")
- **`original_file_name`** (str) - Original document filename
- **`sub_folder`** (str) - Subfolder/category (e.g., "arxiv")
- **`source_json_file`** (str) - Original JSON metadata file
- **`score`** (float) - Quality/relevance score

### Scholarly Metadata (Placeholders for future enrichment)
- **`doi`** (null) - Digital Object Identifier
- **`title`** (null) - Document title
- **`journal`** (null) - Journal name
- **`reference_count`** - Number of references
- **`n_citations`** - Citation count
- **`influential_citation_count`** - Influential citations count
- **`header`** - Document headers/sections

### Example Qdrant Payload

```json
{
  "id": 12345,
  "content": "This is the full text content of the chunk...",
  "chunk_name": "chunk_001.md",
  "original_file_name": "document.pdf",
  "sub_folder": "arxiv",
  "url": "https://arxiv.org/abs/1234.5678",
  "publisher": "arxiv",
  "source_json_file": "arxiv.json",
  "score": 0.95,
  "doi": null,
  "title": null,
  "journal": null,
  "reference_count": 0,
  "n_citations": 0,
  "influential_citation_count": 0,
  "header": []
}
```

**Note:** Scholarly metadata fields are placeholders that can be populated later using external APIs (e.g., Crossref, Semantic Scholar). This allows fast initial upload followed by gradual enrichment.

See **`METADATA_FIELDS.md`** for complete documentation.

## Usage

### 1. Direct S3 Upload (Recommended)

Upload all markdown files from an S3 path:

```bash
export QDRANT_URL="https://your-instance:6333"
export QDRANT_API_KEY="your-api-key"
export S3_CHUNKS_PATH="s3://bucket/path/to/chunks/"

python main.py
```

The uploader will:
- List all `.md` files in the S3 path (recursively)
- Download and process them
- Generate embeddings
- Upload to Qdrant

### 2. Upload from JSON Metadata (Advanced)

If you have JSON files with chunk metadata:

```bash
export QDRANT_URL="https://your-instance:6333"
export QDRANT_API_KEY="your-api-key"
export CHUNKS_FOLDER="/path/to/json/files"

python main.py
```

**JSON Format** (optional):
```json
[
  {
    "chunk_name": "chunk_1.md",
    "output_path": "chunks/folder/doc1/chunk_1.md",
    "document_id": "doc1",
    "source_url": "https://example.com/doc1",
    "score": 0.95
  }
]
```

### 3. Quick Upload to 'try' Collection

```bash
export QDRANT_URL="https://..."
export QDRANT_API_KEY="..."
export S3_CHUNKS_PATH="s3://bucket/chunks/"

python upload_to_try.py
```

## Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `QDRANT_URL` | Yes | Qdrant instance URL | - |
| `QDRANT_API_KEY` | Yes | Qdrant API key | - |
| `S3_CHUNKS_PATH` | No* | S3 path to chunks | - |
| `CHUNKS_FOLDER` | No* | Local JSON metadata folder | - |
| `SCORE_THRESHOLD` | No | Minimum chunk score | -11 |
| `SKIP_FIRST_CHUNKS` | No | Skip first N chunks | 0 |
| `CONFIG_PATH` | No | Config file path | config/config_qwen.yaml |

*Either `S3_CHUNKS_PATH` or `CHUNKS_FOLDER` must be provided.

## Advanced Usage

### Programmatic Usage

```python
from chunk_uploader.config import ConfigLoader
from chunk_uploader.uploader import ChunkUploader

config = ConfigLoader.load_config(
    config_path="config/config_qwen.yaml",
    chunks_folder="/path/to/chunks",  # or s3_path="s3://..."
    db_url="https://your-instance:6333",
    db_api_key="your-api-key",
    score_threshold=-11,
    skip_first_chunks=0
)

uploader = ChunkUploader(config)
uploader.upload_all()
```

### Filter by Score

Only upload high-quality chunks:

```bash
export SCORE_THRESHOLD="0.5"  # Only chunks with score >= 0.5
python main.py
```

### Resume After Interruption

```bash
export SKIP_FIRST_CHUNKS="1000"  # Skip first 1000 chunks
python main.py
```

### Use Different Embedding Model

Edit `config/config_qwen.yaml`:

```yaml
embedding:
  model_name: "Qwen/Qwen3-Embedding-0.6B"  # Smaller, faster model
  vector_size: 1024  # Update vector size accordingly
```

## Creating Qdrant Collection

Before first upload, create the collection:

```bash
python scripts/recreate_collection.py
```

This creates a collection with:
- Name from config (e.g., "try")
- Vector dimensions matching your model
- COSINE distance metric

## Monitoring

The uploader shows real-time progress:

```
Chunk Uploader
Model: Qwen/Qwen3-Embedding-4B
Vector dimensions: 2560
S3 path: s3://bucket/chunks/

Found 1000 markdown files in S3

Processing batch 1/10: 100 files
Downloaded 100 files in 3.2s
Generating embeddings...
Embeddings generated in 15.8s
Uploading to Qdrant...
Uploaded 100 vectors

Final summary:
  Successfully uploaded: 1000 chunks
  Upload success rate: 100.0%
```

Statistics saved to: `upload_stats_pod_XXXXX_TIMESTAMP.json`

## GPU Requirements

| Model | VRAM | Speed |
|-------|------|-------|
| Qwen3-Embedding-0.6B | ~4GB | Fast |
| Qwen3-Embedding-4B | ~16GB | Medium |
| Qwen3-Embedding-8B | ~32GB | Slow |
| NASA SMD | ~4GB | Fast |

**No GPU?** The models will run on CPU (slower but works).

## Troubleshooting

### "Collection not found"
```bash
python scripts/recreate_collection.py
```

### "Credentials not set"
```bash
export QDRANT_URL="https://..."
export QDRANT_API_KEY="..."
```

### "CUDA out of memory"
Use a smaller model in `config/config_qwen.yaml`:
```yaml
embedding:
  model_name: "Qwen/Qwen3-Embedding-0.6B"
  vector_size: 1024
```

### "S3 access denied"
```bash
aws configure  # Set up AWS credentials
aws s3 ls s3://your-bucket/  # Test access
```

### "No files found"
Verify your S3 path:
```bash
aws s3 ls s3://your-bucket/chunks/ --recursive
```

## Examples

See `examples/example_usage.py` for:
- Custom embedding models
- Single file processing
- Custom workflows
- GPU management

## Package Installation

Install as a Python package:

```bash
# Development mode
pip install -e .

# Then import anywhere
from chunk_uploader import ChunkUploader
```

## Performance Tips

1. **Use GPU** - 10-100x faster than CPU
2. **Batch Size** - Larger batches (if memory allows)
3. **Network** - Fast internet for S3 downloads
4. **Model Choice** - Smaller models = faster processing

---

## License

This project is released under the Apache 2.0 License. See the [LICENSE](LICENSE) file for more details.
