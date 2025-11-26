# Quick Start Guide

## Upload from S3 (Recommended)

```bash
# 1. Set credentials
export QDRANT_URL="https://your-instance:6333"
export QDRANT_API_KEY="your-api-key"

# 2. Set S3 path
export S3_CHUNKS_PATH="s3://your-bucket/path/to/chunks/"

# 3. Run
python main.py
```

## Upload from Local JSON Metadata

```bash
# 1. Set credentials
export QDRANT_URL="https://your-instance:6333"
export QDRANT_API_KEY="your-api-key"

# 2. Set local folder
export CHUNKS_FOLDER="/path/to/json/files"

# 3. Run
python main.py
```


## Configuration

Edit `config/config_qwen.yaml` to change:
- Collection name
- Embedding model
- Vector dimensions

See [README.md](README.md) for full documentation.

