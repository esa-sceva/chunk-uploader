#!/usr/bin/env python3
"""Verify setup is correct."""
import sys
import os
from pathlib import Path

print("=" * 60)
print("Chunk Uploader - Setup Verification")
print("=" * 60)
print()

# Check Python version
print(f"✓ Python version: {sys.version}")
print(f"✓ Python executable: {sys.executable}")
print()

# Check working directory
print(f"✓ Current directory: {os.getcwd()}")
print()

# Check src folder exists
src_path = Path(__file__).parent / "src" / "chunk_uploader"
if src_path.exists():
    print(f"✓ Source folder exists: {src_path}")
    modules = list(src_path.glob("*.py"))
    print(f"✓ Found {len(modules)} Python modules")
else:
    print(f"✗ Source folder not found: {src_path}")
    sys.exit(1)

print()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
print(f"✓ Added to path: {Path(__file__).parent / 'src'}")
print()

# Test imports
print("Testing imports...")
print()

errors = []

try:
    from chunk_uploader import config
    print("  ✓ chunk_uploader.config")
except Exception as e:
    print(f"  ✗ chunk_uploader.config: {e}")
    errors.append(("config", e))

try:
    from chunk_uploader import embeddings
    print("  ✓ chunk_uploader.embeddings")
except Exception as e:
    print(f"  ✗ chunk_uploader.embeddings: {e}")
    errors.append(("embeddings", e))

try:
    from chunk_uploader import s3_handler
    print("  ✓ chunk_uploader.s3_handler")
except Exception as e:
    print(f"  ✗ chunk_uploader.s3_handler: {e}")
    errors.append(("s3_handler", e))

try:
    from chunk_uploader import uploader
    print("  ✓ chunk_uploader.uploader")
except Exception as e:
    print(f"  ✗ chunk_uploader.uploader: {e}")
    errors.append(("uploader", e))

print()

if errors:
    print("=" * 60)
    print(f"✗ {len(errors)} import error(s) found:")
    print("=" * 60)
    for module, error in errors:
        print(f"\n{module}:")
        print(f"  {error}")
    sys.exit(1)
else:
    print("=" * 60)
    print("✓ All imports successful!")
    print("=" * 60)
    print()
    print("Setup is correct. You can now run:")
    print()
    print("  1. Set environment variables:")
    print("     set QDRANT_URL=https://your-instance:6333")
    print("     set QDRANT_API_KEY=your-api-key")
    print("     set S3_CHUNKS_PATH=s3://bucket/path/")
    print()
    print("  2. Run the uploader:")
    print("     py main.py")
    print()

