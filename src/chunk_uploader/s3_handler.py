"""Direct S3 chunk handling without metadata files."""
import os
import boto3
from typing import List, Dict, Optional
from botocore.exceptions import ClientError


class S3ChunkHandler:
    """Handle direct S3 chunk operations."""
    
    def __init__(self):
        self.s3_client = boto3.client("s3")
    
    def list_chunks_from_s3(self, s3_path: str, extensions: List[str] = ['.md', '.txt']) -> List[Dict]:
        """
        List all chunk files from S3 path.
        
        Args:
            s3_path: S3 path (e.g., s3://bucket/path/to/chunks/)
            extensions: File extensions to include
            
        Returns:
            List of chunk metadata dictionaries
        """
        # Parse S3 path
        if not s3_path.startswith('s3://'):
            raise ValueError(f"S3 path must start with 's3://': {s3_path}")
        
        path_parts = s3_path.replace('s3://', '').split('/', 1)
        bucket = path_parts[0]
        prefix = path_parts[1] if len(path_parts) > 1 else ''
        
        print(f"📂 Listing files from s3://{bucket}/{prefix}")
        
        chunks = []
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    key = obj['Key']
                    
                    # Check if file has desired extension
                    if not any(key.endswith(ext) for ext in extensions):
                        continue
                    
                    # Extract file name
                    file_name = os.path.basename(key)
                    
                    # Create chunk metadata
                    chunk = {
                        'uid': file_name,
                        's3_uri': f"s3://{bucket}/{key}",
                        'metadata': {
                            'source_url': f"s3://{bucket}/{key}",
                            'score': 1.0,  # Default score
                            'original_file_name': file_name,
                            'sub_folder': os.path.dirname(key).split('/')[-1] if '/' in key else bucket,
                            'chunk_name': file_name,
                            'source_json_file': 'direct_s3',
                        }
                    }
                    
                    chunks.append(chunk)
            
            print(f"✅ Found {len(chunks)} chunk files in S3")
            return chunks
            
        except ClientError as e:
            print(f"❌ Error accessing S3: {e}")
            raise
    
    def download_file(self, s3_uri: str, local_dir: str = "downloads") -> Optional[str]:
        """Download a single file from S3."""
        if not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)

        bucket, key = s3_uri.replace("s3://", "").split("/", 1)
        local_path = os.path.join(local_dir, os.path.basename(key))

        try:
            self.s3_client.download_file(bucket, key, local_path)
            return local_path
        except Exception as e:
            print(f"⚠️ Error downloading {s3_uri}: {e}")
            return None


class DirectS3ChunkMetadata:
    """Simple wrapper for S3 chunk metadata."""
    
    def __init__(self, uid: str, s3_uri: str, metadata: dict):
        self.uid = uid
        self.s3_uri = s3_uri
        self.metadata = metadata

