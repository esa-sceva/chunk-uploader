"""S3 file download management."""
import os
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional


class S3Downloader:
    """Handle S3 file downloads."""
    
    def __init__(self, max_workers: int = 8):
        self.s3_client = boto3.client("s3")
        self.max_workers = max_workers
    
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
    
    def download_batch(self, s3_uris: List[str], local_dir: str = "downloads") -> Dict[str, str]:
        """Download multiple files from S3 concurrently."""
        downloaded_files = {}
        
        def download_single(s3_uri: str) -> tuple[str, Optional[str]]:
            local_path = self.download_file(s3_uri, local_dir)
            if local_path:
                downloaded_files[s3_uri] = local_path
            return s3_uri, local_path
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(download_single, uri): uri for uri in s3_uris}
            
            for future in as_completed(futures):
                s3_uri = futures[future]
                try:
                    uri, local_path = future.result()
                    if local_path:
                        print(f"⬇️ Downloaded {uri}")
                except Exception as e:
                    print(f"⚠️ Error downloading {s3_uri}: {e}")
        
        return downloaded_files


def cleanup_files(file_paths: List[str]):
    """Clean up downloaded files."""
    for p in file_paths:
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception as e:
            print(f"⚠️ Could not delete {p}: {e}")

