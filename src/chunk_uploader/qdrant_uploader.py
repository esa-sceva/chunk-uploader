"""Qdrant database upload management."""
import time
import hashlib
import numpy as np
from qdrant_client import QdrantClient, models
from typing import List, Tuple, Optional


class QdrantUploader:
    """Handle uploads to Qdrant vector database."""
    
    def __init__(self, url: str, api_key: str, collection_name: str,
                 timeout: float = 60.0, prefer_grpc: bool = False,
                 max_retries: int = 3, retry_delay_base: float = 2.0):
        self.collection_name = collection_name
        self.max_retries = max_retries
        self.retry_delay_base = retry_delay_base
        
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout,
            prefer_grpc=prefer_grpc
        )
    
    @staticmethod
    def string_to_uint(s: str) -> int:
        """Convert string to unsigned integer hash."""
        hash_bytes = hashlib.sha256(s.encode("utf-8")).digest()
        return int.from_bytes(hash_bytes[:8], byteorder="big", signed=False)
    
    def check_collection_health(self) -> bool:
        """Check if collection is accessible."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            point_count = collection_info.points_count
            print(f"Collection '{self.collection_name}' status: {point_count} points")
            return True
        except Exception as e:
            print(f"Collection access error: {e}")
            return False
    
    def upload_batch(self, batch_ids: List[str], batch_vectors: List[List[float]], 
                    batch_metadata: List[dict]) -> Tuple[bool, Optional[List[str]]]:
        """Upload a batch of vectors with retry logic."""
        batch_points = [
            models.PointStruct(
                id=self.string_to_uint(uid),
                vector=vec,
                payload=meta
            )
            for uid, vec, meta in zip(batch_ids, batch_vectors, batch_metadata)
        ]
        
        for attempt in range(self.max_retries):
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
                return True, None
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle timeout and connection errors
                is_retryable = any(
                    keyword in error_msg 
                    for keyword in ['timeout', 'timed out', 'connection', 'network', 'write timed out']
                )
                
                if is_retryable and attempt < self.max_retries - 1:
                    delay = self.retry_delay_base * (2 ** attempt) + np.random.uniform(0, 2)
                    print(f"Upload error (attempt {attempt + 1}/{self.max_retries})")
                    print(f"Retrying in {delay:.2f}s... Error: {e}")
                    time.sleep(delay)
                    continue
                else:
                    print(f"Upload failed: {e}")
                    return False, batch_ids
        
        return False, batch_ids
    
    def create_collection(self, vector_size: int, distance: str = "COSINE", force: bool = False):
        """Create a new collection."""
        try:
            if force:
                try:
                    self.client.delete_collection(self.collection_name)
                    print(f"Deleted existing collection")
                except:
                    pass
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE if distance == "COSINE" else models.Distance.EUCLIDEAN
                )
            )
            print(f"Created collection '{self.collection_name}' with vector size {vector_size}")
            
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

