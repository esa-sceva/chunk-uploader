"""Main upload orchestration."""
import os
import time
import numpy as np
from typing import List, Dict
from .config import AppConfig
from .embeddings import EmbeddingModelFactory, BaseEmbedder
from .s3_downloader import S3Downloader, cleanup_files
from .s3_handler import S3ChunkHandler, DirectS3ChunkMetadata
from .chunk_parser import ChunkParser, ChunkMetadata
from .qdrant_uploader import QdrantUploader
from .stats_tracker import StatsTracker
from .gpu_manager import GPUManager


class ChunkUploader:
    """Main orchestrator for chunk upload process."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        
        # Initialize components
        self.downloader = S3Downloader(max_workers=config.upload.download_threads)
        self.parser = ChunkParser(
            score_threshold=config.upload.score_threshold,
            skip_first_chunks=config.upload.skip_first_chunks
        )
        self.uploader = QdrantUploader(
            url=config.database.url,
            api_key=config.database.api_key,
            collection_name=config.database.collection_name,
            timeout=config.database.timeout,
            prefer_grpc=config.database.prefer_grpc,
            max_retries=config.upload.max_retries,
            retry_delay_base=config.upload.retry_delay_base
        )
        self.gpu_manager = GPUManager(memory_threshold=config.upload.gpu_memory_threshold)
        
        # Initialize embedding model
        print(f"Loading embedding model...")
        self.gpu_manager.print_device_info()
        
        self.embedder, self.vector_size = EmbeddingModelFactory.create(
            model_name=config.embedding.model_name,
            model_type=config.embedding.model_type,
            normalize=config.embedding.normalize
        )
        
        print(f"Embedding model loaded: {config.embedding.model_name}")
        print(f"Vector dimensions: {self.vector_size}")
        
        # Test embedding
        self._test_embedding()
        
        # Pod ID for multi-pod scenarios
        self.pod_id = os.environ.get('POD_ID', f'pod_{np.random.randint(1000, 9999)}')
        
        # Initialize stats tracker
        self.stats = StatsTracker(
            pod_id=self.pod_id,
            collection_name=config.database.collection_name,
            score_threshold=config.upload.score_threshold
        )
        
        # Initialize counter for incremental IDs
        self.chunk_counter = 0
    
    def _test_embedding(self):
        """Test embedding generation."""
        if not self.gpu_manager.is_available():
            return
        
        print(f"Testing GPU embedding...")
        try:
            start = time.time()
            test_emb = self.embedder.embed_documents(["Test sentence for GPU verification."])
            duration = time.time() - start
            print(f"Test embedding successful in {duration:.3f}s")
            print(f"Embedding shape: {len(test_emb)} x {len(test_emb[0]) if test_emb else 0}")
        except Exception as e:
            print(f"Test embedding failed: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with memory management."""
        print(f"Generating embeddings for {len(texts)} texts...")
        
        self.gpu_manager.print_memory_stats("Before Embedding")
        self.gpu_manager.check_and_clear_if_needed()
        
        start = time.time()
        embeddings = self.embedder.embed_documents(texts)
        duration = time.time() - start
        
        print(f"Embeddings generated in {duration:.3f}s")
        self.gpu_manager.print_memory_stats("After Embedding")
        
        return embeddings
    
    def _embed_texts_safe(self, texts: List[str], batch_size: int = 16) -> List:
        """Embed texts with retry logic and memory management."""
        vectors = []
        
        # Adaptive batch size based on GPU memory
        if self.gpu_manager.is_available():
            stats = self.gpu_manager.get_memory_stats()
            if stats:
                total_memory = stats['total_gb']
                if total_memory < 40:
                    batch_size = min(batch_size, 4)
                elif total_memory < 80:
                    batch_size = min(batch_size, 8)
                else:
                    batch_size = min(batch_size, 12)
        
        print(f"Using embedding batch size: {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Check memory periodically
            if i > 0 and i % (batch_size * 3) == 0:
                self.gpu_manager.check_and_clear_if_needed()
            
            # Try embedding with retries
            success = False
            for attempt in range(3):
                try:
                    batch_vecs = self.generate_embeddings(batch)
                    vectors.extend(batch_vecs)
                    success = True
                    break
                except Exception as e:
                    print(f"Batch {i}-{i+len(batch)-1} attempt {attempt+1} failed: {e}")
                    self.gpu_manager.clear_cache()
                    time.sleep(1 + attempt * 2)
            
            if not success:
                # Fallback to individual documents
                print(f"Processing documents individually...")
                for j, doc in enumerate(batch):
                    try:
                        vec = self.generate_embeddings([doc])[0]
                        vectors.append(vec)
                    except Exception as e:
                        print(f"   Document {i+j} failed: {e}")
                        vectors.append(None)
                        self.gpu_manager.clear_cache()
        
        self.gpu_manager.clear_cache()
        return vectors
    
    def _enrich_metadata(self, metadata: dict, content: str) -> dict:
        """
        Enrich metadata with additional fields before uploading to Qdrant.
        Based on upload_text_fast.py fields.
        """
        enriched = metadata.copy()
        
        # Add incremental ID
        enriched["id"] = self.chunk_counter
        self.chunk_counter += 1
        
        # Rename source_url to url if it exists
        if "source_url" in enriched:
            enriched["url"] = enriched.pop("source_url")
        
        # Add scholarly metadata fields
        enriched["doi"] = None
        enriched["title"] = None
        enriched["journal"] = None
        enriched["publisher"] = enriched.get("source_json_file", "").replace(".json", "")
        enriched["reference_count"] = 0
        enriched["n_citations"] = 0
        enriched["influential_citation_count"] = 0
        enriched["header"] = []
        
        # Add content
        enriched["content"] = content
        
        return enriched
    
    def process_subset(self, chunk_subset: List[ChunkMetadata], local_dir: str = "downloads") -> tuple:
        """Process a subset of chunks: download, read, prepare for embedding."""
        # Download files
        s3_uris = [chunk.s3_uri for chunk in chunk_subset]
        downloaded_files = self.downloader.download_batch(s3_uris, local_dir)
        
        # Read content
        all_ids, all_texts, all_metadata, local_paths = [], [], [], []
        
        for chunk in chunk_subset:
            if chunk.s3_uri in downloaded_files:
                local_path = downloaded_files[chunk.s3_uri]
                try:
                    with open(local_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    
                    all_ids.append(chunk.uid)
                    all_texts.append(content)
                    all_metadata.append(chunk.metadata)
                    local_paths.append(local_path)
                except Exception as e:
                    print(f"Error reading {local_path}: {e}")
        
        return all_ids, all_texts, all_metadata, local_paths
    
    def upload_file(self, file_path: str):
        """Upload chunks from a single JSON file."""
        file_name = os.path.basename(file_path)
        
        # Parse file
        chunk_metadata, parse_stats = self.parser.parse_file(file_path)
        
        # Update stats
        self.stats.update_global(
            total=parse_stats["total"],
            skipped=parse_stats["skipped"],
            filtered=parse_stats["filtered"]
        )
        self.stats.init_file_stats(
            file_name,
            parse_stats["total"],
            parse_stats["skipped"],
            parse_stats["filtered"],
            parse_stats["processed"]
        )
        
        # Process in subsets
        subset_size = self.config.upload.subset_size
        batch_size = self.config.upload.batch_size
        
        total_subsets = (len(chunk_metadata) + subset_size - 1) // subset_size
        
        for subset_idx in range(0, len(chunk_metadata), subset_size):
            subset = chunk_metadata[subset_idx:subset_idx + subset_size]
            subset_num = subset_idx // subset_size + 1
            
            print(f"\nProcessing subset {subset_num}/{total_subsets}")
            
            try:
                # Download and read
                ids, texts, metadata, paths = self.process_subset(subset)
                
                if not texts:
                    print(f"   No texts downloaded, skipping")
                    continue
                
                # Generate embeddings
                vectors = self._embed_texts_safe(texts, batch_size=8)
                
                # Filter successful embeddings and enrich metadata
                good_items = []
                for uid, vec, meta, text in zip(ids, vectors, metadata, texts):
                    if vec is not None:
                        enriched_meta = self._enrich_metadata(meta, text)
                        good_items.append((uid, vec, enriched_meta))
                
                failed_items = [uid for uid, vec, _ in zip(ids, vectors, metadata) if vec is None]
                
                if failed_items:
                    self.stats.add_failed_ids(failed_items)
                    self.stats.update_global(failed=len(failed_items))
                    self.stats.update_file(file_name, processed=len(failed_items), failed=len(failed_items))
                
                if not good_items:
                    print(f"   No successful embeddings")
                    cleanup_files(paths)
                    continue
                
                # Upload in batches
                for batch_start in range(0, len(good_items), batch_size):
                    batch = good_items[batch_start:batch_start + batch_size]
                    batch_ids = [item[0] for item in batch]
                    batch_vectors = [item[1] for item in batch]
                    batch_meta = [item[2] for item in batch]
                    
                    success, failed_ids = self.uploader.upload_batch(batch_ids, batch_vectors, batch_meta)
                    
                    if success:
                        print(f"   Batch uploaded: {len(batch_ids)} chunks")
                        self.stats.update_global(uploaded=len(batch_ids), succeeded=len(batch_ids), processed=len(batch_ids))
                        self.stats.update_file(file_name, processed=len(batch_ids), succeeded=len(batch_ids))
                    else:
                        print(f"   Batch upload failed")
                        self.stats.add_failed_ids(failed_ids or batch_ids)
                        self.stats.update_global(failed=len(batch_ids), processed=len(batch_ids))
                        self.stats.update_file(file_name, processed=len(batch_ids), failed=len(batch_ids))
                    
                    self.stats.update_global(batches=1)
                    
                    if batch_start + batch_size < len(good_items):
                        time.sleep(0.5)
                
                # Cleanup
                cleanup_files(paths)
                self.gpu_manager.clear_cache()
                self.stats.write_stats()
                
            except Exception as e:
                print(f"   Error processing subset: {e}")
                self.gpu_manager.clear_cache()
                continue
    
    def upload_all(self):
        """Upload all chunks from the configured folder."""
        print(f"Starting upload process")
        print(f"   Pod ID: {self.pod_id}")
        print(f"   Collection: {self.config.database.collection_name}")
        print(f"   Batch size: {self.config.upload.batch_size}")
        print(f"   Subset size: {self.config.upload.subset_size}")
        
        # Check collection health
        if not self.uploader.check_collection_health():
            print("Collection health check failed")
            return
        
        self.gpu_manager.print_memory_stats("Initial")
        
        # Process all JSON files
        json_files = sorted([
            f for f in os.listdir(self.config.chunks_folder) 
            if f.endswith(".json")
        ])
        
        for file_name in json_files:
            print(f"\nProcessing file: {file_name}")
            full_path = os.path.join(self.config.chunks_folder, file_name)
            
            try:
                self.upload_file(full_path)
                print(f"Completed file {file_name}")
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                continue
            
            time.sleep(1.0)  # Delay between files
            self.gpu_manager.clear_cache()
        
        # Final summary
        print("\nAll uploads complete")
        self.gpu_manager.print_memory_stats("Final")
        self.stats.print_summary()
        self.stats.print_validation()
        self.stats.write_stats()
    
    def upload_from_s3(self, s3_path: str):
        """Upload chunks directly from S3 path without metadata files."""
        print(f"Starting direct S3 upload")
        print(f"   S3 path: {s3_path}")
        print(f"   Collection: {self.config.database.collection_name}")
        print(f"   Batch size: {self.config.upload.batch_size}")
        print(f"   Subset size: {self.config.upload.subset_size}")
        
        # Check collection health
        if not self.uploader.check_collection_health():
            print("Collection health check failed")
            return
        
        self.gpu_manager.print_memory_stats("Initial")
        
        # List all chunks from S3
        s3_handler = S3ChunkHandler()
        try:
            chunks = s3_handler.list_chunks_from_s3(s3_path)
        except Exception as e:
            print(f"Failed to list S3 files: {e}")
            return
        
        if not chunks:
            print("No chunk files found in S3 path")
            return
        
        print(f"Found {len(chunks)} chunk files")
        
        # Convert to ChunkMetadata objects
        chunk_metadata = [
            DirectS3ChunkMetadata(c['uid'], c['s3_uri'], c['metadata'])
            for c in chunks
        ]
        
        # Initialize stats
        file_name = f"direct_s3_{os.path.basename(s3_path.rstrip('/'))}"
        self.stats.init_file_stats(file_name, len(chunks), 0, 0, len(chunks))
        self.stats.update_global(total=len(chunks))
        
        # Process in subsets
        subset_size = self.config.upload.subset_size
        batch_size = self.config.upload.batch_size
        total_subsets = (len(chunk_metadata) + subset_size - 1) // subset_size
        
        for subset_idx in range(0, len(chunk_metadata), subset_size):
            subset = chunk_metadata[subset_idx:subset_idx + subset_size]
            subset_num = subset_idx // subset_size + 1
            
            print(f"\nProcessing subset {subset_num}/{total_subsets}")
            
            try:
                # Download and read
                ids, texts, metadata, paths = self.process_subset(subset)
                
                if not texts:
                    print(f"No texts downloaded, skipping")
                    continue
                
                # Generate embeddings
                vectors = self._embed_texts_safe(texts, batch_size=8)
                
                # Filter successful embeddings and enrich metadata
                good_items = []
                for uid, vec, meta, text in zip(ids, vectors, metadata, texts):
                    if vec is not None:
                        enriched_meta = self._enrich_metadata(meta, text)
                        good_items.append((uid, vec, enriched_meta))
                
                failed_items = [uid for uid, vec, _ in zip(ids, vectors, metadata) if vec is None]
                
                if failed_items:
                    self.stats.add_failed_ids(failed_items)
                    self.stats.update_global(failed=len(failed_items))
                    self.stats.update_file(file_name, processed=len(failed_items), failed=len(failed_items))
                
                if not good_items:
                    print(f"No successful embeddings")
                    cleanup_files(paths)
                    continue
                
                # Upload in batches
                for batch_start in range(0, len(good_items), batch_size):
                    batch = good_items[batch_start:batch_start + batch_size]
                    batch_ids = [item[0] for item in batch]
                    batch_vectors = [item[1] for item in batch]
                    batch_meta = [item[2] for item in batch]
                    
                    success, failed_ids = self.uploader.upload_batch(batch_ids, batch_vectors, batch_meta)
                    
                    if success:
                        print(f"Batch uploaded: {len(batch_ids)} chunks")
                        self.stats.update_global(uploaded=len(batch_ids), succeeded=len(batch_ids), processed=len(batch_ids))
                        self.stats.update_file(file_name, processed=len(batch_ids), succeeded=len(batch_ids))
                    else:
                        print(f"Batch upload failed")
                        self.stats.add_failed_ids(failed_ids or batch_ids)
                        self.stats.update_global(failed=len(batch_ids), processed=len(batch_ids))
                        self.stats.update_file(file_name, processed=len(batch_ids), failed=len(batch_ids))
                    
                    self.stats.update_global(batches=1)
                    
                    if batch_start + batch_size < len(good_items):
                        time.sleep(0.5)
                
                # Cleanup
                cleanup_files(paths)
                self.gpu_manager.clear_cache()
                self.stats.write_stats()
                
            except Exception as e:
                print(f"Error processing subset: {e}")
                self.gpu_manager.clear_cache()
                continue
        
        # Final summary
        print("\nS3 upload complete")
        self.gpu_manager.print_memory_stats("Final")
        self.stats.print_summary()
        self.stats.print_validation()
        self.stats.write_stats()

