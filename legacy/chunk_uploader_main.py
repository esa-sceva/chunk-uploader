import os
import json
from datetime import datetime
import boto3
from tqdm import tqdm
import hashlib
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any
from qdrant_client import QdrantClient, models
import yaml
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import gc  # Add garbage collection for memory management

def load_config(path: str = "config_qwen.yaml") -> dict:
    """Loads a YAML configuration file into a Python dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def last_token_pool(last_hidden_states, attention_mask):
    """Pool last token from transformer outputs."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class QwenSentenceTransformerEmbedder:
    """Qwen embedder using sentence-transformers."""
    def __init__(self, model_name="Qwen/Qwen3-Embedding-4B"):
        try:
            print(f"🔄 Loading {model_name} with sentence-transformers...")
            self.model = SentenceTransformer(
                model_name,
                model_kwargs={
                    "torch_dtype": "auto",
                    "device_map": "auto",
                },
                tokenizer_kwargs={
                    "padding_side": "left",
                    "max_length": 2048,
                    "truncation": True
                }
            )
            print(f"✅ Successfully loaded {model_name}")
        except Exception as e:
            print(f"❌ Failed to load {model_name} with sentence-transformers: {e}")
            print("💡 Try: pip install sentence-transformers>=2.7.0 transformers>=4.51.0")
            raise e

    def embed_documents(self, texts, batch_size=8, normalize=True):
        """Encode texts into embeddings."""
        try:
            print(f"🔍 SentenceTransformer: Processing {len(texts)} texts with batch_size={batch_size}")
            
            # Check device before encoding
            if hasattr(self.model, 'device'):
                print(f"🔍 SentenceTransformer model device: {self.model.device}")
            else:
                # Check device of first module
                try:
                    first_module = next(iter(self.model._modules.values()))
                    for name, param in first_module.named_parameters():
                        print(f"🔍 SentenceTransformer parameter {name} on device: {param.device}")
                        break
                except:
                    print(f"🔍 Could not determine SentenceTransformer device")
            
            # Monitor GPU utilization during encoding
            if torch.cuda.is_available():
                print(f"🎮 GPU before SentenceTransformer encode:")
                print(f"    Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
                print(f"    Reserved: {torch.cuda.memory_reserved() / 1024**3:.1f}GB")
            
            start_time = time.time()
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                convert_to_tensor=False,
                show_progress_bar=True  # Add progress bar to see activity
            )
            encode_time = time.time() - start_time
            
            print(f"🎯 SentenceTransformer encode completed in {encode_time:.3f}s")
            
            # Monitor GPU after encoding
            if torch.cuda.is_available():
                print(f"🎮 GPU after SentenceTransformer encode:")
                print(f"    Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
                print(f"    Reserved: {torch.cuda.memory_reserved() / 1024**3:.1f}GB")
                
                # Brief delay to allow monitoring
                print(f"⏳ Waiting 1s before cleanup to allow GPU monitoring...")
                time.sleep(1)
            
            # Clear GPU cache after embedding generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                print(f"🧹 GPU cache cleared after SentenceTransformer")
                
            return embeddings.tolist()
        except Exception as e:
            print(f"❌ SentenceTransformer embedding failed: {e}")
            # Clear cache on error too
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            raise e

class QwenTransformerEmbedder:
    """Qwen embedder using transformers directly."""
    def __init__(self, model_name="Qwen/Qwen3-Embedding-4B", max_length=2048):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )
        self.max_length = max_length

    def embed_documents(self, texts, batch_size=8, normalize=True):
        """Encode texts into embeddings using transformers."""
        all_embeddings = []
        
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_dict = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.model.device)

                with torch.no_grad():
                    outputs = self.model(**batch_dict)
                    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                    if normalize:
                        embeddings = F.normalize(embeddings, p=2, dim=1)

                    all_embeddings.extend(embeddings.cpu().tolist())
                    
                    # Clear intermediate tensors
                    del outputs, embeddings, batch_dict
                    
                # Clear cache every few batches
                if (i // batch_size) % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    
        except Exception as e:
            # Clear cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            raise e
        finally:
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
        return all_embeddings

def load_hf_embeddings(model_name: str, model_type: str = 'sentence', normalize: bool = True):
    """Load embedding model based on configuration."""
    try:
        if "nasa-impact/nasa-smd-ibm-st-v2" in model_name:
            encode_kwargs = {"normalize_embeddings": normalize}
            print('NASA SMD model loaded')
            embedder = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)
            return embedder, 768  # NASA model has 768 dimensions
            
        elif "Qwen/Qwen3-Embedding" in model_name or "Qwen/Qwen2-Embedding" in model_name:
            print(f'Qwen model loaded: {model_name} with type: {model_type}')
            if model_type == 'sentence':
                embedder = QwenSentenceTransformerEmbedder(model_name=model_name)
            elif model_type == 'transformer':
                embedder = QwenTransformerEmbedder(model_name=model_name)
            else:
                raise ValueError('Qwen model type must be "sentence" or "transformer"')
            
            # Determine vector size based on model name
            if "Qwen3-Embedding-0.6B" in model_name:
                vector_size = 1024
            elif "Qwen3-Embedding-4B" in model_name:
                vector_size = 2560
            elif "Qwen3-Embedding-8B" in model_name:
                vector_size = 4096
            elif "Qwen2-Embedding" in model_name:
                vector_size = 1024  # Default for Qwen2
            else:
                vector_size = 2560  # Default fallback
                
            return embedder, vector_size
            
        elif "indus-sde-st-v0.2" in model_name:
            encode_kwargs = {"normalize_embeddings": normalize}
            print('Indus model loaded')
            embedder = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)
            return embedder, 768  # Indus model has 768 dimensions
        else:
            raise ValueError(f'Unsupported model: {model_name}')
            
    except Exception as e:
        print(f"❌ Error loading embedding model {model_name}: {e}")
        print("💡 Suggestions:")
        print("   1. Check internet connectivity")
        print("   2. Verify model name exists on HuggingFace")
        print("   3. Try clearing HuggingFace cache: rm -rf ~/.cache/huggingface/")
        print("   4. Check transformers version: pip install transformers>=4.51.0")
        print("   5. Check sentence-transformers version: pip install sentence-transformers>=2.7.0")
        raise e

#
class MinimalQdrantUploader:
    def __init__(self, config_path: str = "config_qwen.yaml", 
                 chunks_folder: str = "/workspace/chunks_with_metadata",
                 score_threshold: float = 0.0,
                 skip_first_chunks: int = 0):
        # Load configuration
        self.config = load_config(config_path)
        
        # Score threshold for filtering chunks and skip count
        self.score_threshold = score_threshold
        self.skip_first_chunks = skip_first_chunks
        
        # Extract config values
        database_config = self.config["database"]
        embedding_config = self.config["embedding"]
        upload_config = self.config["upload_params"]
        
        self.qdrant_url = ""
        self.qdrant_api_key = ""
        self.collection_name = database_config["collection_name"]
        self.chunks_folder = chunks_folder
        
        # Upload parameters from config
        self.batch_size = upload_config.get("batch_size", 48)
        self.vector_size = upload_config.get("vector_size", 4096)
        self.subset_size = 384  # Keep for concurrent processing
        self.download_threads = 8
        
        # Enhanced Qdrant client configuration for multi-pod scenarios
        self.client = QdrantClient(
            url=self.qdrant_url, 
            api_key=self.qdrant_api_key,
            timeout=60.0,  # Increase timeout to 60 seconds
            prefer_grpc=False,  # Use HTTP instead of gRPC for better timeout handling
        )
        self.s3_client = boto3.client("s3")
        
        # Add retry configuration
        self.max_retries = 3
        self.retry_delay_base = 2  # Base delay in seconds
        
        # Reduce batch sizes for multi-pod scenarios AND memory management
        if self.batch_size > 24:
            self.batch_size = 24  # Reduce from default for multi-pod
            print(f"📉 Reduced batch size to {self.batch_size} for multi-pod scenario")
        if self.subset_size > 96:  # Further reduce for memory management
            self.subset_size = 96  # Reduce subset size for memory management
            print(f"📉 Reduced subset size to {self.subset_size} for memory management")
        
        # Add memory monitoring
        self.gpu_memory_threshold = 0.85  # 85% GPU memory threshold
        
        # Track global chunk counter for skipping
        self.global_chunk_counter = 0
        self.chunks_skipped = 0
        
        print(f"🚀 Uploader Configuration:")
        print(f"   Skip first chunks: {self.skip_first_chunks}")
        print(f"   Score threshold: {self.score_threshold}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Subset size: {self.subset_size}")

        # Initialize embedding model from config
        print(f"🤖 Loading embedding model from config...")
        print(f"🔍 CUDA available during model loading: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🔍 GPU count: {torch.cuda.device_count()}")
            print(f"🔍 Current device: {torch.cuda.current_device()}")
            print(f"🔍 GPU name: {torch.cuda.get_device_name()}")
        
        self.embedder, self.actual_vector_size = load_hf_embeddings(
            embedding_config["model_name"],
            embedding_config.get("type", "sentence")
        )
        
        print(f"✅ Embedding model loaded: {embedding_config['model_name']}")
        print(f"📊 Config vector dimensions: {self.vector_size}")
        print(f"📊 Actual vector dimensions: {self.actual_vector_size}")
        
        # Verify model device after loading
        if hasattr(self.embedder, 'model'):
            try:
                if hasattr(self.embedder.model, 'device'):
                    print(f"🔍 Final embedder device: {self.embedder.model.device}")
                else:
                    # For SentenceTransformer, check internal modules
                    first_module = next(iter(self.embedder.model._modules.values()))
                    for name, param in first_module.named_parameters():
                        print(f"🔍 Final embedder parameter {name} device: {param.device}")
                        break
            except Exception as e:
                print(f"🔍 Could not verify embedder device: {e}")
        
        # Test a small embedding to verify GPU is working
        if torch.cuda.is_available():
            print(f"🧪 Testing GPU embedding with sample text...")
            try:
                test_start = time.time()
                test_embedding = self.embedder.embed_documents(["This is a test sentence to verify GPU usage."])
                test_time = time.time() - test_start
                print(f"✅ Test embedding successful in {test_time:.3f}s")
                print(f"🔍 Test embedding shape: {len(test_embedding)} x {len(test_embedding[0]) if test_embedding else 0}")
            except Exception as e:
                print(f"❌ Test embedding failed: {e}")
        else:
            print(f"⚠️ CUDA not available - embeddings will run on CPU")
        
        # Concurrent processing queues
        self.download_queue = queue.Queue()
        self.embedding_queue = queue.Queue()
        self.upload_queue = queue.Queue()
        self.downloaded_files = {}  # Track downloaded files for cleanup

    def generate_embeddings(self, texts):
        """Generate embeddings using the configured model."""
        try:
            # Debug: Check model device and GPU availability
            print(f"🔍 DEBUG: CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"🔍 DEBUG: Current CUDA device: {torch.cuda.current_device()}")
                print(f"🔍 DEBUG: Device count: {torch.cuda.device_count()}")
            
            # Check embedder device if possible
            if hasattr(self.embedder, 'model'):
                if hasattr(self.embedder.model, 'device'):
                    print(f"🔍 DEBUG: Embedder model device: {self.embedder.model.device}")
                elif hasattr(self.embedder.model, '_modules'):
                    # For SentenceTransformer, try to get device from internal model
                    try:
                        first_module = next(iter(self.embedder.model._modules.values()))
                        if hasattr(first_module, 'device'):
                            print(f"🔍 DEBUG: SentenceTransformer device: {first_module.device}")
                        else:
                            print(f"🔍 DEBUG: Checking SentenceTransformer module parameters...")
                            for name, param in first_module.named_parameters():
                                print(f"🔍 DEBUG: Parameter {name} device: {param.device}")
                                break
                    except Exception as e:
                        print(f"🔍 DEBUG: Could not determine SentenceTransformer device: {e}")
            
            print(f"🧠 Starting embedding generation for {len(texts)} texts...")
            
            # Monitor GPU memory before embedding
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                
                print(f"🎮 Pre-embedding GPU: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
                
                memory_usage = memory_reserved / total_memory
                if memory_usage > self.gpu_memory_threshold:
                    print(f"⚠️ High GPU memory usage: {memory_usage:.2%} ({memory_reserved:.1f}GB/{total_memory:.1f}GB)")
                    print(f"🧹 Clearing GPU cache before embedding...")
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Use the embed_documents method from the loaded embedder
            start_embed_time = time.time()
            embeddings = self.embedder.embed_documents(texts)
            embed_duration = time.time() - start_embed_time
            
            print(f"✅ Embedding completed in {embed_duration:.3f}s for {len(texts)} texts")
            
            # Monitor GPU memory after embedding
            if torch.cuda.is_available():
                memory_allocated_after = torch.cuda.memory_allocated() / 1024**3
                memory_reserved_after = torch.cuda.memory_reserved() / 1024**3
                print(f"🎮 Post-embedding GPU: {memory_allocated_after:.1f}GB allocated, {memory_reserved_after:.1f}GB reserved")
                
                # Don't clear cache immediately - let user see GPU usage
                print(f"⏳ Keeping GPU cache for 2 seconds to allow monitoring...")
                time.sleep(2)  # Give time to see GPU usage
                
                torch.cuda.empty_cache()
                gc.collect()
                print(f"🧹 GPU cache cleared")
                
            return embeddings
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            # Emergency cleanup on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            raise e

    def get_embedding_dimension(self):
        """Get the embedding dimension from config."""
        return self.vector_size

    def download_from_s3(self, s3_uri: str, local_dir="downloads") -> str:
        """Download a file from S3 and return local path."""
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

    def download_batch_concurrent(self, s3_uris: List[str], local_dir="downloads") -> Dict[str, str]:
        """Download multiple files from S3 concurrently."""
        downloaded_files = {}
        
        def download_single(s3_uri):
            local_path = self.download_from_s3(s3_uri, local_dir)
            if local_path:
                downloaded_files[s3_uri] = local_path
            return s3_uri, local_path
        
        with ThreadPoolExecutor(max_workers=self.download_threads) as executor:
            futures = {executor.submit(download_single, uri): uri for uri in s3_uris}
            
            for future in as_completed(futures):
                s3_uri = futures[future]
                try:
                    uri, local_path = future.result()
                    if local_path:
                        print(f"⬇️ Downloaded {uri} -> {local_path}")
                except Exception as e:
                    print(f"⚠️ Error downloading {s3_uri}: {e}")
        
        return downloaded_files

    def process_json_file(self, file_path: str):
        """Extract IDs, content, and metadata from a single JSON file."""
        file_name = os.path.basename(file_path)
        sub_folder = os.path.splitext(file_name)[0]  # e.g., "arxiv"

        print(f"📂 Processing file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle the new array format
        chunk_metadata = []
        total_chunks_count = 0
        filtered_chunks_count = 0
        skipped_chunks_count = 0
        
        # Check if data is a list (new format)
        if isinstance(data, list):
            print(f"   Detected array format with {len(data)} items")
            print(f"   Global chunk counter starts at: {self.global_chunk_counter}")
            print(f"   Will skip chunks until reaching: {self.skip_first_chunks}")
            
            # New array format - each item is a chunk object
            # Process chunks SERIALLY (one by one) in order
            for chunk_index, chunk in enumerate(data):
                if isinstance(chunk, dict) and "output_path" in chunk:
                    total_chunks_count += 1
                    self.global_chunk_counter += 1
                    
                    # Skip first N chunks if specified
                    if self.global_chunk_counter <= self.skip_first_chunks:
                        skipped_chunks_count += 1
                        if skipped_chunks_count <= 5 or skipped_chunks_count % 500 == 0:  # Log first 5 and every 500th
                            print(f"   ⏭️ Skipping chunk {self.global_chunk_counter}/{self.skip_first_chunks}: {chunk.get('chunk_name', 'unknown')}")
                        continue
                    
                    # Filter by score threshold
                    score = chunk.get("score", 0)
                    if score < self.score_threshold:
                        filtered_chunks_count += 1
                        continue
                        
                    output_path = chunk["output_path"]
                    s3_uri = f"s3://esa-satcom-s3/{output_path}"
                    
                    # Extract information from the chunk object
                    chunk_name = chunk.get("chunk_name", os.path.basename(output_path))
                    # Ensure chunk_name has .md extension
                    if not chunk_name.endswith('.md'):
                        chunk_name = chunk_name + '.md'
                    document_id = chunk.get("document_id", "unknown_doc")
                    source = chunk.get("source", sub_folder)
                    
                    # Use document_id as original file name
                    original_file_name = f"{document_id}.pdf"

                    # Collect metadata using the same format as before
                    meta = {
                        "source_url": chunk.get("source_url", ""),
                        "score": chunk.get("score", 0),
                        "original_file_name": original_file_name,
                        "sub_folder": sub_folder,
                        "chunk_name": chunk_name,
                        "source_json_file": file_name,
                    }

                    # Create unique ID using chunk_name or fallback
                    uid = chunk_name if chunk_name else f"{document_id}_{total_chunks_count}"
                    
                    chunk_metadata.append({
                        "uid": uid,
                        "s3_uri": s3_uri,
                        "metadata": meta
                    })
        else:
            # Handle old nested JSON structure (keeping backward compatibility)
            # Extract the top-level key (e.g., "acig_journal")
            if len(data) == 1:
                top_level_key = list(data.keys())[0]
                main_data = data[top_level_key]
                
                # Check if this has the new nested structure
                if "documents" in main_data:
                    # New nested structure
                    documents = main_data["documents"]
                    
                    for doc_id, doc_info in documents.items():
                        original_filename = doc_info.get("original_filename", f"{doc_id}.md")
                        chunks = doc_info.get("chunks", {})
                        
                        for chunk_id, chunk in chunks.items():
                            total_chunks_count += 1
                            self.global_chunk_counter += 1
                            
                            # Skip first N chunks if specified
                            if self.global_chunk_counter <= self.skip_first_chunks:
                                skipped_chunks_count += 1
                                if skipped_chunks_count <= 5 or skipped_chunks_count % 500 == 0:
                                    print(f"   ⏭️ Skipping chunk {self.global_chunk_counter}/{self.skip_first_chunks}: {chunk_id}")
                                continue
                            
                            # Filter by score threshold
                            score = chunk.get("score", 0)
                            if score < self.score_threshold:
                                filtered_chunks_count += 1
                                continue
                                
                            output_path = chunk["output_path"]
                            s3_uri = f"s3://esa-satcom-s3/{output_path}"
                            
                            # Extract chunk name
                            chunk_name = os.path.basename(output_path)
                            # Ensure chunk_name has .md extension
                            if not chunk_name.endswith('.md'):
                                chunk_name = chunk_name + '.md'
                            
                            # Use the doc_id from the document structure
                            original_file_name = original_filename

                            # Collect metadata (keeping exact same format as before)
                            meta = {
                                "source_url": chunk.get("source_url", ""),
                                "score": chunk.get("score", 0),
                                "original_file_name": original_file_name,
                                "sub_folder": sub_folder,
                                "chunk_name": chunk_name,
                                "source_json_file": file_name,
                            }

                            uid = f"{doc_id}_{chunk_id}"
                            
                            chunk_metadata.append({
                                "uid": uid,
                                "s3_uri": s3_uri,
                                "metadata": meta
                            })
                else:
                    # Fallback: treat as old flat structure
                    for chunk_id, chunk in main_data.items():
                        if isinstance(chunk, dict) and "output_path" in chunk:
                            total_chunks_count += 1
                            self.global_chunk_counter += 1
                            
                            # Skip first N chunks if specified
                            if self.global_chunk_counter <= self.skip_first_chunks:
                                skipped_chunks_count += 1
                                if skipped_chunks_count <= 5 or skipped_chunks_count % 500 == 0:
                                    print(f"   ⏭️ Skipping chunk {self.global_chunk_counter}/{self.skip_first_chunks}: {chunk_id}")
                                continue
                            
                            # Filter by score threshold
                            score = chunk.get("score", 0)
                            if score < self.score_threshold:
                                filtered_chunks_count += 1
                                continue
                                
                            output_path = chunk["output_path"]
                            s3_uri = f"s3://esa-satcom-s3/{output_path}"
                            
                            # Extract chunk name
                            chunk_name = os.path.basename(output_path)
                            # Ensure chunk_name has .md extension
                            if not chunk_name.endswith('.md'):
                                chunk_name = chunk_name + '.md'
                            
                            # Derive original_file_name from doc_id in output_path
                            doc_id = chunk.get("doc_id")
                            if not doc_id:
                                # fallback: try to extract from path
                                parts = output_path.split("/")
                                doc_id = parts[1] if len(parts) > 1 else "unknown_doc"
                            original_file_name = f"{doc_id}.md"

                            # Collect metadata
                            meta = {
                                "source_url": chunk.get("source_url", ""),
                                "score": chunk.get("score", 0),
                                "original_file_name": original_file_name,
                                "sub_folder": sub_folder,
                                "chunk_name": chunk_name,
                                "source_json_file": file_name,
                            }

                            uid = f"{doc_id}_{chunk_id}"
                            
                            chunk_metadata.append({
                                "uid": uid,
                                "s3_uri": s3_uri,
                                "metadata": meta
                            })
            else:
                # Old flat structure - direct chunk mapping OR new MDPI format with chunk names as keys
                for chunk_id, chunk in data.items():
                    if isinstance(chunk, dict):
                        total_chunks_count += 1
                        self.global_chunk_counter += 1
                        
                        # Skip first N chunks if specified
                        if self.global_chunk_counter <= self.skip_first_chunks:
                            skipped_chunks_count += 1
                            if skipped_chunks_count <= 5 or skipped_chunks_count % 500 == 0:
                                print(f"   ⏭️ Skipping chunk {self.global_chunk_counter}/{self.skip_first_chunks}: {chunk_id}")
                            continue
                        
                        # Check if this is the new MDPI format (has doc_id, output_path, score, source_url)
                        if "doc_id" in chunk and "output_path" in chunk and "score" in chunk and "source_url" in chunk:
                            # New MDPI format - extract metadata according to requirements
                            if total_chunks_count - skipped_chunks_count == 1:  # Print only once per file (excluding skipped)
                                print(f"   Detected new MDPI format with chunk names as keys")
                            
                            score = chunk.get("score", 0)
                            if score < self.score_threshold:
                                filtered_chunks_count += 1
                                continue
                            
                            output_path = chunk["output_path"]
                            
                            # Extract sub_folder from output_path
                            # e.g., "chunks/mdpi_1/5485f861-7a1c-4eed-bef3-899935e81af9/chunk_5485f861-7a1c-4eed-bef3-899935e81af9_4.md"
                            path_parts = output_path.split("/")
                            extracted_sub_folder = path_parts[1] if len(path_parts) > 1 else sub_folder
                            
                            # Build S3 URI
                            s3_uri = f"s3://esa-satcom-s3/{output_path}"
                            
                            # Build metadata according to the mapping:
                            # source_url -> source_url (from json)
                            # original_file_name -> doc_id + .md
                            # score -> score (from json)
                            # source_json_file -> current json filename
                            # sub_folder -> extracted from output_path
                            meta = {
                                "source_url": chunk.get("source_url", ""),
                                "original_file_name": chunk.get("doc_id", "") + ".md",
                                "score": score,
                                "source_json_file": file_name,
                                "sub_folder": extracted_sub_folder,
                                "chunk_name": chunk_id + ".md",  # Use the chunk name from the key with .md extension
                            }

                            uid = chunk_id  # Use chunk name as UID
                            
                            chunk_metadata.append({
                                "uid": uid,
                                "s3_uri": s3_uri,
                                "metadata": meta
                            })
                        
                        elif "output_path" in chunk:
                            # Old format with output_path
                            # Filter by score threshold
                            score = chunk.get("score", 0)
                            if score < self.score_threshold:
                                filtered_chunks_count += 1
                                continue
                                
                            output_path = chunk["output_path"]
                            s3_uri = f"s3://esa-satcom-s3/{output_path}"
                            
                            # Extract chunk name
                            chunk_name = os.path.basename(output_path)
                            # Ensure chunk_name has .md extension
                            if not chunk_name.endswith('.md'):
                                chunk_name = chunk_name + '.md'
                            
                            # Derive original_file_name from doc_id in output_path
                            doc_id = chunk.get("doc_id")
                            if not doc_id:
                                # fallback: try to extract from path
                                parts = output_path.split("/")
                                doc_id = parts[1] if len(parts) > 1 else "unknown_doc"
                            original_file_name = f"{doc_id}.md"

                            # Collect metadata
                            meta = {
                                "source_url": chunk.get("source_url", ""),
                                "score": chunk.get("score", 0),
                                "original_file_name": original_file_name,
                                "sub_folder": sub_folder,
                                "chunk_name": chunk_name,
                                "source_json_file": file_name,
                            }

                            uid = f"{doc_id}_{chunk_id}"
                            
                            chunk_metadata.append({
                                "uid": uid,
                                "s3_uri": s3_uri,
                                "metadata": meta
                            })

        print(f"   Found {total_chunks_count} total chunks in {sub_folder}")
        print(f"   Skipped {skipped_chunks_count} chunks (first {self.skip_first_chunks} chunks)")
        print(f"   Filtered {filtered_chunks_count} chunks below score threshold {self.score_threshold}")
        print(f"   Remaining {len(chunk_metadata)} chunks for processing")
        
        # Update global tracking
        self.chunks_skipped += skipped_chunks_count
        
        return chunk_metadata, {
            "total": total_chunks_count, 
            "skipped": skipped_chunks_count,
            "filtered": filtered_chunks_count, 
            "processed": len(chunk_metadata)
        }

    def _embed_texts_safe(self, texts: List[str], embed_batch_size: int = 16):  # Increase from 8 to 16 for more visible GPU usage
        """Embed texts in small batches with retries and memory management."""
        vectors = []
        
        # Adaptive batch size based on GPU memory
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            if total_memory < 40:  # Less than 40GB VRAM
                embed_batch_size = min(embed_batch_size, 4)
            elif total_memory < 80:  # Less than 80GB VRAM
                embed_batch_size = min(embed_batch_size, 8)
            else:  # 80GB+ VRAM
                embed_batch_size = min(embed_batch_size, 12)
        
        print(f"   🧠 Using embedding batch size: {embed_batch_size}")
        
        for i in range(0, len(texts), embed_batch_size):
            batch = texts[i:i+embed_batch_size]
            success = False
            
            # Monitor memory before each batch
            if torch.cuda.is_available() and i > 0 and i % (embed_batch_size * 3) == 0:
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                memory_usage = memory_reserved / total_memory
                
                if memory_usage > self.gpu_memory_threshold:
                    print(f"   🧹 Memory cleanup at batch {i//embed_batch_size}: {memory_usage:.2%}")
                    torch.cuda.empty_cache()
                    gc.collect()
            
            for attempt in range(3):
                try:
                    batch_vecs = self.generate_embeddings(batch)
                    # verify length
                    if len(batch_vecs) != len(batch):
                        print(f"   ⚠️ embed returned {len(batch_vecs)} vectors for {len(batch)} inputs")
                    vectors.extend(batch_vecs)
                    success = True
                    break
                except torch.cuda.OutOfMemoryError as e:
                    print(f"   🚨 CUDA OOM at batch {i}-{i+len(batch)-1}, attempt {attempt+1}: {e}")
                    # Aggressive cleanup on OOM
                    torch.cuda.empty_cache()
                    gc.collect()
                    time.sleep(2)  # Give GPU time to release memory
                    if attempt == 2:  # Last attempt, reduce batch size
                        print(f"   🔄 Falling back to individual embedding for batch {i}-{i+len(batch)-1}")
                        break
                except Exception as e:
                    print(f"   ⚠️ embed batch {i}-{i+len(batch)-1} attempt {attempt+1} failed: {e}")
                    time.sleep(1 + attempt*2)
                    
            if not success:
                # fallback to per-doc embedding to isolate problem docs
                print(f"   🔄 Processing individual documents for batch {i}-{i+len(batch)-1}")
                for j, doc in enumerate(batch):
                    try:
                        vec = self.generate_embeddings([doc])[0]
                        vectors.append(vec)
                    except Exception as e2:
                        print(f"      ❌ single doc embed failed at global index {i+j}: {e2}")
                        vectors.append(None)
                        # Clear cache after each failed doc
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        return vectors

    def process_subset_concurrent(self, chunk_subset: List[Dict], local_dir="downloads"):
        """Process a subset of chunks concurrently: download -> read content -> prepare for embedding."""
        # Step 1: Download all files concurrently
        s3_uris = [chunk["s3_uri"] for chunk in chunk_subset]
        downloaded_files = self.download_batch_concurrent(s3_uris, local_dir)
        
        # Step 2: Read content and prepare data
        all_ids, all_chunks, all_metadata, local_paths = [], [], [], []
        
        for chunk in chunk_subset:
            s3_uri = chunk["s3_uri"]
            if s3_uri in downloaded_files:
                local_path = downloaded_files[s3_uri]
                try:
                    with open(local_path, "r", encoding="utf-8") as cf:
                        content = cf.read().strip()
                    
                    all_ids.append(chunk["uid"])
                    all_chunks.append(content)
                    all_metadata.append(chunk["metadata"])
                    local_paths.append(local_path)
                except Exception as e:
                    print(f"⚠️ Error reading {local_path}: {e}")
        
        return all_ids, all_chunks, all_metadata, local_paths

    def upload_batch_with_retry(self, batch_points, batch_ids):
        """Upload a batch with exponential backoff retry logic."""
        for attempt in range(self.max_retries):
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
                return True, None
            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle specific timeout and connection errors
                if any(keyword in error_msg for keyword in ['timeout', 'timed out', 'connection', 'network', 'write timed out']):
                    if attempt < self.max_retries - 1:
                        # Exponential backoff with jitter
                        delay = self.retry_delay_base * (2 ** attempt) + np.random.uniform(0, 2)
                        print(f"   ⏳ Upload timeout/connection error (attempt {attempt + 1}/{self.max_retries})")
                        print(f"   ⏰ Retrying in {delay:.2f}s... Error: {e}")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"   ❌ Final upload attempt failed after {self.max_retries} retries: {e}")
                        return False, batch_ids
                else:
                    # Non-timeout error, don't retry
                    print(f"   ❌ Upload failed with non-retryable error: {e}")
                    return False, batch_ids
        
        return False, batch_ids

    def check_collection_health(self):
        """Check if collection is accessible and get current point count."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            point_count = collection_info.points_count
            print(f"📊 Collection '{self.collection_name}' status: {point_count} points")
            return True
        except Exception as e:
            print(f"❌ Collection access error: {e}")
            return False

    def print_gpu_memory_stats(self, stage=""):
        """Print current GPU memory usage statistics."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            memory_cached = torch.cuda.memory_cached() / 1024**3       # GB
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            usage_percent = (memory_reserved / total_memory) * 100
            
            stage_prefix = f"[{stage}] " if stage else ""
            print(f"   🎮 {stage_prefix}GPU Memory: {memory_allocated:.1f}GB allocated, "
                  f"{memory_reserved:.1f}GB reserved, {memory_cached:.1f}GB cached "
                  f"({usage_percent:.1f}% of {total_memory:.1f}GB)")
            
            # Warning if memory usage is high
            if usage_percent > 85:
                print(f"   ⚠️ High GPU memory usage detected!")
                return True
        return False

    @staticmethod
    def string_to_uint(s: str) -> int:
        hash_bytes = hashlib.sha256(s.encode("utf-8")).digest()
        return int.from_bytes(hash_bytes[:8], byteorder="big", signed=False)

    def upload_to_qdrant(self):
        """Concurrent upload chunks with embeddings to Qdrant."""
        print(f"🚀 Preparing concurrent upload to Qdrant collection `{self.collection_name}`")
        print(f"   Subset size: {self.subset_size} chunks")
        print(f"   Download threads: {self.download_threads}")
        print(f"   Upload batch size: {self.batch_size}")
        print(f"   Max retries per batch: {self.max_retries}")
        
        # Add pod identification for multi-pod scenarios
        pod_id = os.environ.get('POD_ID', f'pod_{np.random.randint(1000, 9999)}')
        print(f"🔧 Running as Pod ID: {pod_id}")

        # Stats tracking
        total_chunks = 0
        total_filtered = 0
        total_batches = 0
        processed = 0
        succeeded = 0
        failed = 0
        failed_ids = []
        per_file_stats = {}
        total_uploaded_to_qdrant = 0  # Track successful Qdrant uploads
        stats_started_at = datetime.now().isoformat() + "Z"
        stats_path = os.path.abspath(f"upload_stats_{pod_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        def write_stats():
            stats = {
                "timestamp": datetime.now().isoformat() + "Z",
                "started_at": stats_started_at,
                "collection_name": self.collection_name,
                "pod_id": pod_id,
                "score_threshold": self.score_threshold,
                "subset_size": self.subset_size,
                "download_threads": self.download_threads,
                "batch_size": self.batch_size,
                "total_chunks": total_chunks,
                "total_filtered": total_filtered,
                "total_available_for_processing": total_chunks - total_filtered,
                "total_uploaded_to_qdrant": total_uploaded_to_qdrant,
                "upload_success_rate": (total_uploaded_to_qdrant / max(1, total_chunks - total_filtered)) * 100,
                "total_batches": total_batches,
                "processed": processed,
                "succeeded": succeeded,
                "failed": failed,
                "failed_ids": failed_ids,
                "per_file_stats": per_file_stats,
            }
            try:
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e:
                print(f"⚠️ Could not write stats file: {e}")

        def cleanup_files(file_paths: List[str]):
            """Clean up downloaded files."""
            for p in file_paths:
                try:
                    if os.path.isfile(p):
                        os.remove(p)
                except Exception as e:
                    print(f"⚠️ Could not delete {p}: {e}")

        try:
            # Check collection health before starting
            print("🔍 Checking collection health...")
            if not self.check_collection_health():
                print("❌ Collection health check failed, aborting upload")
                return
            
            # Print initial GPU memory stats
            self.print_gpu_memory_stats("Initial")
            
            # Process all JSON files
            for file_name in sorted(os.listdir(self.chunks_folder)):
                if not file_name.endswith(".json"):
                    continue

                print(f"📂 Processing file: {file_name}")
                full_path = os.path.join(self.chunks_folder, file_name)
                
                # Get all chunk metadata from the JSON file
                chunk_metadata, filter_stats = self.process_json_file(full_path)
                file_total = filter_stats["total"]
                file_skipped = filter_stats["skipped"]
                file_filtered = filter_stats["filtered"]
                file_processed = filter_stats["processed"]
                
                total_chunks += file_total
                total_filtered += file_filtered
                
                # Track global skip count
                total_skipped = getattr(self, 'total_skipped', 0) + file_skipped
                self.total_skipped = total_skipped
                
                # Ensure entry exists with filtering stats
                per_file_stats[file_name] = {
                    "total_chunks": file_total,
                    "skipped_chunks": file_skipped,
                    "filtered_chunks": file_filtered,
                    "available_for_processing": file_processed,
                    "processed": 0, 
                    "succeeded": 0, 
                    "failed": 0
                }

                print(f"   Found {file_total} total chunks, {file_skipped} skipped, {file_filtered} filtered, {file_processed} available for processing")
                print(f"   Processing in subsets of {self.subset_size}")

                # Process chunks in subsets with comprehensive error handling and tracking
                total_subsets = (len(chunk_metadata) + self.subset_size - 1) // self.subset_size
                processed_subsets = 0
                file_chunks_processed = 0
                
                for subset_start in range(0, len(chunk_metadata), self.subset_size):
                    subset_end = min(subset_start + self.subset_size, len(chunk_metadata))
                    subset = chunk_metadata[subset_start:subset_end]
                    subset_number = subset_start // self.subset_size + 1
                    
                    print(f"\n🔄 Processing subset {subset_number}/{total_subsets}: chunks {subset_start+1}-{subset_end}")
                    
                    # Comprehensive error handling for entire subset processing
                    try:
                        # Track chunk flow through the pipeline
                        subset_expected = len(subset)
                        print(f"   📊 Subset pipeline tracking:")
                        print(f"      ⚡ Expected chunks in subset: {subset_expected}")
                        
                        # Step 1: Download and read content concurrently
                        start_time = time.time()
                        subset_ids, subset_chunks, subset_metadata, downloaded_paths = self.process_subset_concurrent(subset)
                        download_time = time.time() - start_time
                        
                        chunks_after_download = len(subset_chunks)
                        print(f"   ⬇️ Downloaded {chunks_after_download} chunks in {download_time:.2f}s")
                        print(f"      📉 Lost in download/read: {subset_expected - chunks_after_download} chunks")
                        
                        if not subset_chunks:
                            print("   ⚠️ No chunks downloaded, skipping subset")
                            print(f"   📊 Expected {len(subset)} chunks, got {len(subset_chunks)} chunks")
                            # Track the lost chunks
                            lost_chunks = len(subset)
                            print(f"   📉 LOST CHUNKS: {lost_chunks} chunks lost in download/read phase")
                            continue

                        # Step 2: Generate embeddings (GPU intensive)
                        start_time = time.time()
                        print(f"   🧠 Generating embeddings for {len(subset_chunks)} chunks...")
                        
                        # Monitor memory before embedding
                        high_memory = self.print_gpu_memory_stats("Before Embedding")
                        if high_memory:
                            print("   🧹 Performing pre-embedding cleanup...")
                            torch.cuda.empty_cache()
                            gc.collect()
                        
                        vectors = self._embed_texts_safe(subset_chunks, embed_batch_size=8)  # Reduce batch size
                        embedding_time = time.time() - start_time
                        print(f"   ✅ Embedding attempt finished in {embedding_time:.2f}s")
                        
                        chunks_after_embedding = len([v for v in vectors if v is not None])
                        chunks_failed_embedding = len([v for v in vectors if v is None])
                        print(f"      ✅ Successful embeddings: {chunks_after_embedding}")
                        print(f"      ❌ Failed embeddings: {chunks_failed_embedding}")
                        print(f"      📉 Lost in embedding: {chunks_after_download - chunks_after_embedding} chunks")
                        
                        # Monitor memory after embedding
                        self.print_gpu_memory_stats("After Embedding")
                        
                        # Debug: Check vector types
                        if vectors:
                            print(f"   📊 First vector type: {type(vectors[0])}")
                            if vectors[0] is not None:
                                if hasattr(vectors[0], '__len__'):
                                    print(f"   📊 First vector length: {len(vectors[0])}")
                                else:
                                    print(f"   ⚠️ First vector has no length: {vectors[0]}")

                        # Filter out failed vectors
                        good_items = [
                            (uid, vec, meta)
                            for uid, vec, meta in zip(subset_ids, vectors, subset_metadata)
                            if vec is not None
                        ]
                        failed_in_subset = [
                            uid for uid, vec, _ in zip(subset_ids, vectors, subset_metadata) if vec is None
                        ]
                        
                        chunks_ready_for_upload = len(good_items)
                        print(f"      🎯 Ready for upload: {chunks_ready_for_upload} chunks")

                        if failed_in_subset:
                            print(f"   ⚠️ {len(failed_in_subset)} items failed to embed and will be skipped")
                            failed += len(failed_in_subset)
                            failed_ids.extend(failed_in_subset)
                            # Update per-file stats for failed embeds
                            per_file_stats[file_name]["failed"] += len(failed_in_subset)
                            per_file_stats[file_name]["processed"] += len(failed_in_subset)

                        if not good_items:
                            print("   ⚠️ No successful vectors in subset, skipping")
                            print(f"   📉 LOST CHUNKS: {subset_expected} chunks lost - no successful vectors")
                            cleanup_files(downloaded_paths)
                            continue

                        # Step 3: Upload to Qdrant in batches
                        total_good = len(good_items)
                        subset_batches = (total_good + self.batch_size - 1) // self.batch_size
                        total_batches += subset_batches
                        
                        chunks_uploaded_in_subset = 0
                        chunks_failed_upload_in_subset = 0

                        for b_start in range(0, total_good, self.batch_size):
                            b_end = min(b_start + self.batch_size, total_good)
                            batch = good_items[b_start:b_end]
                            batch_ids = [item[0] for item in batch]
                            batch_vectors = [item[1] for item in batch]
                            batch_meta = [item[2] for item in batch]

                            # Create batch points
                            batch_points = [
                                models.PointStruct(
                                    id=self.string_to_uint(uid),
                                    vector=vec,
                                    payload=meta
                                )
                                for uid, vec, meta in zip(batch_ids, batch_vectors, batch_meta)
                            ]

                            # Upload with retry logic
                            print(f"   📤 Uploading batch {b_start//self.batch_size + 1}/{subset_batches} ({len(batch_points)} points)")
                            success, failed_batch_ids = self.upload_batch_with_retry(batch_points, batch_ids)
                            
                            if success:
                                succeeded += len(batch_ids)
                                total_uploaded_to_qdrant += len(batch_ids)  # Track successful Qdrant uploads
                                per_file_stats[file_name]["succeeded"] += len(batch_ids)
                                chunks_uploaded_in_subset += len(batch_ids)
                                print(f"   ✅ Batch uploaded successfully - {len(batch_ids)} chunks uploaded to Qdrant")
                                print(f"   📊 Total uploaded to Qdrant so far: {total_uploaded_to_qdrant} chunks")
                            else:
                                failed += len(batch_ids)
                                failed_ids.extend(failed_batch_ids if failed_batch_ids else batch_ids)
                                per_file_stats[file_name]["failed"] += len(batch_ids)
                                chunks_failed_upload_in_subset += len(batch_ids)
                                print(f"   ❌ Batch upload failed permanently")
                            
                            # Update processed count
                            processed += len(batch_ids)
                            per_file_stats[file_name]["processed"] += len(batch_ids)

                            # Add small delay between batches to reduce load
                            if b_start + self.batch_size < total_good:
                                time.sleep(0.5)  # 500ms delay between batches

                        file_chunks_processed += len(good_items)
                        processed_subsets += 1
                        
                        # Comprehensive subset summary
                        print(f"\n   📊 SUBSET {subset_number}/{total_subsets} COMPLETE PIPELINE SUMMARY:")
                        print(f"      🎯 Started with: {subset_expected} chunks")
                        print(f"      ⬇️ After download/read: {chunks_after_download} chunks")
                        print(f"      🧠 After embedding: {chunks_after_embedding} chunks")
                        print(f"      📤 Ready for upload: {chunks_ready_for_upload} chunks")
                        print(f"      ✅ Successfully uploaded: {chunks_uploaded_in_subset} chunks")
                        print(f"      ❌ Failed upload: {chunks_failed_upload_in_subset} chunks")
                        print(f"      📉 TOTAL LOST: {subset_expected - chunks_uploaded_in_subset} chunks")
                        
                        # Breakdown of losses
                        download_loss = subset_expected - chunks_after_download
                        embedding_loss = chunks_after_download - chunks_after_embedding
                        upload_loss = chunks_ready_for_upload - chunks_uploaded_in_subset
                        
                        if download_loss > 0:
                            print(f"         📉 Lost in download/read: {download_loss} chunks")
                        if embedding_loss > 0:
                            print(f"         📉 Lost in embedding: {embedding_loss} chunks")
                        if upload_loss > 0:
                            print(f"         📉 Lost in upload: {upload_loss} chunks")
                        
                        print(f"   📤 Uploaded subset {subset_number}/{total_subsets} (good: {len(good_items)})")
                        
                        # Print subset upload summary
                        subset_uploaded = sum(len(good_items[b_start:min(b_start + self.batch_size, len(good_items))]) 
                                            for b_start in range(0, len(good_items), self.batch_size))
                        print(f"   🎯 Subset summary: {subset_uploaded} chunks uploaded to Qdrant after score filtering")

                        # Step 4: Cleanup downloaded files for this subset
                        print(f"   🧹 Cleaning up {len(downloaded_paths)} downloaded files")
                        cleanup_files(downloaded_paths)
                        
                        # Aggressive memory cleanup after each subset
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()

                        # Update stats
                        write_stats()
                        
                    except Exception as e:
                        print(f"   ❌ CRITICAL ERROR in subset {subset_number}/{total_subsets}: {e}")
                        print(f"   🔄 Attempting to continue with next subset...")
                        # Cleanup any downloaded files
                        try:
                            cleanup_files(downloaded_paths if 'downloaded_paths' in locals() else [])
                        except:
                            pass
                        # Clear GPU memory on error
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                        continue
                
                # File processing completed - add summary
                print(f"📊 File {file_name} processing summary:")
                print(f"   🎯 Subsets processed: {processed_subsets}/{total_subsets}")
                print(f"   📦 Chunks processed in this file: {file_chunks_processed}")
                print(f"   📊 Expected to process: {file_processed}")
                if file_chunks_processed != file_processed:
                    print(f"   ⚠️ MISMATCH: Expected {file_processed}, actually processed {file_chunks_processed}")

                print(f"✅ Completed file {file_name}")
                
                # Debug verification of file stats
                final_processed = per_file_stats[file_name]["processed"]
                final_succeeded = per_file_stats[file_name]["succeeded"]  
                final_failed = per_file_stats[file_name]["failed"]
                print(f"🔍 File {file_name} final stats verification:")
                print(f"   📊 Available for processing: {file_processed}")
                print(f"   📊 Actually processed: {final_processed}")
                print(f"   📊 Succeeded: {final_succeeded}")
                print(f"   📊 Failed: {final_failed}")
                
                # Print file-level upload summary
                file_uploaded = per_file_stats[file_name]["succeeded"]
                print(f"📊 File summary for {file_name}:")
                print(f"   📥 Total chunks in file: {file_total}")
                print(f"   🚫 Filtered by score threshold ({self.score_threshold}): {file_filtered}")
                print(f"   ⚡ Available for processing: {file_processed}")
                print(f"   � Successfully uploaded to Qdrant: {file_uploaded}")
                print(f"   📈 File upload rate: {(file_uploaded / max(1, file_processed)) * 100:.1f}%")
                
                # Add delay between files when running multiple pods
                time.sleep(1.0)  # 1 second delay between files
                
                # Memory cleanup between files
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

            print("�🎉 All uploads complete")
            
            # Final memory stats
            self.print_gpu_memory_stats("Final")
            
            print(f"📝 Stats file: {stats_path}")
            print(f"📊 Final upload summary:")
            print(f"   📈 Total chunks found across all files: {total_chunks}")
            print(f"   ⏭️ Skipped chunks (first {self.skip_first_chunks}): {getattr(self, 'total_skipped', 0)}")
            print(f"   🚫 Filtered by score threshold ({self.score_threshold}): {total_filtered}")
            print(f"   ⚡ Available for processing: {total_chunks - getattr(self, 'total_skipped', 0) - total_filtered}")
            print(f"   🎯 Successfully uploaded to Qdrant: {total_uploaded_to_qdrant}")
            print(f"   ❌ Failed uploads: {failed}")
            
            available_for_processing = total_chunks - getattr(self, 'total_skipped', 0) - total_filtered
            success_rate = (total_uploaded_to_qdrant / max(1, available_for_processing)) * 100
            print(f"   📊 Upload success rate: {success_rate:.1f}%")
            print(f"   🎯 QDRANT UPLOAD SUMMARY: {total_uploaded_to_qdrant} chunks successfully stored in Qdrant collection '{self.collection_name}' after skipping {getattr(self, 'total_skipped', 0)} and filtering with score ≥ {self.score_threshold}")
            
            # Validation checks
            print(f"\n🔍 VALIDATION CHECKS:")
            expected_available = total_chunks - getattr(self, 'total_skipped', 0) - total_filtered
            print(f"   ✓ Available chunks math: {total_chunks} - {getattr(self, 'total_skipped', 0)} - {total_filtered} = {expected_available}")
            print(f"   ✓ Processed chunks: {processed} {'✅' if processed <= expected_available else '❌ MISMATCH!'}")
            print(f"   ✓ Success + Failure = Total: {total_uploaded_to_qdrant} + {failed} = {total_uploaded_to_qdrant + failed} {'✅' if (total_uploaded_to_qdrant + failed) == processed else '❌ MISMATCH!'}")
            
            # Per-file validation
            print(f"\n📁 PER-FILE VALIDATION:")
            for fname, fstats in per_file_stats.items():
                ftotal = fstats['total_chunks']
                ffiltered = fstats['filtered_chunks'] 
                favailable = fstats['available_for_processing']
                fprocessed = fstats['processed']
                fsucceeded = fstats['succeeded']
                ffailed = fstats['failed']
                
                print(f"   📄 {fname}:")
                print(f"      Total: {ftotal}, Filtered: {ffiltered}, Available: {favailable}")
                print(f"      Processed: {fprocessed}, Succeeded: {fsucceeded}, Failed: {ffailed}")
                
                # Validation
                expected_avail = ftotal - ffiltered
                total_outcome = fsucceeded + ffailed
                if expected_avail != favailable:
                    print(f"      ❌ Available mismatch: {ftotal} - {ffiltered} = {expected_avail} != {favailable}")
                if fprocessed != total_outcome:
                    print(f"      ❌ Processed mismatch: {fprocessed} != {fsucceeded} + {ffailed} = {total_outcome}")
                if fprocessed > favailable:
                    print(f"      ❌ Over-processed: {fprocessed} > {favailable}")
            
            print(f"\n🎯 FINAL RESULT: {total_uploaded_to_qdrant} chunks successfully stored in Qdrant collection '{self.collection_name}' after filtering with score ≥ {self.score_threshold}")
            
            # Write final stats
            write_stats()
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user. Writing final stats before exit...")
            write_stats()
            print(f"📝 Stats file (live-updated): {stats_path}")
            return



if __name__ == "__main__":
    # First configure aws credentials before running this script
    # aws configure
    # aws s3 sync s3://esa-satcom-s3/chunks/_analytics_/chunks_with_metadata/ /workspace/chunks_with_metadata/
    
    # Configuration-based uploader using YAML config
    # pip install sentence-transformers transformers torch accelerate
    
    # Initialize uploader with config file - starting from first chunk
    uploader = MinimalQdrantUploader(
        config_path="/workspace/config_qwen.yaml",
        chunks_folder="/workspace/pod4",
        score_threshold=-11,  # Set your desired threshold here
        skip_first_chunks=0  # Start from the beginning (no skipping)
    )

    print("🚀 Starting concurrent upload process with config-based embedding")
    print(f"🤖 Using model: {uploader.config['embedding']['model_name']}")
    print(f"   Model type: {uploader.config['embedding']['type']}")
    print(f"   Config vector dimension: {uploader.vector_size}")
    print(f"   Actual vector dimension: {uploader.actual_vector_size}")
    print(f"   Score threshold: {uploader.score_threshold}")
    print(f"   Skip first chunks: {uploader.skip_first_chunks} (starting from beginning)")
    print(f"   Subset size: {uploader.subset_size} chunks")
    print(f"   Download threads: {uploader.download_threads}")
    print(f"   Batch size: {uploader.batch_size}")
    print()
    
    # Start upload
    uploader.upload_to_qdrant()