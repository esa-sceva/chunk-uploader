"""Embedding model management."""
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from abc import ABC, abstractmethod
from typing import List
import gc


def last_token_pool(last_hidden_states, attention_mask):
    """Pool last token from transformer outputs."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class BaseEmbedder(ABC):
    """Base class for embedding models."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str], batch_size: int = 8, normalize: bool = True) -> List[List[float]]:
        """Embed a list of documents."""
        pass
    
    def _clear_gpu_cache(self):
        """Clear GPU cache if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


class QwenSentenceTransformerEmbedder(BaseEmbedder):
    """Qwen embedder using sentence-transformers."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B"):
        try:
            print(f"Loading {model_name} with sentence-transformers...")
            self.model = SentenceTransformer(
                model_name,
                model_kwargs={"torch_dtype": "auto", "device_map": "auto"},
                tokenizer_kwargs={"padding_side": "left", "max_length": 2048, "truncation": True}
            )
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            raise

    def embed_documents(self, texts: List[str], batch_size: int = 8, normalize: bool = True) -> List[List[float]]:
        """Encode texts into embeddings."""
        try:
            print(f"Processing {len(texts)} texts with batch_size={batch_size}")
            
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.memory_allocated() / 1024**3:.1f}GB allocated")
            
            start_time = time.time()
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                convert_to_tensor=False,
                show_progress_bar=True
            )
            
            print(f"Embedding completed in {time.time() - start_time:.3f}s")
            
            self._clear_gpu_cache()
            return embeddings.tolist()
            
        except Exception as e:
            print(f"Embedding failed: {e}")
            self._clear_gpu_cache()
            raise


class QwenTransformerEmbedder(BaseEmbedder):
    """Qwen embedder using transformers directly."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B", max_length: int = 2048):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )
        self.max_length = max_length

    def embed_documents(self, texts: List[str], batch_size: int = 8, normalize: bool = True) -> List[List[float]]:
        """Encode texts into embeddings."""
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
                    
                    del outputs, embeddings, batch_dict
                    
                if (i // batch_size) % 5 == 0:
                    self._clear_gpu_cache()
                    
        finally:
            self._clear_gpu_cache()
                
        return all_embeddings


class EmbeddingModelFactory:
    """Factory for creating embedding models."""
    
    VECTOR_SIZES = {
        "Qwen3-Embedding-0.6B": 1024,
        "Qwen3-Embedding-4B": 2560,
        "Qwen3-Embedding-8B": 4096,
        "Qwen2-Embedding": 1024,
        "nasa-impact/nasa-smd-ibm-st-v2": 768,
        "indus-sde-st-v0.2": 768,
    }
    
    @classmethod
    def create(cls, model_name: str, model_type: str = 'sentence', normalize: bool = True) -> tuple[BaseEmbedder, int]:
        """Create and return an embedder and its vector size."""
        try:
            if "nasa-impact/nasa-smd-ibm-st-v2" in model_name:
                encode_kwargs = {"normalize_embeddings": normalize}
                print('NASA SMD model loaded')
                embedder = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)
                return embedder, 768
                
            elif "Qwen/Qwen3-Embedding" in model_name or "Qwen/Qwen2-Embedding" in model_name:
                print(f'Qwen model loaded: {model_name} with type: {model_type}')
                if model_type == 'sentence':
                    embedder = QwenSentenceTransformerEmbedder(model_name=model_name)
                elif model_type == 'transformer':
                    embedder = QwenTransformerEmbedder(model_name=model_name)
                else:
                    raise ValueError('Qwen model type must be "sentence" or "transformer"')
                
                # Determine vector size
                vector_size = next(
                    (size for key, size in cls.VECTOR_SIZES.items() if key in model_name),
                    2560  # Default fallback
                )
                return embedder, vector_size
                
            elif "indus-sde-st-v0.2" in model_name:
                encode_kwargs = {"normalize_embeddings": normalize}
                print('Indus model loaded')
                embedder = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)
                return embedder, 768
            else:
                raise ValueError(f'Unsupported model: {model_name}')
                
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise

