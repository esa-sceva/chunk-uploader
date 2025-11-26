"""JSON chunk file parsing."""
import json
import os
from typing import List, Dict, Tuple


class ChunkMetadata:
    """Represents chunk metadata."""
    
    def __init__(self, uid: str, s3_uri: str, metadata: dict):
        self.uid = uid
        self.s3_uri = s3_uri
        self.metadata = metadata


class ChunkParser:
    """Parse JSON files containing chunk information."""
    
    S3_BUCKET = "esa-satcom-s3"
    
    def __init__(self, score_threshold: float = 0.0, skip_first_chunks: int = 0):
        self.score_threshold = score_threshold
        self.skip_first_chunks = skip_first_chunks
        self.global_chunk_counter = 0
        self.chunks_skipped = 0
    
    def parse_file(self, file_path: str) -> Tuple[List[ChunkMetadata], Dict[str, int]]:
        """Parse a JSON file and extract chunk metadata."""
        file_name = os.path.basename(file_path)
        sub_folder = os.path.splitext(file_name)[0]
        
        print(f"📂 Processing file: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        chunk_metadata = []
        stats = {
            "total": 0,
            "skipped": 0,
            "filtered": 0,
            "processed": 0
        }
        
        if isinstance(data, list):
            # Array format
            self._parse_array_format(data, file_name, sub_folder, chunk_metadata, stats)
        else:
            # Nested object format
            self._parse_object_format(data, file_name, sub_folder, chunk_metadata, stats)
        
        stats["processed"] = len(chunk_metadata)
        self._print_stats(stats, sub_folder)
        
        return chunk_metadata, stats
    
    def _parse_array_format(self, data: list, file_name: str, sub_folder: str, 
                           chunk_metadata: List[ChunkMetadata], stats: Dict[str, int]):
        """Parse array format JSON."""
        for chunk in data:
            if not isinstance(chunk, dict) or "output_path" not in chunk:
                continue
            
            stats["total"] += 1
            self.global_chunk_counter += 1
            
            if self._should_skip_chunk(chunk, stats):
                continue
            
            metadata = self._extract_metadata(chunk, file_name, sub_folder)
            chunk_metadata.append(metadata)
    
    def _parse_object_format(self, data: dict, file_name: str, sub_folder: str,
                            chunk_metadata: List[ChunkMetadata], stats: Dict[str, int]):
        """Parse nested object format JSON."""
        if len(data) == 1:
            top_level_key = list(data.keys())[0]
            main_data = data[top_level_key]
            
            if "documents" in main_data:
                self._parse_documents_structure(main_data["documents"], file_name, sub_folder, chunk_metadata, stats)
            else:
                self._parse_flat_structure(main_data, file_name, sub_folder, chunk_metadata, stats)
        else:
            self._parse_flat_structure(data, file_name, sub_folder, chunk_metadata, stats)
    
    def _parse_documents_structure(self, documents: dict, file_name: str, sub_folder: str,
                                   chunk_metadata: List[ChunkMetadata], stats: Dict[str, int]):
        """Parse documents structure."""
        for doc_id, doc_info in documents.items():
            chunks = doc_info.get("chunks", {})
            for chunk_id, chunk in chunks.items():
                stats["total"] += 1
                self.global_chunk_counter += 1
                
                if self._should_skip_chunk(chunk, stats):
                    continue
                
                metadata = self._extract_metadata(chunk, file_name, sub_folder, chunk_id, doc_id)
                chunk_metadata.append(metadata)
    
    def _parse_flat_structure(self, data: dict, file_name: str, sub_folder: str,
                             chunk_metadata: List[ChunkMetadata], stats: Dict[str, int]):
        """Parse flat structure."""
        for chunk_id, chunk in data.items():
            if not isinstance(chunk, dict):
                continue
            
            stats["total"] += 1
            self.global_chunk_counter += 1
            
            if self._should_skip_chunk(chunk, stats):
                continue
            
            metadata = self._extract_metadata(chunk, file_name, sub_folder, chunk_id)
            chunk_metadata.append(metadata)
    
    def _should_skip_chunk(self, chunk: dict, stats: Dict[str, int]) -> bool:
        """Determine if chunk should be skipped."""
        # Skip first N chunks
        if self.global_chunk_counter <= self.skip_first_chunks:
            stats["skipped"] += 1
            self.chunks_skipped += 1
            return True
        
        # Filter by score
        score = chunk.get("score", 0)
        if score < self.score_threshold:
            stats["filtered"] += 1
            return True
        
        return False
    
    def _extract_metadata(self, chunk: dict, file_name: str, sub_folder: str, 
                         chunk_id: str = None, doc_id: str = None) -> ChunkMetadata:
        """Extract metadata from chunk."""
        output_path = chunk.get("output_path", "")
        s3_uri = f"s3://{self.S3_BUCKET}/{output_path}"
        
        chunk_name = chunk.get("chunk_name") or os.path.basename(output_path)
        if not chunk_name.endswith('.md'):
            chunk_name = chunk_name + '.md'
        
        doc_id = doc_id or chunk.get("document_id") or chunk.get("doc_id", "unknown_doc")
        original_file_name = f"{doc_id}.pdf"
        
        meta = {
            "source_url": chunk.get("source_url", ""),
            "score": chunk.get("score", 0),
            "original_file_name": original_file_name,
            "sub_folder": sub_folder,
            "chunk_name": chunk_name,
            "source_json_file": file_name,
        }
        
        uid = chunk_name if not chunk_id else f"{doc_id}_{chunk_id}"
        
        return ChunkMetadata(uid=uid, s3_uri=s3_uri, metadata=meta)
    
    def _print_stats(self, stats: Dict[str, int], sub_folder: str):
        """Print parsing statistics."""
        print(f"   Found {stats['total']} total chunks in {sub_folder}")
        print(f"   Skipped {stats['skipped']} chunks (first {self.skip_first_chunks} chunks)")
        print(f"   Filtered {stats['filtered']} chunks below score threshold {self.score_threshold}")
        print(f"   Remaining {stats['processed']} chunks for processing")

