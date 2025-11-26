"""Statistics tracking for upload process."""
import json
import os
from datetime import datetime
from typing import Dict, List


class StatsTracker:
    """Track and persist upload statistics."""
    
    def __init__(self, pod_id: str, collection_name: str, score_threshold: float):
        self.pod_id = pod_id
        self.collection_name = collection_name
        self.score_threshold = score_threshold
        self.started_at = datetime.now().isoformat() + "Z"
        
        # Global stats
        self.total_chunks = 0
        self.total_skipped = 0
        self.total_filtered = 0
        self.total_uploaded = 0
        self.total_batches = 0
        self.processed = 0
        self.succeeded = 0
        self.failed = 0
        self.failed_ids: List[str] = []
        
        # Per-file stats
        self.per_file_stats: Dict[str, Dict] = {}
        
        # Stats file path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.stats_path = os.path.abspath(f"upload_stats_{pod_id}_{timestamp}.json")
    
    def init_file_stats(self, file_name: str, total: int, skipped: int, filtered: int, available: int):
        """Initialize statistics for a file."""
        self.per_file_stats[file_name] = {
            "total_chunks": total,
            "skipped_chunks": skipped,
            "filtered_chunks": filtered,
            "available_for_processing": available,
            "processed": 0,
            "succeeded": 0,
            "failed": 0
        }
    
    def update_global(self, total: int = 0, skipped: int = 0, filtered: int = 0, 
                     uploaded: int = 0, batches: int = 0, processed: int = 0,
                     succeeded: int = 0, failed: int = 0):
        """Update global statistics."""
        self.total_chunks += total
        self.total_skipped += skipped
        self.total_filtered += filtered
        self.total_uploaded += uploaded
        self.total_batches += batches
        self.processed += processed
        self.succeeded += succeeded
        self.failed += failed
    
    def update_file(self, file_name: str, processed: int = 0, succeeded: int = 0, failed: int = 0):
        """Update file-level statistics."""
        if file_name in self.per_file_stats:
            self.per_file_stats[file_name]["processed"] += processed
            self.per_file_stats[file_name]["succeeded"] += succeeded
            self.per_file_stats[file_name]["failed"] += failed
    
    def add_failed_ids(self, failed_ids: List[str]):
        """Add failed chunk IDs."""
        self.failed_ids.extend(failed_ids)
    
    def write_stats(self):
        """Write statistics to JSON file."""
        stats = {
            "timestamp": datetime.now().isoformat() + "Z",
            "started_at": self.started_at,
            "collection_name": self.collection_name,
            "pod_id": self.pod_id,
            "score_threshold": self.score_threshold,
            "total_chunks": self.total_chunks,
            "total_skipped": self.total_skipped,
            "total_filtered": self.total_filtered,
            "total_available_for_processing": self.total_chunks - self.total_skipped - self.total_filtered,
            "total_uploaded_to_qdrant": self.total_uploaded,
            "upload_success_rate": (self.total_uploaded / max(1, self.total_chunks - self.total_skipped - self.total_filtered)) * 100,
            "total_batches": self.total_batches,
            "processed": self.processed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "failed_ids": self.failed_ids,
            "per_file_stats": self.per_file_stats,
        }
        
        try:
            with open(self.stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            print(f"⚠️ Could not write stats file: {e}")
    
    def print_summary(self):
        """Print final statistics summary."""
        print(f"\n📝 Stats file: {self.stats_path}")
        print(f"📊 Final upload summary:")
        print(f"   📈 Total chunks found: {self.total_chunks}")
        print(f"   ⏭️ Skipped chunks: {self.total_skipped}")
        print(f"   🚫 Filtered by score threshold ({self.score_threshold}): {self.total_filtered}")
        
        available = self.total_chunks - self.total_skipped - self.total_filtered
        print(f"   ⚡ Available for processing: {available}")
        print(f"   🎯 Successfully uploaded to Qdrant: {self.total_uploaded}")
        print(f"   ❌ Failed uploads: {self.failed}")
        
        success_rate = (self.total_uploaded / max(1, available)) * 100
        print(f"   📊 Upload success rate: {success_rate:.1f}%")
    
    def print_validation(self):
        """Print validation checks."""
        print(f"\n🔍 VALIDATION CHECKS:")
        expected_available = self.total_chunks - self.total_skipped - self.total_filtered
        print(f"   ✓ Available chunks: {self.total_chunks} - {self.total_skipped} - {self.total_filtered} = {expected_available}")
        print(f"   ✓ Processed chunks: {self.processed} {'✅' if self.processed <= expected_available else '❌ MISMATCH!'}")
        print(f"   ✓ Success + Failure: {self.total_uploaded} + {self.failed} = {self.total_uploaded + self.failed} {'✅' if (self.total_uploaded + self.failed) == self.processed else '❌ MISMATCH!'}")
        
        print(f"\n📁 PER-FILE VALIDATION:")
        for fname, fstats in self.per_file_stats.items():
            print(f"   📄 {fname}:")
            print(f"      Total: {fstats['total_chunks']}, Filtered: {fstats['filtered_chunks']}, Available: {fstats['available_for_processing']}")
            print(f"      Processed: {fstats['processed']}, Succeeded: {fstats['succeeded']}, Failed: {fstats['failed']}")

