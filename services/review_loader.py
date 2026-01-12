"""
Review Loader Service

Downloads and streams Amazon reviews from HuggingFace Hub.
"""

import os
import gzip
import json
import structlog
from typing import Iterator, Dict, Any, Optional
from huggingface_hub import hf_hub_download

log = structlog.get_logger()


class ReviewLoader:
    """
    Loads Amazon reviews from HuggingFace Hub, streaming them efficiently.
    """

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.cache_dir = os.getenv("HF_CACHE_DIR", "/tmp/hf_cache")

    def get_review_file(self, category: str) -> str:
        """
        Downloads the review file for a category and returns its local path.
        """
        # Convert category name to HuggingFace filename format
        # e.g., "Clothing_Shoes_and_Jewelry" -> "raw_review_Clothing_Shoes_and_Jewelry.jsonl"
        filename = f"raw_review_{category}.jsonl"
        
        log.info("Downloading review file", category=category, filename=filename)
        
        try:
            local_path = hf_hub_download(
                repo_id="McAuley-Lab/Amazon-Reviews-2023",
                filename=f"raw/review_categories/{filename}",
                repo_type="dataset",
                token=self.hf_token,
                cache_dir=self.cache_dir,
            )
            return local_path
        except Exception as e:
            log.error("Failed to download review file", category=category, error=str(e))
            raise

    def stream_reviews(
        self,
        category: str,
        max_reviews: Optional[int] = None,
        filter_categories: Optional[set] = None,
        asin_map: Optional[Dict[str, str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Streams reviews from a category file using HuggingFace Streaming (no full download).
        
        Args:
            category: The Amazon category (e.g., "Clothing_Shoes_and_Jewelry")
            max_reviews: Maximum number of reviews to yield
            filter_categories: If provided, only yield reviews for products in these leaf categories
            asin_map: Pre-computed mapping of ASIN -> leaf_category (required if filtering)
        """
        from datasets import load_dataset
        
        filename = f"raw_review_{category}.jsonl"
        data_files = {"train": f"raw/review_categories/{filename}"}
        
        log.info("Streaming reviews from HuggingFace (no filtered download)", category=category)
        
        # Use streaming=True to avoid downloading 11GB file
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            data_files=data_files,
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        count = 0
        skipped = 0
        
        for item in dataset:
            if max_reviews and count >= max_reviews:
                break
            
            # If filtering by category, we need the ASIN map
            if filter_categories and asin_map:
                asin = item.get("parent_asin")
                product_category = asin_map.get(asin)
                
                if not product_category or product_category not in filter_categories:
                    skipped += 1
                    continue
            
            yield item
            count += 1
            
            if count % 1000 == 0:
                log.info("Streaming progress", count=count, skipped=skipped)
        
        log.info("Finished streaming reviews", total=count, skipped=skipped)

    def build_asin_to_leaf_category(
        self,
        category: str,
        leaf_categories: set,
    ) -> Dict[str, str]:
        """
        Builds a mapping from ASIN to leaf category name.
        Checks MongoDB first to avoid re-downloading massive metadata.
        """
        # 1. Try to load from MongoDB
        from services.mongo_storage import MongoStorage
        try:
            mongo = MongoStorage()
            if mongo.db is not None:
                stored_map = mongo.get_asin_map(category)
                if stored_map:
                    log.info("Loaded ASIN mapping from MongoDB", count=len(stored_map))
                    return stored_map
        except Exception as e:
            log.warn("Could not load from MongoDB, falling back to file", error=str(e))

        # 2. Fallback: Download and build from file
        metadata_path = self.get_metadata_file(category)
        asin_map = {}
        
        log.info("Building ASIN to leaf category mapping from file (this happens once)")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    asin = item.get("parent_asin")
                    categories = item.get("categories", [])
                    
                    # Check if any of the item's categories match our leaf categories
                    for cat in categories:
                        if cat in leaf_categories:
                            asin_map[asin] = cat
                            break
                except json.JSONDecodeError:
                    continue
        
        log.info("Built ASIN mapping", total_asins=len(asin_map))
        
        # 3. Save to MongoDB for next time
        try:
            if mongo.db is not None:
                mongo.save_asin_map(category, asin_map)
                log.info("Saved ASIN mapping to MongoDB")
        except Exception as e:
            log.warn("Could not save to MongoDB", error=str(e))
            
        return asin_map
