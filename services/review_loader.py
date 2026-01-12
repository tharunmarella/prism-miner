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
    ) -> Iterator[Dict[str, Any]]:
        """
        Streams reviews from a category file.
        
        Args:
            category: The Amazon category (e.g., "Clothing_Shoes_and_Jewelry")
            max_reviews: Maximum number of reviews to yield
            filter_categories: If provided, only yield reviews for products in these leaf categories
        """
        local_path = self.get_review_file(category)
        
        count = 0
        with open(local_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_reviews and count >= max_reviews:
                    break
                
                try:
                    review = json.loads(line)
                    
                    # If we have a filter, check if this review's product matches
                    # (We'll need the asin2category mapping for this)
                    
                    yield review
                    count += 1
                    
                    if count % 100000 == 0:
                        log.info("Streaming reviews", count=count)
                        
                except json.JSONDecodeError:
                    continue
        
        log.info("Finished streaming reviews", total=count)

    def get_metadata_file(self, category: str) -> str:
        """
        Downloads the metadata file for a category and returns its local path.
        """
        filename = f"meta_{category}.jsonl"
        
        log.info("Downloading metadata file", category=category, filename=filename)
        
        try:
            local_path = hf_hub_download(
                repo_id="McAuley-Lab/Amazon-Reviews-2023",
                filename=f"raw/meta_categories/{filename}",
                repo_type="dataset",
                token=self.hf_token,
                cache_dir=self.cache_dir,
            )
            return local_path
        except Exception as e:
            log.error("Failed to download metadata file", category=category, error=str(e))
            raise

    def build_asin_to_leaf_category(
        self,
        category: str,
        leaf_categories: set,
    ) -> Dict[str, str]:
        """
        Builds a mapping from ASIN to leaf category name.
        This is needed to filter reviews by specific product categories.
        """
        metadata_path = self.get_metadata_file(category)
        asin_map = {}
        
        log.info("Building ASIN to leaf category mapping")
        
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
        return asin_map
