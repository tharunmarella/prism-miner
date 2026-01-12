"""
FULL PIPELINE: Mine Hidden Dimensions for ALL 2,195 Clothing Categories

This script:
1. Downloads 66M reviews from HuggingFace
2. Maps reviews to 2,195 leaf categories
3. Extracts opinion units with spaCy
4. Clusters semantically with embeddings
5. Calls Groq Live API for each category
6. Saves to MongoDB with checkpointing

Usage:
    # Full run (all categories)
    python3 prism_miner/scripts/run_full_pipeline.py
    
    # Test with 5 categories first
    python3 prism_miner/scripts/run_full_pipeline.py --test 5
    
    # Resume from checkpoint
    python3 prism_miner/scripts/run_full_pipeline.py --resume

Environment:
    MONGODB_URI=mongodb://...
    GROQ_API_KEY=gsk_...
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Set
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import structlog
from pymongo import MongoClient
from huggingface_hub import hf_hub_download
from groq import Groq

from prism_miner.services.opinion_extractor import OpinionUnitExtractor
from prism_miner.services.aggregator import OpinionAggregator

# Setup logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer()
    ]
)
log = structlog.get_logger()

# Config
REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
REVIEW_FILENAME = "raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl"
META_FILENAME = "raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
RATE_LIMIT_PER_MIN = 28  # Stay under 30 to be safe
MIN_REVIEWS_PER_CATEGORY = 50  # Skip categories with too few reviews


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute
        self.last_call = 0
        self.lock = threading.Lock()
    
    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()


class FullPipeline:
    """
    Full pipeline to mine hidden dimensions for all clothing categories.
    """
    
    def __init__(
        self,
        mongodb_uri: str,
        groq_api_key: str,
        max_reviews_per_category: int = 10000,
        cache_dir: str = "/tmp/hf_cache",
    ):
        self.mongodb_uri = mongodb_uri
        self.groq_api_key = groq_api_key
        self.max_reviews_per_category = max_reviews_per_category
        self.cache_dir = cache_dir
        
        # Initialize clients
        self.mongo_client = MongoClient(mongodb_uri)
        self.db = self.mongo_client.get_database("prism_miner")
        self.groq_client = Groq(api_key=groq_api_key)
        self.rate_limiter = RateLimiter(RATE_LIMIT_PER_MIN)
        
        # Collections
        self.dimensions_col = self.db["category_dimensions"]
        self.progress_col = self.db["pipeline_progress"]
        self.reviews_col = self.db["reviews"]
        
        # Services
        self.extractor = OpinionUnitExtractor()
        
        # Stats
        self.stats = {
            "started_at": datetime.utcnow(),
            "categories_processed": 0,
            "categories_skipped": 0,
            "total_reviews": 0,
            "total_patterns": 0,
            "errors": [],
        }
    
    def load_leaf_categories(self, path: str) -> Dict[str, int]:
        """Load leaf categories from file."""
        categories = {}
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2 and parts[0].isdigit():
                    count = int(parts[0])
                    name = parts[1]
                    categories[name] = count
        log.info("Loaded leaf categories", count=len(categories))
        return categories
    
    def get_completed_categories(self) -> Set[str]:
        """Get categories that have already been processed."""
        completed = set()
        for doc in self.dimensions_col.find({}, {"_id": 1}):
            completed.add(doc["_id"])
        return completed
    
    def build_asin_to_category_map(self, leaf_categories: Set[str]) -> Dict[str, str]:
        """
        Build ASIN → leaf category mapping from metadata.
        This is the slowest part - only needs to be done once.
        """
        # Check if we already have this cached
        cache_path = Path(self.cache_dir) / "asin_to_category.json"
        if cache_path.exists():
            log.info("Loading cached ASIN mapping", path=str(cache_path))
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        log.info("Building ASIN to category mapping (this takes ~30 min)...")
        
        # Download metadata
        meta_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=META_FILENAME,
            repo_type="dataset",
            cache_dir=self.cache_dir,
        )
        
        asin_map = {}
        scanned = 0
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    parent_asin = data.get("parent_asin")
                    categories = data.get("categories", [])
                    
                    # Find the deepest (most specific) matching category
                    for cat in reversed(categories):
                        if cat in leaf_categories:
                            asin_map[parent_asin] = cat
                            break
                    
                    scanned += 1
                    if scanned % 500000 == 0:
                        log.info("Scanning metadata", scanned=f"{scanned:,}", mapped=f"{len(asin_map):,}")
                        
                except json.JSONDecodeError:
                    continue
        
        log.info("ASIN mapping complete", total=f"{len(asin_map):,}")
        
        # Cache it
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(asin_map, f)
        
        return asin_map
    
    def stream_and_aggregate_reviews(
        self,
        asin_to_category: Dict[str, str],
        target_categories: Optional[Set[str]] = None,
    ) -> Dict[str, OpinionAggregator]:
        """
        Stream all reviews and aggregate by category.
        Returns a dict of category -> aggregator.
        """
        log.info("Downloading reviews file...")
        reviews_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=REVIEW_FILENAME,
            repo_type="dataset",
            cache_dir=self.cache_dir,
        )
        
        # One aggregator per category
        aggregators: Dict[str, OpinionAggregator] = defaultdict(OpinionAggregator)
        category_review_counts: Dict[str, int] = defaultdict(int)
        
        log.info("Streaming and extracting from reviews...")
        processed = 0
        skipped_no_category = 0
        skipped_max_reached = 0
        
        with open(reviews_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    parent_asin = data.get("parent_asin")
                    text = data.get("text", "")
                    
                    # Skip short reviews
                    if not text or len(text) < 30:
                        continue
                    
                    # Get category
                    category = asin_to_category.get(parent_asin)
                    if not category:
                        skipped_no_category += 1
                        continue
                    
                    # Filter to target categories if specified
                    if target_categories and category not in target_categories:
                        continue
                    
                    # Check if we've hit the limit for this category
                    if category_review_counts[category] >= self.max_reviews_per_category:
                        skipped_max_reached += 1
                        continue
                    
                    # Extract opinion units
                    units = self.extractor.extract(text)
                    if units:
                        aggregators[category].add_units(category, units, raw_review=text)
                    
                    category_review_counts[category] += 1
                    processed += 1
                    
                    if processed % 100000 == 0:
                        log.info("Processing reviews",
                                processed=f"{processed:,}",
                                categories=len(aggregators),
                                skipped_no_cat=f"{skipped_no_category:,}")
                        
                except json.JSONDecodeError:
                    continue
        
        log.info("Review processing complete",
                total_processed=f"{processed:,}",
                categories=len(aggregators),
                skipped_no_category=f"{skipped_no_category:,}",
                skipped_max_reached=f"{skipped_max_reached:,}")
        
        self.stats["total_reviews"] = processed
        return aggregators
    
    def mine_category_dimensions(
        self,
        category: str,
        aggregator: OpinionAggregator,
    ) -> Optional[Dict]:
        """
        Call Groq to mine dimensions for a single category.
        """
        # Get semantically clustered data
        try:
            prompt_data = aggregator.get_semantic_prompt_data(category, limit=150)
        except Exception as e:
            # Fall back to regular aggregation if clustering fails
            log.warning("Semantic clustering failed, using raw", category=category, error=str(e))
            prompt_data = aggregator.get_aggregated_prompt_data(category, limit=150)
        
        stats = aggregator.get_stats()
        
        prompt = f"""You are analyzing aggregated customer feedback for "{category}" products.

The data below has been semantically grouped - similar expressions are merged with combined counts.

{prompt_data}

Based on this analysis, identify the "Hidden Dimensions" - qualitative product attributes that 
customers frequently discuss but are NOT in standard product specs.

For each dimension:
1. name: Clear attribute name specific to {category}
2. importance: "High" (>5%), "Medium" (1-5%), or "Low" (<1%)  
3. description: What this measures and why customers care
4. positive_vocabulary: 5-8 words/phrases when satisfied
5. negative_vocabulary: 5-8 words/phrases when dissatisfied
6. recommendation: How a retailer should use this (filter, badge, warning)

Focus on 8-12 most impactful dimensions specific to {category}.

Respond with a JSON object containing a "dimensions" array."""

        # Rate limit
        self.rate_limiter.wait()
        
        try:
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": f"You are an expert e-commerce analyst specializing in {category}. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=3000,
                temperature=0.3,
            )
            
            result = response.choices[0].message.content
            dimensions = json.loads(result)
            
            return {
                "category": category,
                "reviews_analyzed": stats["total_reviews"],
                "unique_patterns": stats["unique_units"],
                "dimensions": dimensions.get("dimensions", []),
                "tokens_used": response.usage.total_tokens,
                "mined_at": datetime.utcnow(),
            }
            
        except Exception as e:
            log.error("Groq API error", category=category, error=str(e))
            self.stats["errors"].append({"category": category, "error": str(e)})
            return None
    
    def save_dimensions(self, category: str, data: Dict):
        """Save mined dimensions to MongoDB."""
        self.dimensions_col.update_one(
            {"_id": category},
            {"$set": {
                "_id": category,
                **data,
            }},
            upsert=True,
        )
    
    def run(
        self,
        leaf_categories_path: str,
        test_count: Optional[int] = None,
        resume: bool = False,
    ):
        """
        Run the full pipeline.
        
        Args:
            leaf_categories_path: Path to clothing_leaf_categories.txt
            test_count: If set, only process this many categories (for testing)
            resume: If True, skip already-completed categories
        """
        print("=" * 70)
        print("🚀 PRISM MINER - Full Pipeline")
        print("=" * 70)
        print(f"   Model: {GROQ_MODEL}")
        print(f"   Max reviews/category: {self.max_reviews_per_category:,}")
        print(f"   Rate limit: {RATE_LIMIT_PER_MIN}/min")
        print(f"   Test mode: {test_count if test_count else 'OFF'}")
        print("=" * 70)
        
        # Step 1: Load leaf categories
        log.info("Step 1: Loading leaf categories...")
        leaf_categories = self.load_leaf_categories(leaf_categories_path)
        category_names = set(leaf_categories.keys())
        
        # Get completed categories if resuming
        completed = set()
        if resume:
            completed = self.get_completed_categories()
            log.info("Resuming from checkpoint", completed=len(completed))
        
        # Filter to uncompleted
        categories_to_process = category_names - completed
        
        # Limit for testing
        if test_count:
            # Sort by review count (most popular first) for testing
            sorted_cats = sorted(leaf_categories.items(), key=lambda x: -x[1])
            categories_to_process = set(cat for cat, _ in sorted_cats[:test_count])
            categories_to_process -= completed
        
        log.info("Categories to process", 
                total=len(category_names),
                completed=len(completed),
                remaining=len(categories_to_process))
        
        if not categories_to_process:
            log.info("All categories already processed!")
            return
        
        # Step 2: Build ASIN mapping
        log.info("Step 2: Building ASIN to category mapping...")
        asin_to_category = self.build_asin_to_category_map(category_names)
        
        # Step 3: Stream and aggregate reviews
        log.info("Step 3: Streaming and extracting reviews...")
        aggregators = self.stream_and_aggregate_reviews(
            asin_to_category,
            target_categories=categories_to_process,
        )
        
        # Step 4: Mine dimensions for each category
        log.info("Step 4: Mining dimensions with Groq...")
        
        total = len(aggregators)
        processed = 0
        
        for category, aggregator in aggregators.items():
            stats = aggregator.get_stats()
            
            # Skip categories with too few reviews
            if stats["total_reviews"] < MIN_REVIEWS_PER_CATEGORY:
                log.info("Skipping (too few reviews)", 
                        category=category, 
                        reviews=stats["total_reviews"])
                self.stats["categories_skipped"] += 1
                continue
            
            processed += 1
            log.info(f"Processing [{processed}/{total}]",
                    category=category,
                    reviews=stats["total_reviews"],
                    patterns=stats["unique_units"])
            
            # Mine dimensions
            result = self.mine_category_dimensions(category, aggregator)
            
            if result:
                self.save_dimensions(category, result)
                dims_count = len(result.get("dimensions", []))
                log.info(f"✅ Saved",
                        category=category,
                        dimensions=dims_count,
                        tokens=result.get("tokens_used", 0))
                self.stats["categories_processed"] += 1
                self.stats["total_patterns"] += stats["unique_units"]
            else:
                log.warning("❌ Failed", category=category)
        
        # Final summary
        self.stats["finished_at"] = datetime.utcnow()
        duration = (self.stats["finished_at"] - self.stats["started_at"]).total_seconds()
        
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETE")
        print("=" * 70)
        print(f"   Duration: {duration/60:.1f} minutes")
        print(f"   Categories processed: {self.stats['categories_processed']}")
        print(f"   Categories skipped: {self.stats['categories_skipped']}")
        print(f"   Total reviews: {self.stats['total_reviews']:,}")
        print(f"   Total patterns: {self.stats['total_patterns']:,}")
        print(f"   Errors: {len(self.stats['errors'])}")
        print("=" * 70)
        
        # Save progress
        self.progress_col.insert_one({
            "run_id": self.stats["started_at"].isoformat(),
            **self.stats,
            "finished_at": self.stats["finished_at"],
        })
        
        self.mongo_client.close()


def main():
    parser = argparse.ArgumentParser(description="Run full dimension mining pipeline")
    parser.add_argument("--test", type=int, help="Test with N categories only")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max-reviews", type=int, default=10000, 
                       help="Max reviews per category (default: 10000)")
    args = parser.parse_args()
    
    # Check env vars
    mongodb_uri = os.getenv("MONGODB_URI")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not mongodb_uri:
        print("❌ MONGODB_URI not set!")
        sys.exit(1)
    if not groq_key:
        print("❌ GROQ_API_KEY not set!")
        sys.exit(1)
    
    # Find categories file
    categories_path = Path(__file__).parent.parent.parent / "datasets" / "clothing_leaf_categories.txt"
    if not categories_path.exists():
        categories_path = Path(__file__).parent.parent / "data" / "categories" / "clothing_leaf_categories.txt"
    
    if not categories_path.exists():
        print(f"❌ Categories file not found: {categories_path}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = FullPipeline(
        mongodb_uri=mongodb_uri,
        groq_api_key=groq_key,
        max_reviews_per_category=args.max_reviews,
    )
    
    pipeline.run(
        leaf_categories_path=str(categories_path),
        test_count=args.test,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
