"""
BATCH PIPELINE: Mine Hidden Dimensions for ALL 2,195 Categories using Groq Batch API

Batch API benefits:
- 50% cheaper than live API
- No TPD limits (processes async)
- All 2,195 categories in ONE 24h batch

Pipeline:
1. Download reviews from HuggingFace
2. Map reviews to leaf categories  
3. Extract & cluster opinion units
4. Create batch file with all category requests
5. Submit to Groq Batch API
6. Poll for completion
7. Download and parse results to MongoDB

Usage:
    # Step 1: Prepare batch (extract, cluster, create batch file)
    python3 prism_miner/scripts/run_batch_pipeline.py prepare --max-reviews 10000
    
    # Step 2: Submit batch
    python3 prism_miner/scripts/run_batch_pipeline.py submit
    
    # Step 3: Check status
    python3 prism_miner/scripts/run_batch_pipeline.py status
    
    # Step 4: Download results
    python3 prism_miner/scripts/run_batch_pipeline.py download
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import structlog
from pymongo import MongoClient
from huggingface_hub import hf_hub_download
from groq import Groq

from services.opinion_extractor import OpinionUnitExtractor
from services.aggregator import OpinionAggregator

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
OUTPUT_DIR = Path(__file__).parent.parent / "output"
MIN_REVIEWS_PER_CATEGORY = 50


class BatchPipeline:
    """
    Batch pipeline for mining all categories at once.
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
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize clients
        self.mongo_client = MongoClient(mongodb_uri)
        self.db = self.mongo_client.get_database("prism_miner")
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Collections
        self.dimensions_col = self.db["category_dimensions"]
        self.batch_jobs_col = self.db["batch_jobs"]
        
        # Services
        self.extractor = OpinionUnitExtractor()
    
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
    
    def build_asin_to_category_map(self, leaf_categories: Set[str]) -> Dict[str, str]:
        """Build ASIN → leaf category mapping from metadata."""
        cache_path = Path(self.cache_dir) / "asin_to_category.json"
        if cache_path.exists():
            log.info("Loading cached ASIN mapping")
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        log.info("Building ASIN to category mapping...")
        
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
        
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(asin_map, f)
        
        return asin_map
    
    def extract_and_aggregate(
        self,
        asin_to_category: Dict[str, str],
    ) -> Dict[str, OpinionAggregator]:
        """Stream reviews and aggregate by category."""
        log.info("Downloading reviews file...")
        reviews_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=REVIEW_FILENAME,
            repo_type="dataset",
            cache_dir=self.cache_dir,
        )
        
        aggregators: Dict[str, OpinionAggregator] = defaultdict(OpinionAggregator)
        category_review_counts: Dict[str, int] = defaultdict(int)
        
        log.info("Streaming and extracting...")
        processed = 0
        
        with open(reviews_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    parent_asin = data.get("parent_asin")
                    text = data.get("text", "")
                    
                    if not text or len(text) < 30:
                        continue
                    
                    category = asin_to_category.get(parent_asin)
                    if not category:
                        continue
                    
                    if category_review_counts[category] >= self.max_reviews_per_category:
                        continue
                    
                    units = self.extractor.extract(text)
                    if units:
                        aggregators[category].add_units(category, units, raw_review=text)
                    
                    category_review_counts[category] += 1
                    processed += 1
                    
                    if processed % 100000 == 0:
                        log.info("Processing", processed=f"{processed:,}", categories=len(aggregators))
                        
                except json.JSONDecodeError:
                    continue
        
        log.info("Extraction complete", processed=f"{processed:,}", categories=len(aggregators))
        return aggregators
    
    def create_batch_file(
        self,
        aggregators: Dict[str, OpinionAggregator],
    ) -> str:
        """Create JSONL batch file for Groq API."""
        batch_file_path = self.output_dir / "batch_input.jsonl"
        
        requests = []
        skipped = 0
        
        for category, aggregator in aggregators.items():
            stats = aggregator.get_stats()
            
            if stats["total_reviews"] < MIN_REVIEWS_PER_CATEGORY:
                skipped += 1
                continue
            
            # Get semantic prompt data (or fall back to raw)
            try:
                prompt_data = aggregator.get_semantic_prompt_data(category, limit=150)
            except Exception:
                prompt_data = aggregator.get_aggregated_prompt_data(category, limit=150)
            
            prompt = f"""You are analyzing aggregated customer feedback for "{category}" products.

The data has been semantically grouped - similar expressions are merged with combined counts.

{prompt_data}

Identify the "Hidden Dimensions" - qualitative product attributes customers frequently discuss 
but are NOT in standard product specs.

For each dimension:
1. name: Clear attribute name specific to {category}
2. importance: "High" (>5%), "Medium" (1-5%), or "Low" (<1%)
3. description: What this measures and why customers care
4. positive_vocabulary: 5-8 words/phrases when satisfied
5. negative_vocabulary: 5-8 words/phrases when dissatisfied
6. recommendation: How a retailer should use this (filter, badge, warning)

Focus on 8-12 most impactful dimensions.

Respond with JSON containing a "dimensions" array."""

            request = {
                "custom_id": f"cat_{category.replace(' ', '_').replace('/', '_').replace('&', 'and')}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": f"You are an e-commerce analyst for {category}. Respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 3000,
                    "temperature": 0.3,
                }
            }
            requests.append(request)
        
        # Write batch file
        with open(batch_file_path, 'w') as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
        
        log.info("Batch file created", 
                path=str(batch_file_path), 
                requests=len(requests),
                skipped=skipped)
        
        # Save metadata
        metadata = {
            "created_at": datetime.utcnow().isoformat(),
            "model": GROQ_MODEL,
            "total_categories": len(requests),
            "skipped_categories": skipped,
            "max_reviews_per_category": self.max_reviews_per_category,
        }
        with open(self.output_dir / "batch_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(batch_file_path)
    
    def submit_batch(self, batch_file_path: str) -> str:
        """Upload batch file and submit job."""
        log.info("Uploading batch file...")
        
        with open(batch_file_path, 'rb') as f:
            file_response = self.groq_client.files.create(file=f, purpose="batch")
        
        file_id = file_response.id
        log.info("File uploaded", file_id=file_id)
        
        log.info("Creating batch job...")
        batch_response = self.groq_client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        
        batch_id = batch_response.id
        log.info("Batch job created", batch_id=batch_id)
        
        # Save to MongoDB
        self.batch_jobs_col.insert_one({
            "_id": batch_id,
            "file_id": file_id,
            "status": "submitted",
            "created_at": datetime.utcnow(),
            "model": GROQ_MODEL,
        })
        
        # Save locally
        with open(self.output_dir / "batch_job.json", 'w') as f:
            json.dump({
                "batch_id": batch_id,
                "file_id": file_id,
                "created_at": datetime.utcnow().isoformat(),
            }, f, indent=2)
        
        print(f"\n✅ Batch submitted!")
        print(f"   Batch ID: {batch_id}")
        print(f"   Check status: python3 prism_miner/scripts/run_batch_pipeline.py status")
        
        return batch_id
    
    def check_status(self) -> Dict:
        """Check batch job status."""
        job_file = self.output_dir / "batch_job.json"
        if not job_file.exists():
            print("❌ No batch job found. Run 'prepare' and 'submit' first.")
            return {}
        
        with open(job_file, 'r') as f:
            job_info = json.load(f)
        
        batch_id = job_info["batch_id"]
        
        status = self.groq_client.batches.retrieve(batch_id)
        
        print(f"\n📊 Batch Status")
        print(f"   ID: {status.id}")
        print(f"   Status: {status.status}")
        print(f"   Created: {status.created_at}")
        
        if hasattr(status, 'request_counts') and status.request_counts:
            rc = status.request_counts
            print(f"   Progress: {rc.completed}/{rc.total} completed, {rc.failed} failed")
        
        if status.output_file_id:
            print(f"   Output file: {status.output_file_id}")
        
        if status.error_file_id:
            print(f"   Error file: {status.error_file_id}")
        
        return {
            "batch_id": status.id,
            "status": status.status,
            "output_file_id": status.output_file_id,
            "error_file_id": status.error_file_id,
        }
    
    def download_results(self):
        """Download and parse batch results."""
        status = self.check_status()
        
        if status.get("status") != "completed":
            print(f"\n⏳ Batch not complete yet. Status: {status.get('status')}")
            return
        
        output_file_id = status.get("output_file_id")
        if not output_file_id:
            print("❌ No output file available")
            return
        
        log.info("Downloading results...")
        
        response = self.groq_client.files.content(output_file_id)
        
        results_path = self.output_dir / "batch_results.jsonl"
        with open(results_path, 'wb') as f:
            f.write(response.content)
        
        log.info("Results downloaded", path=str(results_path))
        
        # Parse and save to MongoDB
        log.info("Parsing results and saving to MongoDB...")
        
        saved = 0
        errors = 0
        
        with open(results_path, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    custom_id = result.get("custom_id", "")
                    category = custom_id.replace("cat_", "").replace("_", " ")
                    
                    if result.get("error"):
                        log.warning("Request failed", category=category, error=result["error"])
                        errors += 1
                        continue
                    
                    response_body = result.get("response", {}).get("body", {})
                    choices = response_body.get("choices", [])
                    
                    if choices:
                        content = choices[0].get("message", {}).get("content", "{}")
                        dimensions_data = json.loads(content)
                        
                        self.dimensions_col.update_one(
                            {"_id": category},
                            {"$set": {
                                "_id": category,
                                "category": category,
                                "dimensions": dimensions_data.get("dimensions", []),
                                "mined_at": datetime.utcnow(),
                                "source": "groq_batch_api",
                                "model": GROQ_MODEL,
                            }},
                            upsert=True,
                        )
                        saved += 1
                        
                except (json.JSONDecodeError, KeyError) as e:
                    log.warning("Parse error", error=str(e))
                    errors += 1
                    continue
        
        print(f"\n✅ Results processed!")
        print(f"   Saved: {saved} categories")
        print(f"   Errors: {errors}")
        
        # Export to JSON
        all_dimensions = {}
        for doc in self.dimensions_col.find():
            all_dimensions[doc["_id"]] = {
                "dimensions": doc.get("dimensions", []),
                "mined_at": doc.get("mined_at", "").isoformat() if doc.get("mined_at") else None,
            }
        
        export_path = self.output_dir / "all_dimensions.json"
        with open(export_path, 'w') as f:
            json.dump(all_dimensions, f, indent=2)
        
        print(f"   Exported to: {export_path}")
    
    def prepare(self, leaf_categories_path: str, test_count: Optional[int] = None):
        """Prepare batch file (extract, cluster, create batch)."""
        print("=" * 70)
        print("🚀 BATCH PIPELINE - Prepare Phase")
        print("=" * 70)
        
        # Load categories
        leaf_categories = self.load_leaf_categories(leaf_categories_path)
        category_names = set(leaf_categories.keys())
        
        if test_count:
            sorted_cats = sorted(leaf_categories.items(), key=lambda x: -x[1])
            category_names = set(cat for cat, _ in sorted_cats[:test_count])
            log.info(f"Test mode: processing top {test_count} categories")
        
        # Build ASIN mapping
        asin_to_category = self.build_asin_to_category_map(category_names)
        
        # Filter to only target categories
        asin_to_category = {k: v for k, v in asin_to_category.items() if v in category_names}
        
        # Extract and aggregate
        aggregators = self.extract_and_aggregate(asin_to_category)
        
        # Create batch file
        batch_file = self.create_batch_file(aggregators)
        
        print(f"\n✅ Preparation complete!")
        print(f"   Batch file: {batch_file}")
        print(f"\n   Next: python3 prism_miner/scripts/run_batch_pipeline.py submit")


def main():
    parser = argparse.ArgumentParser(description="Batch pipeline for dimension mining")
    parser.add_argument("action", choices=["prepare", "submit", "status", "download"],
                       help="Pipeline action")
    parser.add_argument("--test", type=int, help="Test with N categories")
    parser.add_argument("--max-reviews", type=int, default=10000,
                       help="Max reviews per category")
    args = parser.parse_args()
    
    mongodb_uri = os.getenv("MONGODB_URI")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not mongodb_uri or not groq_key:
        print("❌ Set MONGODB_URI and GROQ_API_KEY")
        sys.exit(1)
    
    categories_path = Path(__file__).parent.parent.parent / "datasets" / "clothing_leaf_categories.txt"
    if not categories_path.exists():
        categories_path = Path(__file__).parent.parent / "data" / "categories" / "clothing_leaf_categories.txt"
    
    pipeline = BatchPipeline(
        mongodb_uri=mongodb_uri,
        groq_api_key=groq_key,
        max_reviews_per_category=args.max_reviews,
    )
    
    if args.action == "prepare":
        pipeline.prepare(str(categories_path), test_count=args.test)
    elif args.action == "submit":
        batch_file = pipeline.output_dir / "batch_input.jsonl"
        if not batch_file.exists():
            print("❌ Run 'prepare' first")
            sys.exit(1)
        pipeline.submit_batch(str(batch_file))
    elif args.action == "status":
        pipeline.check_status()
    elif args.action == "download":
        pipeline.download_results()


if __name__ == "__main__":
    main()
