"""
Prism Miner - Main Pipeline (Full Scale)

Processes ALL 66M reviews:
1. Stream reviews from HuggingFace
2. Extract opinion units using spaCy
3. Aggregate by category (count frequencies)
4. Submit aggregated data to Groq Batch API
5. Parse and store results in MongoDB
"""

import os
import json
import structlog
from collections import defaultdict
from typing import Dict, List, Optional

from prism_miner.config import config
from prism_miner.services.opinion_extractor import OpinionUnitExtractor
from prism_miner.services.review_loader import ReviewLoader
from prism_miner.services.batch_api import GroqBatchService
from prism_miner.services.mongo_storage import MongoStorage
from prism_miner.services.aggregator import OpinionAggregator

log = structlog.get_logger()


def load_leaf_categories(path: str) -> set:
    """
    Loads leaf category names from the categories file.
    """
    categories = set()
    
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and parts[0].isdigit():
                categories.add(parts[1])
    
    log.info("Loaded leaf categories", count=len(categories))
    return categories


def run_pipeline(
    target_category: str = "Clothing_Shoes_and_Jewelry",
    leaf_categories_path: Optional[str] = None,
    output_dir: str = "/app/output",
    max_reviews: Optional[int] = None,  # None = process ALL
):
    """
    Main pipeline execution - processes ALL reviews.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize services
    extractor = OpinionUnitExtractor()
    loader = ReviewLoader()
    batch_service = GroqBatchService()
    mongo = MongoStorage()
    aggregator = OpinionAggregator()
    
    # Load leaf categories if provided
    leaf_categories = None
    if leaf_categories_path and os.path.exists(leaf_categories_path):
        leaf_categories = load_leaf_categories(leaf_categories_path)
    
    log.info("Starting FULL SCALE mining pipeline", 
             target_category=target_category,
             leaf_categories=len(leaf_categories) if leaf_categories else "all",
             max_reviews=max_reviews or "ALL")
    
    # Step 1: Build ASIN to leaf category mapping
    asin_to_leaf = {}
    if leaf_categories:
        log.info("Building ASIN to category mapping...")
        asin_to_leaf = loader.build_asin_to_leaf_category(target_category, leaf_categories)
        log.info("ASIN mapping complete", total_products=len(asin_to_leaf))
    
    # Step 2: Stream ALL reviews and extract + aggregate opinion units
    log.info("Streaming and extracting opinion units from ALL reviews...")
    
    processed = 0
    skipped = 0
    
    for review in loader.stream_reviews(target_category, max_reviews=max_reviews):
        asin = review.get("parent_asin") or review.get("asin")
        text = review.get("text", "")
        
        # Determine category
        if asin_to_leaf and asin in asin_to_leaf:
            category = asin_to_leaf[asin]
        elif not leaf_categories:
            category = target_category
        else:
            skipped += 1
            continue
        
        # Extract opinion units
        units = extractor.extract(text)
        if units:
            aggregator.add_units(category, units, raw_review=text)
        
        processed += 1
        
        # Progress logging
        if processed % 100000 == 0:
            stats = aggregator.get_stats()
            log.info("Progress", 
                     reviews_processed=processed,
                     reviews_skipped=skipped,
                     categories=stats["categories"],
                     unique_patterns=stats["unique_units"])
    
    # Final stats
    stats = aggregator.get_stats()
    log.info("Extraction complete", 
             total_processed=processed,
             total_skipped=skipped,
             **stats)
    
    # Save aggregation stats
    with open(os.path.join(output_dir, "aggregation_stats.json"), 'w') as f:
        json.dump({
            "processed": processed,
            "skipped": skipped,
            **stats
        }, f, indent=2)
    
    # Step 3: Create batch file with AGGREGATED data
    log.info("Creating batch file with aggregated opinion patterns...")
    
    category_data = {}
    for category in aggregator.get_all_categories():
        category_data[category] = aggregator.get_aggregated_prompt_data(category, limit=200)
    
    batch_file_path = os.path.join(output_dir, "batch_input.jsonl")
    create_aggregated_batch_file(batch_service, category_data, batch_file_path)
    
    # Step 4: Upload and submit batch job
    file_id = batch_service.upload_batch_file(batch_file_path)
    batch_id = batch_service.create_batch_job(file_id, completion_window="24h")
    
    # Save to MongoDB
    batch_metadata = {
        "categories": len(category_data),
        "total_reviews_processed": processed,
        "total_unique_patterns": stats["unique_units"],
        "target_category": target_category,
        "mode": "full_scale",
    }
    mongo.save_batch_job(batch_id, file_id, batch_metadata)
    
    # Save locally
    with open(os.path.join(output_dir, "batch_info.json"), 'w') as f:
        json.dump({
            "batch_id": batch_id,
            "file_id": file_id,
            **batch_metadata
        }, f, indent=2)
    
    log.info("Batch job submitted", batch_id=batch_id)
    
    # Step 5: Wait for completion (optional)
    if os.getenv("WAIT_FOR_COMPLETION", "false").lower() == "true":
        log.info("Waiting for batch completion...")
        status = batch_service.wait_for_completion(batch_id)
        
        if status.get("output_file_id"):
            results_path = os.path.join(output_dir, "batch_results.jsonl")
            batch_service.download_results(status["output_file_id"], results_path)
            
            results = batch_service.parse_results(results_path)
            
            mongo.update_batch_status(batch_id, "completed", status["output_file_id"])
            mongo.save_all_dimensions(results)
            
            with open(os.path.join(output_dir, "taxonomy_dimensions.json"), 'w') as f:
                json.dump(results, f, indent=2)
            
            log.info("Pipeline complete", categories_mined=len(results))
    else:
        log.info("Batch submitted. Poll for status:", batch_id=batch_id)
    
    return batch_id


def create_aggregated_batch_file(
    batch_service: GroqBatchService,
    category_data: Dict[str, str],
    output_path: str,
):
    """
    Creates a batch file with aggregated opinion data.
    """
    MINING_PROMPT = """You are analyzing aggregated customer feedback patterns for the "{category}" product category.

Below is data from analyzing THOUSANDS of real customer reviews. The "Count" shows how many times each opinion pattern appeared, and "%" shows what percentage of reviews mentioned it.

{aggregated_data}

Based on this comprehensive analysis, identify the "Hidden Dimensions" - qualitative product attributes that customers frequently discuss.

For each dimension:
1. name: A clear, title-cased attribute name (e.g., "Size Accuracy", "Fabric Breathability")
2. importance: "High" (>5% of reviews), "Medium" (1-5%), or "Low" (<1%)
3. description: What this dimension measures and why customers care
4. positive_vocabulary: Words customers use when satisfied
5. negative_vocabulary: Words customers use when dissatisfied
6. recommendation: How a retailer should use this dimension (filter, warning, etc.)

Focus on the 10-15 most impactful dimensions that would help shoppers make better decisions.

Respond with a JSON object containing a "dimensions" array."""

    with open(output_path, 'w') as f:
        for category, aggregated_data in category_data.items():
            request = {
                "custom_id": f"cat_{category.replace(' ', '_').replace('/', '_')}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": batch_service.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert e-commerce product analyst. Respond only with valid JSON."
                        },
                        {
                            "role": "user",
                            "content": MINING_PROMPT.format(
                                category=category,
                                aggregated_data=aggregated_data
                            )
                        }
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 3000,
                }
            }
            f.write(json.dumps(request) + "\n")
    
    log.info("Created aggregated batch file", path=output_path, categories=len(category_data))


def check_batch_status(batch_id: str):
    """Check the status of a submitted batch job."""
    batch_service = GroqBatchService()
    status = batch_service.check_batch_status(batch_id)
    log.info("Batch status", **status)
    return status


def download_batch_results(batch_id: str, output_dir: str = "/app/output"):
    """Download and parse completed batch results."""
    batch_service = GroqBatchService()
    mongo = MongoStorage()
    
    status = batch_service.check_batch_status(batch_id)
    
    if status["status"] != "completed":
        log.warning("Batch not yet complete", status=status["status"])
        return None
    
    if not status.get("output_file_id"):
        log.error("No output file available")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "batch_results.jsonl")
    batch_service.download_results(status["output_file_id"], results_path)
    
    results = batch_service.parse_results(results_path)
    
    mongo.update_batch_status(batch_id, "completed", status["output_file_id"])
    mongo.save_all_dimensions(results)
    
    with open(os.path.join(output_dir, "taxonomy_dimensions.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    log.info("Results saved", path=output_dir, categories=len(results))
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        if len(sys.argv) > 2:
            check_batch_status(sys.argv[2])
        else:
            print("Usage: python -m prism_miner.main status <batch_id>")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "download":
        if len(sys.argv) > 2:
            download_batch_results(sys.argv[2])
        else:
            print("Usage: python -m prism_miner.main download <batch_id>")
    
    else:
        # Run the full pipeline
        run_pipeline(
            target_category=config.TARGET_CATEGORY,
            leaf_categories_path="/app/data/categories/clothing_leaf_categories.txt",
            output_dir=config.OUTPUT_DIR,
            max_reviews=None,  # Process ALL reviews
        )
