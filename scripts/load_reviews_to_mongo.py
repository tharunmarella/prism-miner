"""
Download Amazon Reviews from HuggingFace and load to MongoDB.

Usage:
    # Set MongoDB connection
    export MONGODB_URI="mongodb://localhost:27017" 
    
    # Or use MongoDB Atlas
    export MONGODB_URI="mongodb+srv://user:pass@cluster.mongodb.net"
    
    # Run script
    python3 prism_miner/scripts/load_reviews_to_mongo.py --count 1000
"""

import os
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from huggingface_hub import hf_hub_download
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, BulkWriteError

# Config
REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
REVIEW_FILENAME = "raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl"
META_FILENAME = "raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl"


def get_mongo_client():
    """Connect to MongoDB."""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        print("❌ MONGODB_URI not set!")
        print("   For local: export MONGODB_URI='mongodb://localhost:27017'")
        print("   For Atlas: export MONGODB_URI='mongodb+srv://user:pass@cluster.mongodb.net'")
        sys.exit(1)
    
    try:
        client = MongoClient(uri)
        client.admin.command('ping')
        print(f"✅ Connected to MongoDB")
        return client
    except ConnectionFailure as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        sys.exit(1)


def download_file(filename: str, desc: str) -> str:
    """Download a file from HuggingFace Hub."""
    print(f"\n📥 Downloading {desc}...")
    print(f"   File: {filename}")
    print("   (This may take a while for large files)")
    
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        cache_dir="/tmp/hf_cache",
    )
    print(f"   ✅ Downloaded to: {local_path}")
    return local_path


def load_reviews(
    count: int = 1000,
    category_filter: list = None,
    with_metadata: bool = True,
):
    """
    Download reviews and load to MongoDB.
    
    Args:
        count: Number of reviews to load
        category_filter: Optional list of categories to filter by (e.g., ["Dresses", "T-Shirts"])
        with_metadata: Whether to also download product metadata
    """
    client = get_mongo_client()
    db = client.get_database("prism_miner")
    
    # Collections
    reviews_col = db["reviews"]
    products_col = db["products"]
    
    # Create indexes
    print("\n📊 Setting up indexes...")
    reviews_col.create_index("asin")
    reviews_col.create_index("parent_asin")
    reviews_col.create_index("rating")
    reviews_col.create_index("category")
    products_col.create_index("parent_asin", unique=True)
    print("   ✅ Indexes created")
    
    # Step 1: If we have category filters, first build ASIN map from metadata
    asin_to_category = {}
    if category_filter and with_metadata:
        meta_path = download_file(META_FILENAME, "product metadata")
        print(f"\n🔍 Scanning metadata for categories: {category_filter}")
        
        products_to_insert = []
        scanned = 0
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    parent_asin = data.get("parent_asin")
                    title = data.get("title", "").lower()
                    categories = data.get("categories", [])
                    
                    # Check if product matches any filter
                    matched_cat = None
                    for cat_filter in category_filter:
                        if cat_filter.lower() in title or cat_filter in categories:
                            matched_cat = cat_filter
                            break
                    
                    if matched_cat:
                        asin_to_category[parent_asin] = matched_cat
                        products_to_insert.append({
                            "parent_asin": parent_asin,
                            "title": data.get("title"),
                            "category": matched_cat,
                            "categories": categories,
                            "price": data.get("price"),
                            "average_rating": data.get("average_rating"),
                            "rating_number": data.get("rating_number"),
                            "loaded_at": datetime.utcnow(),
                        })
                    
                    scanned += 1
                    if scanned % 500000 == 0:
                        print(f"   Scanned {scanned:,} products, found {len(asin_to_category):,} matches...")
                        
                except json.JSONDecodeError:
                    continue
        
        print(f"   ✅ Found {len(asin_to_category):,} products matching filters")
        
        # Insert products
        if products_to_insert:
            print(f"\n💾 Inserting {len(products_to_insert):,} products...")
            try:
                # Use ordered=False to continue on duplicates
                result = products_col.insert_many(products_to_insert, ordered=False)
                print(f"   ✅ Inserted {len(result.inserted_ids):,} products")
            except BulkWriteError as e:
                inserted = e.details.get('nInserted', 0)
                print(f"   ⚠️  Inserted {inserted:,} products (some duplicates skipped)")
    
    # Step 2: Download and stream reviews
    reviews_path = download_file(REVIEW_FILENAME, "reviews")
    
    print(f"\n📖 Loading {count:,} reviews...")
    
    reviews_to_insert = []
    loaded = 0
    skipped = 0
    category_counts = defaultdict(int)
    
    with open(reviews_path, 'r', encoding='utf-8') as f:
        for line in f:
            if loaded >= count:
                break
            
            try:
                data = json.loads(line)
                parent_asin = data.get("parent_asin")
                text = data.get("text", "")
                
                # Skip empty reviews
                if not text or len(text) < 20:
                    skipped += 1
                    continue
                
                # Apply category filter if specified
                if category_filter and asin_to_category:
                    if parent_asin not in asin_to_category:
                        continue
                    category = asin_to_category[parent_asin]
                else:
                    category = "Clothing_Shoes_and_Jewelry"
                
                review_doc = {
                    "asin": data.get("asin"),
                    "parent_asin": parent_asin,
                    "user_id": data.get("user_id"),
                    "rating": data.get("rating"),
                    "title": data.get("title"),
                    "text": text,
                    "helpful_vote": data.get("helpful_vote", 0),
                    "verified_purchase": data.get("verified_purchase", False),
                    "timestamp": data.get("timestamp"),
                    "category": category,
                    "loaded_at": datetime.utcnow(),
                }
                
                reviews_to_insert.append(review_doc)
                category_counts[category] += 1
                loaded += 1
                
                # Batch insert every 500 reviews
                if len(reviews_to_insert) >= 500:
                    reviews_col.insert_many(reviews_to_insert)
                    print(f"   Loaded {loaded:,}/{count:,} reviews...")
                    reviews_to_insert = []
                    
            except json.JSONDecodeError:
                continue
    
    # Insert remaining
    if reviews_to_insert:
        reviews_col.insert_many(reviews_to_insert)
    
    # Summary
    print(f"\n{'='*60}")
    print("✅ LOADING COMPLETE")
    print(f"{'='*60}")
    print(f"   • Reviews loaded: {loaded:,}")
    print(f"   • Reviews skipped (too short): {skipped:,}")
    print(f"\n   Reviews by category:")
    for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"      {cat}: {cnt:,}")
    
    # Show collection stats
    total_reviews = reviews_col.count_documents({})
    total_products = products_col.count_documents({})
    print(f"\n   📊 MongoDB Collections:")
    print(f"      reviews: {total_reviews:,} documents")
    print(f"      products: {total_products:,} documents")
    
    client.close()
    return loaded


def main():
    parser = argparse.ArgumentParser(description="Load Amazon reviews to MongoDB")
    parser.add_argument("--count", type=int, default=1000, help="Number of reviews to load")
    parser.add_argument("--categories", type=str, nargs="+", 
                       help="Filter by categories (e.g., --categories Dresses T-Shirts)")
    parser.add_argument("--no-metadata", action="store_true", 
                       help="Skip downloading product metadata")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 PRISM MINER - Load Reviews to MongoDB")
    print("=" * 60)
    
    load_reviews(
        count=args.count,
        category_filter=args.categories,
        with_metadata=not args.no_metadata,
    )


if __name__ == "__main__":
    main()
