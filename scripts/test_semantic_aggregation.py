"""
Test Semantic Aggregation: Before vs After

Shows how embedding-based clustering merges similar patterns:
  - "runs small", "too tight", "size up" → one cluster with combined count

Usage:
    export MONGODB_URI="mongodb://..."
    export GROQ_API_KEY="gsk_..."
    python3 prism_miner/scripts/test_semantic_aggregation.py --count 500
"""

import os
import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pymongo import MongoClient
from groq import Groq
from prism_miner.services.opinion_extractor import OpinionUnitExtractor
from prism_miner.services.aggregator import OpinionAggregator

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def run_comparison(count: int = 500):
    print("=" * 70)
    print(f"🧪 SEMANTIC AGGREGATION - Before vs After ({count} reviews)")
    print("=" * 70)
    
    # Check env vars
    mongo_uri = os.getenv("MONGODB_URI")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not mongo_uri:
        print("❌ MONGODB_URI not set!")
        return
    if not groq_key:
        print("❌ GROQ_API_KEY not set!")
        return
    
    # Connect to MongoDB
    print("\n📦 Connecting to MongoDB...")
    client = MongoClient(mongo_uri)
    db = client.get_database("prism_miner")
    reviews_col = db["reviews"]
    
    total_reviews = reviews_col.count_documents({})
    print(f"   ✅ Connected! Found {total_reviews:,} reviews")
    
    # Initialize services
    print("\n📦 Initializing spaCy + Embeddings...")
    extractor = OpinionUnitExtractor()
    aggregator = OpinionAggregator()
    
    # Load reviews and extract
    print(f"\n📥 Processing {count} reviews...")
    
    cursor = reviews_col.find().limit(count)
    for i, doc in enumerate(cursor):
        text = doc.get("text", "")
        category = doc.get("category", "Clothing")
        
        units = extractor.extract(text)
        if units:
            aggregator.add_units(category, units, raw_review=text)
        
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{count}...")
    
    stats = aggregator.get_stats()
    print(f"\n✅ Extraction complete")
    print(f"   • Unique patterns: {stats['unique_units']}")
    print(f"   • Total occurrences: {stats['total_occurrences']}")
    
    # Get categories
    categories = aggregator.get_all_categories()
    category = categories[0] if categories else "Clothing"
    
    # ===== BEFORE: Raw Aggregation =====
    print("\n" + "=" * 70)
    print("📊 BEFORE: Raw Aggregation (no semantic grouping)")
    print("=" * 70)
    
    raw_units = aggregator.get_top_units(category, limit=20)
    print(f"\nTop 20 patterns (raw):")
    for item in raw_units:
        print(f"   {item['rank']:>3}. {item['unit']:<40} (count: {item['count']})")
    
    # ===== AFTER: Semantic Aggregation =====
    print("\n" + "=" * 70)
    print("📊 AFTER: Semantic Aggregation (similar patterns merged)")
    print("=" * 70)
    
    semantic_units = aggregator.get_semantically_grouped_units(category, limit=20)
    print(f"\nTop 20 patterns (semantically grouped):")
    for item in semantic_units:
        cluster_info = f"(+{item['cluster_size']-1} merged)" if item['cluster_size'] > 1 else ""
        print(f"   {item['rank']:>3}. {item['unit']:<40} (count: {item['count']}) {cluster_info}")
        if item['variations']:
            print(f"        └─ merged: {', '.join(item['variations'][:4])}")
    
    # ===== Show the Improvement =====
    print("\n" + "=" * 70)
    print("📈 IMPROVEMENT SUMMARY")
    print("=" * 70)
    
    comparison = aggregator.compare_raw_vs_semantic(category, limit=50)
    
    print(f"\n   Raw patterns:      {comparison['raw']['count']}")
    print(f"   Semantic clusters: {comparison['semantic']['count']}")
    print(f"   Compression:       {comparison['raw']['count'] / max(comparison['semantic']['count'], 1):.1f}x")
    
    # Show interesting merges
    print(f"\n   🔗 Notable Merges (patterns grouped together):")
    for cluster in comparison['semantic']['clusters_with_variations'][:5]:
        if cluster['merged'] > 1:
            print(f"      • \"{cluster['pattern']}\" absorbed {cluster['merged']-1} variations:")
            print(f"        {cluster['variations'][:3]}")
    
    # ===== Test with Groq =====
    print("\n" + "=" * 70)
    print(f"🤖 Testing with Groq ({GROQ_MODEL})")
    print("=" * 70)
    
    # Get semantic prompt data
    semantic_prompt = aggregator.get_semantic_prompt_data(category, limit=100)
    
    prompt = f"""You are analyzing aggregated customer feedback for "{category}" products.

The data below has been SEMANTICALLY GROUPED - similar expressions have been merged into clusters.
For example, "runs small", "too tight", and "size up" are grouped together with combined counts.

{semantic_prompt}

Based on this semantically-organized analysis, identify the "Hidden Dimensions" - qualitative 
product attributes that customers frequently discuss.

For each dimension:
1. name: Clear attribute name
2. importance: "High" (>5%), "Medium" (1-5%), or "Low" (<1%)
3. description: What this measures
4. positive_vocabulary: 5-8 words/phrases when satisfied
5. negative_vocabulary: 5-8 words/phrases when dissatisfied
6. recommendation: How a retailer should use this

Focus on 8-12 most impactful dimensions.

Respond with a JSON object containing a "dimensions" array."""

    groq_client = Groq(api_key=groq_key)
    
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert e-commerce analyst. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=3000,
            temperature=0.3,
        )
        
        result = response.choices[0].message.content
        dimensions = json.loads(result)
        
        print("\n🎯 MINED DIMENSIONS (from semantic clusters):")
        print("-" * 50)
        
        for dim in dimensions.get("dimensions", [])[:10]:
            print(f"\n📌 {dim.get('name', 'Unknown')} [{dim.get('importance', '?')}]")
            print(f"   {dim.get('description', '')[:80]}...")
            if dim.get('positive_vocabulary'):
                pos = dim['positive_vocabulary']
                if isinstance(pos, list):
                    print(f"   ✓ {', '.join(str(p) for p in pos[:4])}")
            if dim.get('negative_vocabulary'):
                neg = dim['negative_vocabulary']
                if isinstance(neg, list):
                    print(f"   ✗ {', '.join(str(n) for n in neg[:4])}")
        
        # Save results
        output_path = Path(__file__).parent.parent / "output" / "semantic_dimensions.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                "reviews_processed": count,
                "raw_patterns": comparison['raw']['count'],
                "semantic_clusters": comparison['semantic']['count'],
                "compression_ratio": comparison['raw']['count'] / max(comparison['semantic']['count'], 1),
                "notable_merges": comparison['semantic']['clusters_with_variations'][:10],
                "dimensions": dimensions,
            }, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
        
        # Token usage
        usage = response.usage
        cost = (usage.prompt_tokens * 0.59 + usage.completion_tokens * 0.79) / 1_000_000 * 0.5
        print(f"\n📈 Tokens: {usage.total_tokens:,} | Est. cost: ${cost:.4f}")
        
    except Exception as e:
        print(f"\n❌ Groq Error: {e}")
        import traceback
        traceback.print_exc()
    
    client.close()


def main():
    parser = argparse.ArgumentParser(description="Test semantic aggregation")
    parser.add_argument("--count", type=int, default=500, help="Reviews to process")
    args = parser.parse_args()
    run_comparison(count=args.count)


if __name__ == "__main__":
    main()
