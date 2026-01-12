"""
Test pipeline: MongoDB reviews → spaCy extraction → Groq LIVE API

Usage:
    export MONGODB_URI="mongodb://..."
    export GROQ_API_KEY="gsk_..."
    python3 prism_miner/scripts/test_mongo_to_groq.py --count 100
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


def run_test(count: int = 100):
    print("=" * 60)
    print(f"🧪 PRISM MINER - MongoDB → Groq Test ({count} reviews)")
    print("=" * 60)
    
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
    print(f"   ✅ Connected! Found {total_reviews:,} reviews in database")
    
    # Initialize services
    print("\n📦 Initializing spaCy...")
    extractor = OpinionUnitExtractor()
    aggregator = OpinionAggregator()
    
    # Step 1: Load reviews from MongoDB
    print(f"\n📥 Loading {count} reviews from MongoDB...")
    
    sample_extractions = []
    reviews_with_units = 0
    
    cursor = reviews_col.find().limit(count)
    
    for i, doc in enumerate(cursor):
        text = doc.get("text", "")
        rating = doc.get("rating", 0)
        category = doc.get("category", "Clothing")
        
        # Extract opinion units
        units = extractor.extract(text)
        if units:
            aggregator.add_units(category, units, raw_review=text)
            reviews_with_units += 1
        
        # Save samples for display
        if i < 5:
            sample_extractions.append({
                "text": text[:120] + "..." if len(text) > 120 else text,
                "rating": rating,
                "units": units[:5] if units else []
            })
        
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{count}...")
    
    # Step 2: Show extraction results
    stats = aggregator.get_stats()
    print(f"\n✅ Processed {count} reviews")
    print(f"   • Reviews with opinion units: {reviews_with_units}")
    print(f"   • Unique opinion patterns: {stats['unique_units']}")
    print(f"   • Total occurrences: {stats['total_occurrences']}")
    
    print("\n📝 Sample Extractions from REAL reviews:")
    for i, sample in enumerate(sample_extractions, 1):
        print(f"\n   Review {i} (⭐{sample['rating']}): \"{sample['text']}\"")
        if sample['units']:
            print(f"   → Extracted: {sample['units']}")
        else:
            print(f"   → No patterns extracted")
    
    # Step 3: Show top patterns
    categories = aggregator.get_all_categories()
    print(f"\n📊 Categories found: {categories}")
    
    for category in categories:
        top_units = aggregator.get_top_units(category, limit=50)
        print(f"\n   Top 15 patterns for '{category}':")
        for item in top_units[:15]:
            print(f"      {item['rank']:>3}. {item['unit']:<35} (count: {item['count']}, {item['percentage']:.1f}%)")
    
    # Step 4: Call Groq LIVE API
    print(f"\n🤖 Calling Groq LIVE API ({GROQ_MODEL})...")
    
    # Use first category
    category = categories[0] if categories else "Clothing"
    aggregated_data = aggregator.get_aggregated_prompt_data(category, limit=100)
    
    prompt = f"""You are analyzing aggregated customer feedback patterns for the "{category}" product category.

Below is data from analyzing {count} REAL Amazon customer reviews. The "Count" shows how many times each opinion pattern appeared.

{aggregated_data}

Based on this real customer feedback analysis, identify the "Hidden Dimensions" - qualitative product attributes that customers frequently discuss but are NOT in standard product specs.

For each dimension, provide:
1. name: A clear attribute name (e.g., "Size Accuracy", "Fabric Softness")
2. importance: "High" (>5% of reviews), "Medium" (1-5%), or "Low" (<1%)
3. description: What this dimension measures and why customers care
4. positive_vocabulary: 5-8 words/phrases customers use when satisfied
5. negative_vocabulary: 5-8 words/phrases customers use when dissatisfied
6. recommendation: How a retailer should use this (filter, warning, badge, etc.)

Focus on the 8-12 most impactful dimensions that would help shoppers make better decisions.

Respond with a JSON object containing a "dimensions" array."""

    groq_client = Groq(api_key=groq_key)
    
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert e-commerce product analyst specializing in clothing and apparel. Respond only with valid JSON."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=3000,
            temperature=0.3,
        )
        
        result = response.choices[0].message.content
        dimensions = json.loads(result)
        
        # Step 5: Display results
        print("\n" + "=" * 60)
        print("🎯 MINED HIDDEN DIMENSIONS (from REAL reviews)")
        print("=" * 60)
        
        for dim in dimensions.get("dimensions", []):
            print(f"\n📌 {dim.get('name', 'Unknown')}")
            print(f"   Importance: {dim.get('importance', 'N/A')}")
            desc = dim.get('description', 'N/A')
            print(f"   Description: {desc[:100]}{'...' if len(desc) > 100 else ''}")
            
            if dim.get('positive_vocabulary'):
                pos = dim['positive_vocabulary']
                if isinstance(pos, list):
                    print(f"   ✓ Positive: {', '.join(str(p) for p in pos[:6])}")
                else:
                    print(f"   ✓ Positive: {pos}")
                    
            if dim.get('negative_vocabulary'):
                neg = dim['negative_vocabulary']
                if isinstance(neg, list):
                    print(f"   ✗ Negative: {', '.join(str(n) for n in neg[:6])}")
                else:
                    print(f"   ✗ Negative: {neg}")
                    
            if dim.get('recommendation'):
                rec = dim['recommendation']
                print(f"   💡 Recommendation: {rec[:80]}{'...' if len(rec) > 80 else ''}")
        
        # Save results
        output_path = Path(__file__).parent.parent / "output" / "real_dimensions.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                "source": "mongodb_real_reviews",
                "reviews_processed": count,
                "reviews_with_patterns": reviews_with_units,
                "unique_patterns": stats['unique_units'],
                "total_occurrences": stats['total_occurrences'],
                "top_patterns": aggregator.get_top_units(category, limit=30),
                "dimensions": dimensions
            }, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
        
        # Token usage and cost
        usage = response.usage
        cost_estimate = (usage.prompt_tokens * 0.59 + usage.completion_tokens * 0.79) / 1_000_000
        print(f"\n📈 Token Usage: {usage.prompt_tokens:,} prompt + {usage.completion_tokens:,} completion = {usage.total_tokens:,} total")
        print(f"   Estimated cost: ${cost_estimate:.4f}")
        
    except Exception as e:
        print(f"\n❌ Groq API Error: {e}")
        import traceback
        traceback.print_exc()
    
    client.close()


def main():
    parser = argparse.ArgumentParser(description="Test MongoDB → Groq pipeline")
    parser.add_argument("--count", type=int, default=100, help="Number of reviews to process")
    args = parser.parse_args()
    
    run_test(count=args.count)


if __name__ == "__main__":
    main()
