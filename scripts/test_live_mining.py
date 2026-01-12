"""
Quick Test: 100 reviews → spaCy extraction → Groq LIVE API

Run with:
    python3 prism_miner/scripts/test_live_mining.py
"""

import os
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openai import OpenAI
from prism_miner.services.opinion_extractor import OpinionUnitExtractor
from prism_miner.services.aggregator import OpinionAggregator

# Config
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Sample clothing reviews for testing (real reviews from various sources)
SAMPLE_REVIEWS = [
    "The fabric is incredibly soft and comfortable. Fits true to size. I've washed it 5 times and it hasn't shrunk at all. Great quality for the price!",
    "Terrible quality. The zipper broke after wearing it twice. The stitching is coming apart. Complete waste of money.",
    "Love this dress! The color is exactly as shown in the picture. Runs a bit small though, so I'd recommend sizing up.",
    "The material feels cheap and scratchy against my skin. Not breathable at all - I was sweating within minutes.",
    "Perfect fit! The length is just right and the pockets are deep enough to actually use. Very flattering silhouette.",
    "Disappointed. The buttons fell off after the first wash. Quality control is clearly lacking.",
    "Super cozy and warm. Great for winter. The fleece lining is thick and soft. A bit bulky though.",
    "The color faded significantly after washing. Went from vibrant blue to a dull grey. Very disappointing.",
    "Fits perfectly! True to size. The elastic waistband is comfortable and doesn't roll down. Would buy again.",
    "Too short! I'm 5'4 and it barely covers my butt. The description said midi length. Very misleading.",
    "Amazing value for money. Looks way more expensive than it is. The stitching is neat and even.",
    "The sleeves are too tight even though the body fits fine. Uncomfortable around the armpits.",
    "Beautiful print and the fabric doesn't wrinkle easily. Perfect for travel. Packs well too.",
    "Runs extremely large. I ordered my usual size and it's swimming on me. Had to return for smaller.",
    "The material pills after just a few washes. Already looks worn out. Not durable at all.",
    "So comfortable! I wear this every day. Soft cotton, doesn't lose shape. Perfect loungewear.",
    "Cheaply made. The hem started unraveling on day one. Threads hanging everywhere. Awful.",
    "Love the style but wish the fabric was thicker. You can see through it in bright light.",
    "Great for layering. Lightweight but warm enough. The hood fits well and stays in place.",
    "The sizing chart was completely wrong. Waist measurement was off by at least 2 inches.",
    "Absolutely love this jacket! Keeps me warm without being too heavy. Pockets are well-placed.",
    "Horrible smell when I first opened the package. Had to wash three times before it was wearable.",
    "The collar is itchy and irritating. Otherwise nice shirt but can't wear it comfortably.",
    "Perfect summer dress! Light and airy. The pattern is gorgeous and gets lots of compliments.",
    "Way too long. I'm 5'8 and had to hem it significantly. Proportions seem off.",
    "Excellent quality denim. Thick and sturdy. The stretch is just right - not too tight.",
    "The colors bleed in the wash. Ruined other clothes in the same load. Very frustrating.",
    "Fits like a glove! Finally jeans that don't gap at the waist. So flattering on curves.",
    "Material is paper thin. Ripped within the first week. Do not recommend at all.",
    "Super soft hoodie! Perfect weight for fall. The drawstrings are nice quality too.",
    "Ordered medium, received something that fits like an XS. Inconsistent sizing is ridiculous.",
    "The embroidery is beautiful and well-done. This looks handmade. Very impressed with detail.",
    "Shrunk two sizes after one wash following care instructions. Now unwearable.",
    "Great workout leggings! Squat-proof and the high waist stays put. No sliding down.",
    "Buttons are cheap plastic and look tacky. Otherwise nice blouse but details matter.",
    "So breathable! Perfect for hot weather. Doesn't cling when you sweat. Love the linen feel.",
    "The lace trim is already fraying after one wear. Quality is not there for this price.",
    "True to size and very flattering. The ruching hides problem areas. Feel confident wearing it.",
    "Looks nothing like the picture. Color is completely different. Very deceiving photos.",
    "Amazing find! Sturdy construction, reinforced seams, quality zippers. Built to last.",
    "The elastic gave out after a month. Waistband is now useless. Poor quality materials.",
    "Lovely floral pattern but the fabric is stiff and uncomfortable. Needs multiple washes to soften.",
    "Perfect length for petites! Finally something that doesn't need hemming. Fits great.",
    "Pilling started immediately. Looks worn and old after just two wears. Waste of money.",
    "The mesh lining is so comfortable. Not sweaty at all even during intense workouts.",
    "Armholes are too small. Can barely lift my arms. Very restrictive and uncomfortable.",
    "Gorgeous velvet texture. Rich color that hasn't faded. Feels luxurious at an affordable price.",
    "Seams split the first time I bent over. Embarrassing and poor construction.",
    "Love that it has real pockets! Deep enough for my phone. Functional and cute.",
    "The white is see-through. Had to wear a camisole underneath. Should have been noted.",
    "Keeps its shape wash after wash. No stretching out. Quality material that lasts.",
    "Itchy sweater. Tags say cotton but feels synthetic. Can't wear without an undershirt.",
    "The cut is very modern and stylish. Gets tons of compliments every time I wear it.",
    "Ordered based on size chart, way too small. Their measurements are inaccurate.",
    "Super quick-dry material. Great for the gym. No lingering sweat smell either.",
    "The drawstring broke on first use. Cheap cord that snapped immediately.",
    "Wrinkle-resistant and travel-friendly. Pulled it out of my suitcase looking fresh.",
    "The inseam is too long even for tall sizes. Bunches up at the ankles awkwardly.",
    "Beautiful embroidered details. Hand-finished look. Worth every penny for the craftsmanship.",
    "Faded after sun exposure. Was on a beach trip and the color completely washed out.",
    "Moisture-wicking works great. Stayed dry during my entire 5-mile run. Impressed.",
    "The neckline stretches out after wearing. Becomes a boat neck unintentionally.",
    "Perfect transitional piece. Light enough for spring, works for cool summer evenings.",
    "The reflective strips are actually visible at night. Great for safety while running.",
    "Too boxy. No shape whatsoever. Makes me look wider than I am. Unflattering cut.",
    "Soft jersey material. Drapes nicely and moves with you. Very comfortable all day.",
    "The zipper keeps getting stuck. Frustrating to get on and off every time.",
    "Finally a sports bra with actual support! No more bouncing. Worth the higher price.",
    "Thread started unraveling from the logo after first wash. Cheap manufacturing.",
    "Love the vintage wash. Looks authentically worn-in. Great distressing on the denim.",
    "The ribbed texture is nice but the material is thin. Can see my bra underneath.",
    "Perfect hiking pants. Durable, water-resistant, and the pockets have zippers.",
    "Buttons pop open constantly. Annoying to keep re-buttoning throughout the day.",
    "The cashmere blend is so luxurious. Soft without being too warm. My favorite sweater.",
    "Disappointed in the length. Listed as tunic but barely covers my hips. False advertising.",
    "Great athleisure piece. Looks put-together but feels like pajamas. Best of both worlds.",
    "The pleats are uneven and messy-looking. Poor quality control on the finishing.",
    "Breathes well in the heat. Didn't feel sweaty even in 90-degree weather. Good ventilation.",
    "Collar is too stiff. Pokes my neck uncomfortably. Needs to be broken in more.",
    "Love the side slits. Adds nice movement and makes walking easier. Thoughtful design.",
    "The black faded to a weird purple after a few washes. Follow care instructions exactly.",
    "Compression fit is perfect. Supports muscles during workouts without being too tight.",
    "Shoulder seams are in the wrong place. Sits awkwardly and looks off.",
    "The modal fabric is heavenly soft. Like wearing a cloud. So comfortable for sleeping.",
    "Decorative stitching came undone. Loose threads everywhere after one wash.",
    "High-waisted and tummy control actually works! Smooths everything out nicely.",
    "The belt loops are too small. My normal belts don't fit through them. Annoying.",
    "Quality is amazing for this price point. Feels like a designer piece.",
    "Smells like chemicals. Even after multiple washes, there's still a weird odor.",
    "Perfect yoga pants. Stretchy enough for all poses. Doesn't ride up during practice.",
    "The bottom hem came unstitched. Had to sew it myself on a brand new item.",
    "Love the cowl neck. Drapes beautifully and adds elegance. Very sophisticated look.",
    "Runs really short in the torso. Shows my stomach when I lift my arms.",
    "The thermal lining is amazing. Stayed warm in freezing temperatures. Great insulation.",
    "Pocket placement is weird. Makes my hips look bigger than they are.",
    "Finally a white tee that's not see-through! Opaque material that washes well.",
    "The ruffle details are cute but poorly attached. Coming loose after light wear.",
    "Super stretchy and forgiving. Comfortable even after a big meal. Perfect everyday pants.",
    "The plaid pattern doesn't line up at the seams. Looks cheap and poorly made.",
    "Excellent range of motion. Great for CrossFit. Doesn't restrict any movements.",
]


def run_test():
    print("=" * 60)
    print("🧪 PRISM MINER - LIVE TEST (100 sample reviews)")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not set!")
        print("   Run: export GROQ_API_KEY=your_key")
        return
    
    # Initialize services
    print("\n📦 Initializing spaCy...")
    extractor = OpinionUnitExtractor()
    aggregator = OpinionAggregator()
    
    # Step 1: Process reviews
    print(f"\n📥 Processing {len(SAMPLE_REVIEWS)} sample reviews...")
    
    sample_extractions = []
    
    for i, text in enumerate(SAMPLE_REVIEWS):
        units = extractor.extract(text)
        if units:
            aggregator.add_units("Clothing", units, raw_review=text)
        
        # Save some samples for display
        if i < 5:
            sample_extractions.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "units": units
            })
    
    # Step 2: Show extraction results
    stats = aggregator.get_stats()
    print(f"\n✅ Processed {len(SAMPLE_REVIEWS)} reviews")
    print(f"   • Unique opinion patterns: {stats['unique_units']}")
    print(f"   • Total occurrences: {stats['total_occurrences']}")
    
    print("\n📝 Sample Extractions:")
    for i, sample in enumerate(sample_extractions, 1):
        print(f"\n   Review {i}: \"{sample['text']}\"")
        print(f"   → Extracted: {sample['units']}")
    
    # Step 3: Get aggregated data for LLM
    top_units = aggregator.get_top_units("Clothing", limit=50)
    print(f"\n📊 Top 15 Opinion Patterns:")
    for item in top_units[:15]:
        print(f"   {item['rank']:>3}. {item['unit']:<35} (count: {item['count']}, {item['percentage']:.1f}%)")
    
    # Step 4: Call Groq LIVE API
    print(f"\n🤖 Calling Groq LIVE API ({GROQ_MODEL})...")
    
    # Build prompt
    aggregated_data = aggregator.get_aggregated_prompt_data("Clothing", limit=50)
    
    prompt = f"""You are analyzing aggregated customer feedback patterns for the "Clothing" product category.

Below is data from analyzing {len(SAMPLE_REVIEWS)} real customer reviews. The "Count" shows how many times each opinion pattern appeared.

{aggregated_data}

Based on this analysis, identify the "Hidden Dimensions" - qualitative product attributes that customers frequently discuss but are NOT in standard product specs.

For each dimension, provide:
1. name: A clear attribute name (e.g., "Size Accuracy", "Fabric Softness")
2. importance: "High" (frequently mentioned), "Medium", or "Low"
3. description: What this dimension measures and why customers care
4. positive_vocabulary: Words/phrases customers use when satisfied
5. negative_vocabulary: Words/phrases customers use when dissatisfied
6. recommendation: How a retailer should use this (filter, warning, badge, etc.)

Focus on the 8-10 most impactful dimensions that would help shoppers make better decisions.

Respond with a JSON object containing a "dimensions" array."""

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert e-commerce product analyst specializing in clothing. Respond only with valid JSON."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=2500,
            temperature=0.3,
        )
        
        result = response.choices[0].message.content
        dimensions = json.loads(result)
        
        # Step 5: Display results
        print("\n" + "=" * 60)
        print("🎯 MINED HIDDEN DIMENSIONS")
        print("=" * 60)
        
        for dim in dimensions.get("dimensions", []):
            print(f"\n📌 {dim.get('name', 'Unknown')}")
            print(f"   Importance: {dim.get('importance', 'N/A')}")
            print(f"   Description: {dim.get('description', 'N/A')[:80]}...")
            if dim.get('positive_vocabulary'):
                pos = dim['positive_vocabulary']
                if isinstance(pos, list):
                    print(f"   ✓ Positive: {', '.join(pos[:5])}")
                else:
                    print(f"   ✓ Positive: {pos}")
            if dim.get('negative_vocabulary'):
                neg = dim['negative_vocabulary']
                if isinstance(neg, list):
                    print(f"   ✗ Negative: {', '.join(neg[:5])}")
                else:
                    print(f"   ✗ Negative: {neg}")
            if dim.get('recommendation'):
                print(f"   💡 Recommendation: {dim['recommendation'][:60]}...")
        
        # Save results
        output_path = Path(__file__).parent.parent / "output" / "test_dimensions.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                "reviews_processed": len(SAMPLE_REVIEWS),
                "unique_patterns": stats['unique_units'],
                "top_patterns": top_units[:20],
                "dimensions": dimensions
            }, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
        
        # Token usage
        usage = response.usage
        cost_estimate = (usage.prompt_tokens * 0.59 + usage.completion_tokens * 0.79) / 1_000_000
        print(f"\n📈 Token Usage: {usage.prompt_tokens:,} prompt + {usage.completion_tokens:,} completion = {usage.total_tokens:,} total")
        print(f"   Estimated cost: ${cost_estimate:.4f}")
        
    except Exception as e:
        print(f"\n❌ Groq API Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_test()
