import json
import os
import asyncio
import google.generativeai as genai
from huggingface_hub import hf_hub_download
from collections import defaultdict

# Configuration
REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
# Switched to Clothing
META_FILENAME = "raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl"
REVIEW_FILENAME = "raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl"
OUTPUT_DIR = "datasets/amazon_reviews_insights"

# Sample limits
MAX_PRODUCTS_PER_CATEGORY = 5
MAX_REVIEWS_PER_PRODUCT = 20
TARGET_CATEGORIES = ["Dresses", "Leggings", "Coats", "Running Shoes"] 

# Setup Gemini
if os.environ.get("GEMINI_API_KEY"):
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def get_target_products():
    """Download and stream metadata to find products matching our categories."""
    print("Downloading/Streaming metadata to find target products...")
    # This might take a while for Clothing (3GB)
    file_path = hf_hub_download(repo_id=REPO_ID, filename=META_FILENAME, repo_type="dataset")
    print(f"Metadata file: {file_path}")
    
    asin_map = {}
    category_counts = defaultdict(int)
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                title = data.get('title', '').lower()
                parent_asin = data.get('parent_asin')
                
                found_cat = None
                for cat in TARGET_CATEGORIES:
                    if cat.lower() in title:
                        found_cat = cat
                        break
                
                if found_cat and category_counts[found_cat] < MAX_PRODUCTS_PER_CATEGORY:
                    asin_map[parent_asin] = {
                        "title": data.get('title'),
                        "category": found_cat
                    }
                    category_counts[found_cat] += 1
                    
                if all(category_counts[c] >= MAX_PRODUCTS_PER_CATEGORY for c in TARGET_CATEGORIES):
                    break
                    
            except Exception:
                continue
            
    print(f"Found {len(asin_map)} target products.")
    return asin_map

def get_reviews(asin_map):
    """Download and stream reviews for the found products."""
    print("Downloading/Streaming reviews (This is HUGE)...")
    # This is 15GB+ !!
    file_path = hf_hub_download(repo_id=REPO_ID, filename=REVIEW_FILENAME, repo_type="dataset")
    print(f"Reviews file: {file_path}")
    
    reviews_by_asin = defaultdict(list)
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                parent_asin = data.get('parent_asin')
                
                if parent_asin in asin_map:
                    if len(reviews_by_asin[parent_asin]) < MAX_REVIEWS_PER_PRODUCT:
                        reviews_by_asin[parent_asin].append({
                            "text": data.get("text", ""),
                            "rating": data.get("rating")
                        })
            except Exception:
                continue
    
    print(f"Collected reviews for {len(reviews_by_asin)} products.")
    return reviews_by_asin

async def analyze_category_insights(category, reviews_text):
    print(f"Analyzing insights for {category}...")
    
    prompt = f"""
    You are an Expert Product Researcher. Analyze these User Reviews for **{category}**.
    
    Goal: Identify "Hidden Dimensions" - specific attributes users care about that are often missing from standard specs.
    
    Output JSON:
    {{
      "category": "{category}",
      "hidden_dimensions": [
        {{
          "name": "Dimension Name",
          "importance": "High/Medium",
          "description": "What users are looking for",
          "example_vocabulary": ["words", "users", "use"]
        }}
      ]
    }}
    
    Reviews:
    {reviews_text[:20000]} 
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return response.text
    except Exception as e:
        print(f"LLM failed for {category}: {e}")
        return None

async def main():
    # 1. Identify Targets
    asin_map = get_target_products()
    if not asin_map:
        print("No targets found.")
        return

    # 2. Get Reviews
    reviews_by_asin = get_reviews(asin_map)

    # 3. Group & Analyze
    category_reviews = defaultdict(list)
    for asin, reviews in reviews_by_asin.items():
        cat = asin_map[asin]['category']
        texts = [f"Rating {r['rating']}: {r['text']}" for r in reviews]
        category_reviews[cat].extend(texts)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for category, texts in category_reviews.items():
        if not texts:
            print(f"No reviews for {category}")
            continue
            
        full_text = "\n---\n".join(texts)
        json_result = await analyze_category_insights(category, full_text)
        
        if json_result:
            filename = f"{OUTPUT_DIR}/{category}_insights.json"
            with open(filename, "w") as f:
                f.write(json_result)
            print(f"✅ Saved {category}")

if __name__ == "__main__":
    asyncio.run(main())
