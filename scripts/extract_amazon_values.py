import json
import os
import glob
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import ast

# Configuration
INPUT_DIR = "/Users/tharun/Documents/projects/Prism/datasets/amazon_2023/raw/meta_categories"
OUTPUT_DIR = "/Users/tharun/Documents/projects/Prism/datasets/amazon_taxonomy_enriched"
TOP_N_VALUES = 15  # Keep top 15 values per key
MIN_KEY_OCCURRENCE = 50 # Ignore keys that appear less than 50 times (noise reduction)

def parse_details(details_str):
    """Safely parse the details string which can be malformed."""
    if not details_str:
        return {}
    try:
        # Most are simple JSON
        return json.loads(details_str)
    except json.JSONDecodeError:
        try:
            # Some might be python-dict style strings
            return ast.literal_eval(details_str)
        except:
            return {}

def process_file(filepath):
    """Process a single JSONL file and extract keys + values."""
    filename = os.path.basename(filepath)
    category_name = filename.replace("meta_", "").replace(".jsonl", "")
    print(f"Starting {category_name}...")
    
    key_counts = Counter()
    key_values = defaultdict(Counter)
    total_scanned = 0
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    total_scanned += 1
                    
                    # Extract from 'details'
                    details = data.get('details')
                    if details:
                        # details can be a dict or a stringified dict
                        if isinstance(details, str):
                            details_dict = parse_details(details)
                        elif isinstance(details, dict):
                            details_dict = details
                        else:
                            details_dict = {}
                            
                        for k, v in details_dict.items():
                            k_clean = k.strip()
                            v_clean = str(v).strip()
                            
                            if k_clean and v_clean:
                                key_counts[k_clean] += 1
                                key_values[k_clean][v_clean] += 1
                                
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

    # Format Output
    enriched_attributes = []
    
    # Sort keys by occurrence (most common first)
    sorted_keys = key_counts.most_common()
    
    for key, count in sorted_keys:
        if count < MIN_KEY_OCCURRENCE:
            continue
            
        # Get top N values for this key
        top_values_tuples = key_values[key].most_common(TOP_N_VALUES)
        examples = [val for val, _ in top_values_tuples]
        
        enriched_attributes.append({
            "name": key,
            "occurrence": count,
            "frequency_pct": round((count / total_scanned) * 100, 2) if total_scanned > 0 else 0,
            "examples": examples
        })
    
    result = {
        "category": category_name,
        "total_items_scanned": total_scanned,
        "total_unique_keys": len(enriched_attributes),
        "attributes": enriched_attributes
    }
    
    # Write to file immediately
    output_path = os.path.join(OUTPUT_DIR, f"{category_name}_enriched.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
        
    print(f"✅ Finished {category_name}: Found {len(enriched_attributes)} attributes")
    return category_name

def main():
    files = glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))
    if not files:
        print(f"No files found in {INPUT_DIR}")
        return

    print(f"Found {len(files)} files to process.")
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_file, files))
        
    print("Done!")

if __name__ == "__main__":
    main()
