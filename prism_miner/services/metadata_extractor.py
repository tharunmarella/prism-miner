"""
Metadata Extractor Service

Extracts attribute keys and example values from Amazon product metadata.
"""

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import structlog

log = structlog.get_logger()

# Attributes to exclude (logistics, not product features)
BLOCKLIST = {
    "Date First Available",
    "Package Dimensions", 
    "Product Dimensions",
    "Item Weight",
    "Manufacturer",
    "ASIN",
    "Best Sellers Rank",
    "Customer Reviews",
    "Item model number",
    "Is Discontinued By Manufacturer",
    "Country of Origin",
    "UPC",
    "EAN",
    "Batteries",
    "Domestic Shipping",
    "International Shipping",
}


class MetadataExtractor:
    """Extract attribute keys and values from Amazon product metadata."""
    
    def __init__(self, data_dir: str = "data/amazon_taxonomy"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def extract_from_jsonl(
        self,
        jsonl_path: str,
        category_name: str,
        max_examples: int = 15,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract attributes from a JSONL metadata file.
        
        Args:
            jsonl_path: Path to the metadata JSONL file
            category_name: Name of the category
            max_examples: Max example values per attribute
            limit: Optional limit on items to process
            
        Returns:
            Dict with extracted attributes and statistics
        """
        key_counter = Counter()
        key_values = defaultdict(Counter)
        total_items = 0
        
        log.info("Extracting metadata", path=jsonl_path, category=category_name)
        
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(tqdm(f, desc=f"Scanning {category_name}")):
                if limit and i >= limit:
                    break
                    
                try:
                    data = json.loads(line)
                    details = data.get('details', '{}')
                    
                    if isinstance(details, str):
                        details = json.loads(details) if details else {}
                    
                    if not details:
                        continue
                        
                    total_items += 1
                    
                    for key, value in details.items():
                        # Skip blocklisted keys
                        if key in BLOCKLIST:
                            continue
                            
                        key_counter[key] += 1
                        
                        # Track example values
                        if isinstance(value, str) and len(value) < 100:
                            key_values[key][value] += 1
                            
                except Exception as e:
                    continue
        
        # Build result
        attributes = []
        for key, count in key_counter.most_common():
            # Get top example values
            examples = [v for v, _ in key_values[key].most_common(max_examples)]
            
            attributes.append({
                "name": key,
                "occurrence": count,
                "frequency_pct": round(count / total_items * 100, 2) if total_items > 0 else 0,
                "examples": examples
            })
        
        result = {
            "category": category_name,
            "total_items_scanned": total_items,
            "total_unique_keys": len(attributes),
            "attributes": attributes
        }
        
        log.info(
            "Extraction complete",
            category=category_name,
            items=total_items,
            attributes=len(attributes)
        )
        
        return result
    
    def save_result(self, result: Dict[str, Any], filename: str):
        """Save extraction result to JSON file."""
        output_path = os.path.join(self.data_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        log.info("Saved result", path=output_path)
    
    def load_result(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load extraction result from JSON file."""
        path = os.path.join(self.data_dir, filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None
