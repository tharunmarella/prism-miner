"""
Review Miner Service

Mines customer reviews to discover "hidden dimensions" - 
attributes customers care about that aren't in official metadata.
"""

import json
import os
from typing import Dict, List, Any, Optional
from openai import OpenAI
import structlog

log = structlog.get_logger()

REVIEW_MINING_PROMPT = """Analyze these customer reviews for "{category}" products.

Identify "hidden dimensions" - product attributes that customers frequently mention 
in reviews but may not be in official product specifications.

Focus on:
1. Quality aspects customers praise or complain about
2. Fit/sizing characteristics
3. Durability and longevity
4. Comfort factors
5. Value perception
6. Use-case specific features

Reviews:
{reviews}

Return a JSON object with this structure:
{{
  "hidden_dimensions": [
    {{
      "name": "Dimension name (e.g., 'Arch Support')",
      "importance": "High/Medium/Low",
      "description": "What this dimension means for shoppers",
      "example_vocabulary": ["word1", "word2", "phrase1", ...]
    }}
  ]
}}

Extract 5-10 dimensions. Focus on actionable, filter-worthy attributes."""


class ReviewMiner:
    """Mine customer reviews for hidden product dimensions."""
    
    def __init__(
        self,
        data_dir: str = "data/amazon_reviews_insights",
        model: str = "openai/gpt-oss-120b"
    ):
        self.data_dir = data_dir
        self.model = model
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize Groq client
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY")
        )
    
    def mine_reviews(
        self,
        category: str,
        reviews: List[str],
        max_reviews: int = 50
    ) -> Dict[str, Any]:
        """
        Analyze reviews to extract hidden dimensions.
        
        Args:
            category: Product category name
            reviews: List of review texts
            max_reviews: Max reviews to include in prompt
            
        Returns:
            Dict with hidden dimensions
        """
        # Truncate reviews to fit context
        review_sample = reviews[:max_reviews]
        reviews_text = "\n\n---\n\n".join([
            f"Review {i+1}: {r[:500]}" 
            for i, r in enumerate(review_sample)
        ])
        
        prompt = REVIEW_MINING_PROMPT.format(
            category=category,
            reviews=reviews_text
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an e-commerce analyst extracting product insights from customer reviews."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON from response
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            result["category"] = category
            result["reviews_analyzed"] = len(review_sample)
            
            log.info(
                "Review mining complete",
                category=category,
                dimensions=len(result.get("hidden_dimensions", []))
            )
            
            return result
            
        except Exception as e:
            log.error("Review mining failed", category=category, error=str(e))
            return {
                "category": category,
                "hidden_dimensions": [],
                "error": str(e)
            }
    
    def save_result(self, result: Dict[str, Any], filename: str):
        """Save mining result to JSON file."""
        output_path = os.path.join(self.data_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        log.info("Saved result", path=output_path)
    
    def load_result(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load mining result from JSON file."""
        path = os.path.join(self.data_dir, filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None
