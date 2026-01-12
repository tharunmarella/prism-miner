"""
Opinion Unit Aggregator

Aggregates opinion units across all reviews, counting frequencies.
This allows us to process 66M reviews but send only the most important patterns to the LLM.

NEW: Semantic Aggregation using embeddings to group similar patterns:
  - "runs small", "too tight", "size up" → merged into one cluster
"""

import structlog
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional

log = structlog.get_logger()

# Lazy load embedding model (only when needed)
_embedding_model = None

def get_embedding_model():
    """Lazy load the embedding model to avoid slow imports."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model for semantic clustering...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        log.info("Embedding model loaded")
    return _embedding_model


class OpinionAggregator:
    """
    Aggregates opinion units by category, tracking frequency and examples.
    
    Instead of sending raw units like:
        ["fabric: soft", "fabric: soft", "fabric: soft", ...]
    
    We send aggregated data:
        {"fabric: soft": {"count": 15000, "rank": 1}, ...}
    
    This gives the LLM much richer signal about what actually matters.
    """

    def __init__(self):
        # {category: Counter of opinion units}
        self.category_units: Dict[str, Counter] = defaultdict(Counter)
        # {category: total reviews processed}
        self.category_review_counts: Dict[str, int] = defaultdict(int)
        # {category: sample raw reviews for context}
        self.category_samples: Dict[str, List[str]] = defaultdict(list)

    def add_units(self, category: str, units: List[str], raw_review: str = None):
        """
        Add opinion units for a category.
        """
        for unit in units:
            # Normalize: lowercase, strip whitespace
            normalized = unit.lower().strip()
            if normalized and len(normalized) > 3:  # Skip very short units
                self.category_units[category][normalized] += 1
        
        self.category_review_counts[category] += 1
        
        # Keep some sample reviews for context (max 20 per category)
        if raw_review and len(self.category_samples[category]) < 20:
            if len(raw_review) > 50:  # Only meaningful reviews
                self.category_samples[category].append(raw_review[:500])

    def get_top_units(self, category: str, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Get the top N most frequent opinion units for a category.
        """
        counter = self.category_units.get(category, Counter())
        total_reviews = self.category_review_counts.get(category, 1)
        
        top_units = []
        for rank, (unit, count) in enumerate(counter.most_common(limit), 1):
            pct = (count / total_reviews) * 100
            top_units.append({
                "unit": unit,
                "count": count,
                "percentage": round(pct, 1),
                "rank": rank,
            })
        
        return top_units

    def get_aggregated_prompt_data(self, category: str, limit: int = 200) -> str:
        """
        Format aggregated data for LLM prompt.
        """
        top_units = self.get_top_units(category, limit)
        total_reviews = self.category_review_counts.get(category, 0)
        total_unique_units = len(self.category_units.get(category, {}))
        
        lines = [
            f"Category: {category}",
            f"Total Reviews Analyzed: {total_reviews:,}",
            f"Unique Opinion Patterns Found: {total_unique_units:,}",
            f"",
            f"Top {len(top_units)} Most Frequent Opinion Patterns:",
            f"{'Rank':<6} {'Count':<10} {'%':<8} Opinion Unit",
            "-" * 60,
        ]
        
        for item in top_units:
            lines.append(
                f"{item['rank']:<6} {item['count']:<10,} {item['percentage']:<8.1f}% {item['unit']}"
            )
        
        # Add sample reviews for context
        samples = self.category_samples.get(category, [])
        if samples:
            lines.append("")
            lines.append("Sample Reviews for Context:")
            for i, sample in enumerate(samples[:5], 1):
                lines.append(f"  {i}. \"{sample}...\"")
        
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get overall aggregation statistics.
        """
        total_reviews = sum(self.category_review_counts.values())
        total_units = sum(len(c) for c in self.category_units.values())
        total_occurrences = sum(sum(c.values()) for c in self.category_units.values())
        
        return {
            "categories": len(self.category_units),
            "total_reviews": total_reviews,
            "unique_units": total_units,
            "total_occurrences": total_occurrences,
        }

    def get_all_categories(self) -> List[str]:
        """
        Get list of all categories with data.
        """
        return list(self.category_units.keys())

    def get_semantically_grouped_units(
        self, 
        category: str, 
        similarity_threshold: float = 0.75,
        min_cluster_count: int = 2,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Groups semantically similar opinion units together before sending to LLM.
        
        Example:
            Before: ["runs small": 1000, "too tight": 500, "size up": 300]
            After:  ["sizing: runs small": 1800, variations: ["too tight", "size up"]]
        
        Args:
            category: The product category
            similarity_threshold: 0.0-1.0, higher = stricter grouping (0.75 recommended)
            min_cluster_count: Minimum count for a cluster to be included
            limit: Max clusters to return
            
        Returns:
            List of clustered opinion units with combined counts
        """
        from sklearn.cluster import AgglomerativeClustering
        
        counter = self.category_units.get(category, Counter())
        if not counter:
            return []
        
        # Get all unique units
        unique_units = list(counter.keys())
        counts = [counter[u] for u in unique_units]
        
        # If too few units, skip clustering
        if len(unique_units) < 10:
            return self.get_top_units(category, limit)
        
        log.info("Starting semantic clustering", 
                 category=category, 
                 unique_units=len(unique_units))
        
        # Generate embeddings
        model = get_embedding_model()
        embeddings = model.encode(unique_units, show_progress_bar=False)
        
        # Cluster by similarity
        # distance_threshold = 1 - similarity (cosine distance)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - similarity_threshold,
            metric='cosine',
            linkage='average',
        ).fit(embeddings)
        
        # Aggregate counts by cluster
        clusters: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
        for idx, cluster_id in enumerate(clustering.labels_):
            clusters[cluster_id].append((unique_units[idx], counts[idx]))
        
        log.info("Clustering complete", 
                 original_units=len(unique_units),
                 clusters=len(clusters))
        
        # Build final clustered data
        total_reviews = self.category_review_counts.get(category, 1)
        clustered_data = []
        
        for cluster_id, members in clusters.items():
            # Sort by count (most frequent first)
            members.sort(key=lambda x: x[1], reverse=True)
            
            # Use most frequent as the representative
            representative = members[0][0]
            total_count = sum(m[1] for m in members)
            
            if total_count < min_cluster_count:
                continue
            
            pct = (total_count / total_reviews) * 100
            
            clustered_data.append({
                "unit": representative,
                "count": total_count,
                "percentage": round(pct, 1),
                "cluster_size": len(members),
                "variations": [m[0] for m in members[1:6]],  # Top 5 variations
            })
        
        # Sort by total count
        clustered_data.sort(key=lambda x: x["count"], reverse=True)
        
        # Add rank
        for i, item in enumerate(clustered_data[:limit], 1):
            item["rank"] = i
        
        return clustered_data[:limit]

    def get_semantic_prompt_data(
        self, 
        category: str, 
        limit: int = 200,
        similarity_threshold: float = 0.75,
    ) -> str:
        """
        Format semantically-grouped data for LLM prompt.
        This produces MUCH cleaner input for the LLM.
        """
        clustered = self.get_semantically_grouped_units(
            category, 
            similarity_threshold=similarity_threshold,
            limit=limit,
        )
        
        total_reviews = self.category_review_counts.get(category, 0)
        original_units = len(self.category_units.get(category, {}))
        
        lines = [
            f"Category: {category}",
            f"Total Reviews Analyzed: {total_reviews:,}",
            f"Original Unique Patterns: {original_units:,}",
            f"After Semantic Grouping: {len(clustered)} concept clusters",
            f"",
            f"Top {len(clustered)} Opinion Clusters (semantically grouped):",
            f"{'Rank':<5} {'Count':<9} {'%':<7} {'Vars':<5} Representative Pattern",
            "-" * 70,
        ]
        
        for item in clustered:
            variations_str = f"(+{item['cluster_size']-1})" if item['cluster_size'] > 1 else ""
            lines.append(
                f"{item['rank']:<5} {item['count']:<9,} {item['percentage']:<7.1f}% {variations_str:<5} {item['unit']}"
            )
            # Show variations for large clusters
            if item['variations'] and item['cluster_size'] > 2:
                vars_preview = ", ".join(item['variations'][:3])
                lines.append(f"      └─ also: {vars_preview}")
        
        # Add sample reviews for context
        samples = self.category_samples.get(category, [])
        if samples:
            lines.append("")
            lines.append("Sample Reviews for Context:")
            for i, sample in enumerate(samples[:5], 1):
                lines.append(f"  {i}. \"{sample[:200]}...\"")
        
        return "\n".join(lines)

    def compare_raw_vs_semantic(self, category: str, limit: int = 20) -> Dict[str, Any]:
        """
        Compare raw aggregation vs semantic grouping for debugging/demo.
        """
        raw = self.get_top_units(category, limit)
        semantic = self.get_semantically_grouped_units(category, limit=limit)
        
        raw_total = sum(u['count'] for u in raw)
        semantic_total = sum(u['count'] for u in semantic)
        
        return {
            "raw": {
                "count": len(raw),
                "top_patterns": [u['unit'] for u in raw[:10]],
                "total_mentions": raw_total,
            },
            "semantic": {
                "count": len(semantic),
                "top_patterns": [u['unit'] for u in semantic[:10]],
                "total_mentions": semantic_total,
                "clusters_with_variations": [
                    {"pattern": u['unit'], "merged": u['cluster_size'], "variations": u['variations']}
                    for u in semantic[:10] if u['cluster_size'] > 1
                ],
            },
        }
