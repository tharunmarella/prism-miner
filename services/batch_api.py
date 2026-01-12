"""
Groq Batch API Service

Handles submission and retrieval of batch jobs for review mining.
"""

import os
import json
import time
import structlog
from typing import List, Dict, Any, Optional
from openai import OpenAI

log = structlog.get_logger()


MINING_PROMPT = """You are analyzing customer reviews for the "{category}" product category.

Your task is to identify "Hidden Dimensions" - qualitative product attributes that customers frequently discuss but are NOT typically found in product specifications.

For each dimension you discover, provide:
1. name: A clear, title-cased name (e.g., "Size Accuracy", "Fabric Breathability")
2. importance: "High", "Medium", or "Low" based on how often it appears
3. description: A brief explanation of what this dimension measures
4. example_vocabulary: 5-10 words/phrases customers use to describe this dimension

Here are the customer opinion units to analyze:

{opinion_units}

Respond with a JSON object containing a "dimensions" array. Focus on the top 10-15 most important dimensions."""


class GroqBatchService:
    """
    Handles Groq Batch API operations for large-scale review mining.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    def create_batch_file(
        self,
        category_units: Dict[str, List[str]],
        output_path: str,
    ) -> str:
        """
        Creates a JSONL batch file for Groq API.
        
        Args:
            category_units: Dict mapping category name to list of opinion units
            output_path: Path to write the JSONL file
        
        Returns:
            Path to the created file
        """
        with open(output_path, 'w') as f:
            for category, units in category_units.items():
                # Create the request payload
                request = {
                    "custom_id": f"cat_{category.replace(' ', '_')}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert e-commerce product analyst. Respond only with valid JSON."
                            },
                            {
                                "role": "user",
                                "content": MINING_PROMPT.format(
                                    category=category,
                                    opinion_units="\n".join(units[:500])  # Limit to avoid token overflow
                                )
                            }
                        ],
                        "response_format": {"type": "json_object"},
                        "max_tokens": 2000,
                    }
                }
                f.write(json.dumps(request) + "\n")
        
        log.info("Created batch file", path=output_path, categories=len(category_units))
        return output_path

    def upload_batch_file(self, file_path: str) -> str:
        """
        Uploads a batch file to Groq and returns the file ID.
        """
        with open(file_path, 'rb') as f:
            response = self.client.files.create(file=f, purpose="batch")
        
        file_id = response.id
        log.info("Uploaded batch file", file_id=file_id)
        return file_id

    def create_batch_job(
        self,
        file_id: str,
        completion_window: str = "24h",
    ) -> str:
        """
        Creates a batch job and returns the batch ID.
        """
        response = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
        )
        
        batch_id = response.id
        log.info("Created batch job", batch_id=batch_id, window=completion_window)
        return batch_id

    def check_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Checks the status of a batch job.
        """
        response = self.client.batches.retrieve(batch_id)
        return {
            "id": response.id,
            "status": response.status,
            "request_counts": response.request_counts,
            "output_file_id": response.output_file_id,
            "error_file_id": response.error_file_id,
        }

    def wait_for_completion(
        self,
        batch_id: str,
        poll_interval: int = 60,
        timeout: int = 86400,  # 24 hours
    ) -> Dict[str, Any]:
        """
        Polls until batch job is complete.
        """
        start_time = time.time()
        
        while True:
            status = self.check_batch_status(batch_id)
            log.info("Batch status", **status)
            
            if status["status"] == "completed":
                return status
            elif status["status"] in ["failed", "expired", "cancelled"]:
                raise Exception(f"Batch job failed with status: {status['status']}")
            
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise Exception(f"Batch job timed out after {timeout} seconds")
            
            time.sleep(poll_interval)

    def download_results(self, output_file_id: str, output_path: str) -> str:
        """
        Downloads batch results to a local file.
        """
        response = self.client.files.content(output_file_id)
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        log.info("Downloaded batch results", path=output_path)
        return output_path

    def parse_results(self, results_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Parses the batch results JSONL file.
        
        Returns:
            Dict mapping category name to extracted dimensions
        """
        results = {}
        
        with open(results_path, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    custom_id = result.get("custom_id", "")
                    category = custom_id.replace("cat_", "").replace("_", " ")
                    
                    if result.get("error"):
                        log.warning("Batch request failed", category=category, error=result["error"])
                        continue
                    
                    response_body = result.get("response", {}).get("body", {})
                    choices = response_body.get("choices", [])
                    
                    if choices:
                        content = choices[0].get("message", {}).get("content", "{}")
                        dimensions = json.loads(content)
                        results[category] = dimensions
                        
                except (json.JSONDecodeError, KeyError) as e:
                    log.warning("Failed to parse result", error=str(e))
                    continue
        
        log.info("Parsed batch results", categories=len(results))
        return results
