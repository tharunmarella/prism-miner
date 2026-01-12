"""
Prism Miner API - Start and monitor the dimension mining pipeline

Endpoints:
    POST /pipeline/start     - Start the full batch pipeline
    GET  /pipeline/status    - Get current pipeline status
    GET  /pipeline/progress  - Get detailed progress
    POST /pipeline/download  - Download and parse results
    GET  /categories         - List all mined categories
    GET  /categories/{name}  - Get dimensions for a category

Run:
    uvicorn api:app --reload --port 8000
"""

import os
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import threading

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from groq import Groq

# Pipeline imports
from services.opinion_extractor import OpinionUnitExtractor
from services.aggregator import OpinionAggregator

# Config
MONGODB_URI = os.getenv("MONGODB_URI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Global state for pipeline progress
pipeline_state = {
    "status": "idle",  # idle, preparing, extracting, submitting, processing, completed, failed
    "started_at": None,
    "batch_id": None,
    "progress": {
        "phase": None,
        "current": 0,
        "total": 0,
        "message": "",
    },
    "error": None,
    "results": {
        "categories_processed": 0,
        "dimensions_mined": 0,
    }
}

# MongoDB client
mongo_client = None
db = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize connections on startup."""
    global mongo_client, db
    
    if MONGODB_URI:
        mongo_client = MongoClient(MONGODB_URI)
        db = mongo_client.get_database("prism_miner")
        print("✅ Connected to MongoDB")
    else:
        print("⚠️  MONGODB_URI not set")
    
    yield
    
    if mongo_client:
        mongo_client.close()


app = FastAPI(
    title="Prism Miner API",
    description="Mine hidden product dimensions from Amazon reviews",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Models =====

class PipelineStartRequest(BaseModel):
    max_reviews_per_category: int = 10000
    test_categories: Optional[int] = None  # Limit to N categories for testing
    

class PipelineStatus(BaseModel):
    status: str
    started_at: Optional[str]
    batch_id: Optional[str]
    progress: Dict[str, Any]
    error: Optional[str]
    results: Dict[str, Any]


class CategoryDimensions(BaseModel):
    category: str
    dimensions: List[Dict[str, Any]]
    mined_at: Optional[str]


# ===== Helper Functions =====

def get_groq_client():
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")
    return Groq(api_key=GROQ_API_KEY)


def update_progress(phase: str, current: int, total: int, message: str):
    """Update pipeline progress state."""
    pipeline_state["progress"] = {
        "phase": phase,
        "current": current,
        "total": total,
        "percentage": round(current / total * 100, 1) if total > 0 else 0,
        "message": message,
    }


# ===== Background Pipeline Task =====

def run_pipeline_background(max_reviews: int, test_count: Optional[int]):
    """Run the full pipeline in the background."""
    global pipeline_state
    
    try:
        pipeline_state["status"] = "preparing"
        pipeline_state["started_at"] = datetime.utcnow().isoformat()
        pipeline_state["error"] = None
        
        from huggingface_hub import hf_hub_download
        from collections import defaultdict
        
        REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
        REVIEW_FILENAME = "raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl"
        META_FILENAME = "raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl"
        cache_dir = "/tmp/hf_cache"
        
        # Load leaf categories
        update_progress("setup", 1, 5, "Loading leaf categories...")
        categories_path = Path(__file__).parent / "data" / "categories" / "clothing_leaf_categories.txt"
        
        leaf_categories = {}
        with open(categories_path, 'r') as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2 and parts[0].isdigit():
                    leaf_categories[parts[1]] = int(parts[0])
        
        category_names = set(leaf_categories.keys())
        
        if test_count:
            sorted_cats = sorted(leaf_categories.items(), key=lambda x: -x[1])
            category_names = set(cat for cat, _ in sorted_cats[:test_count])
        
        update_progress("setup", 2, 5, f"Processing {len(category_names)} categories...")
        
        # Build ASIN mapping (check cache first)
        cache_path = Path(cache_dir) / "asin_to_category.json"
        if cache_path.exists():
            update_progress("setup", 3, 5, "Loading cached ASIN mapping...")
            with open(cache_path, 'r') as f:
                asin_to_category = json.load(f)
        else:
            update_progress("setup", 3, 5, "Building ASIN mapping (this takes a while)...")
            
            meta_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=META_FILENAME,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
            
            asin_to_category = {}
            with open(meta_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        parent_asin = data.get("parent_asin")
                        categories = data.get("categories", [])
                        
                        for cat in reversed(categories):
                            if cat in category_names:
                                asin_to_category[parent_asin] = cat
                                break
                    except:
                        continue
            
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(asin_to_category, f)
        
        # Filter to target categories
        asin_to_category = {k: v for k, v in asin_to_category.items() if v in category_names}
        
        # Extract and aggregate
        pipeline_state["status"] = "extracting"
        update_progress("extraction", 0, 1, "Downloading reviews...")
        
        reviews_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=REVIEW_FILENAME,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        
        extractor = OpinionUnitExtractor()
        aggregators: Dict[str, OpinionAggregator] = defaultdict(OpinionAggregator)
        category_review_counts: Dict[str, int] = defaultdict(int)
        
        processed = 0
        total_estimate = min(len(asin_to_category) * 100, 10000000)  # Rough estimate
        
        with open(reviews_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    parent_asin = data.get("parent_asin")
                    text = data.get("text", "")
                    
                    if not text or len(text) < 30:
                        continue
                    
                    category = asin_to_category.get(parent_asin)
                    if not category:
                        continue
                    
                    if category_review_counts[category] >= max_reviews:
                        continue
                    
                    units = extractor.extract(text)
                    if units:
                        aggregators[category].add_units(category, units, raw_review=text)
                    
                    category_review_counts[category] += 1
                    processed += 1
                    
                    if processed % 50000 == 0:
                        update_progress("extraction", processed, total_estimate, 
                                       f"Extracted {processed:,} reviews from {len(aggregators)} categories")
                except:
                    continue
        
        update_progress("extraction", processed, processed, 
                       f"Extraction complete: {processed:,} reviews, {len(aggregators)} categories")
        
        # Create batch file
        pipeline_state["status"] = "submitting"
        update_progress("batch", 1, 3, "Creating batch file...")
        
        batch_file_path = OUTPUT_DIR / "batch_input.jsonl"
        requests = []
        
        for category, aggregator in aggregators.items():
            stats = aggregator.get_stats()
            if stats["total_reviews"] < 50:
                continue
            
            try:
                prompt_data = aggregator.get_semantic_prompt_data(category, limit=150)
            except:
                prompt_data = aggregator.get_aggregated_prompt_data(category, limit=150)
            
            prompt = f"""Analyze customer feedback for "{category}" products.

{prompt_data}

Identify 8-12 "Hidden Dimensions" - qualitative attributes customers discuss but aren't in specs.

For each: name, importance (High/Medium/Low), description, positive_vocabulary (5-8 words), 
negative_vocabulary (5-8 words), recommendation (filter/badge/warning).

Respond with JSON containing a "dimensions" array."""

            request = {
                "custom_id": f"cat_{category.replace(' ', '_').replace('/', '_').replace('&', 'and')}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are an e-commerce analyst. Respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 3000,
                    "temperature": 0.3,
                }
            }
            requests.append(request)
        
        with open(batch_file_path, 'w') as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
        
        update_progress("batch", 2, 3, f"Uploading batch file ({len(requests)} requests)...")
        
        # Submit batch
        groq_client = get_groq_client()
        
        with open(batch_file_path, 'rb') as f:
            file_response = groq_client.files.create(file=f, purpose="batch")
        
        batch_response = groq_client.batches.create(
            input_file_id=file_response.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        
        pipeline_state["batch_id"] = batch_response.id
        pipeline_state["status"] = "processing"
        
        update_progress("batch", 3, 3, f"Batch submitted: {batch_response.id}")
        
        # Save batch info
        with open(OUTPUT_DIR / "batch_job.json", 'w') as f:
            json.dump({
                "batch_id": batch_response.id,
                "file_id": file_response.id,
                "created_at": datetime.utcnow().isoformat(),
                "categories": len(requests),
            }, f, indent=2)
        
        # Store in MongoDB
        if db is not None:
            db.batch_jobs.update_one(
                {"_id": batch_response.id},
                {"$set": {
                    "_id": batch_response.id,
                    "file_id": file_response.id,
                    "status": "submitted",
                    "categories": len(requests),
                    "created_at": datetime.utcnow(),
                }},
                upsert=True,
            )
        
        pipeline_state["results"]["categories_processed"] = len(requests)
        
    except Exception as e:
        pipeline_state["status"] = "failed"
        pipeline_state["error"] = str(e)
        import traceback
        traceback.print_exc()


# ===== API Endpoints =====

@app.get("/")
async def root():
    """API info."""
    return {
        "name": "Prism Miner API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /pipeline/start": "Start the mining pipeline",
            "GET /pipeline/status": "Get pipeline status",
            "GET /pipeline/progress": "Get detailed progress",
            "POST /pipeline/download": "Download batch results",
            "GET /categories": "List mined categories",
            "GET /categories/{name}": "Get category dimensions",
        }
    }


@app.post("/pipeline/start")
async def start_pipeline(
    request: PipelineStartRequest,
    background_tasks: BackgroundTasks,
):
    """Start the full batch pipeline."""
    global pipeline_state
    
    if pipeline_state["status"] in ["preparing", "extracting", "submitting"]:
        raise HTTPException(status_code=400, detail="Pipeline already running")
    
    # Reset state
    pipeline_state = {
        "status": "starting",
        "started_at": datetime.utcnow().isoformat(),
        "batch_id": None,
        "progress": {"phase": "initializing", "current": 0, "total": 0, "message": "Starting..."},
        "error": None,
        "results": {"categories_processed": 0, "dimensions_mined": 0},
    }
    
    # Run in background
    background_tasks.add_task(
        run_pipeline_background,
        request.max_reviews_per_category,
        request.test_categories,
    )
    
    return {
        "message": "Pipeline started",
        "status": "starting",
        "config": {
            "max_reviews_per_category": request.max_reviews_per_category,
            "test_categories": request.test_categories,
        }
    }


@app.get("/pipeline/status", response_model=PipelineStatus)
async def get_pipeline_status():
    """Get current pipeline status."""
    # If processing, check batch status
    if pipeline_state["status"] == "processing" and pipeline_state["batch_id"]:
        try:
            groq_client = get_groq_client()
            batch = groq_client.batches.retrieve(pipeline_state["batch_id"])
            
            if batch.request_counts:
                pipeline_state["progress"] = {
                    "phase": "groq_processing",
                    "current": batch.request_counts.completed,
                    "total": batch.request_counts.total,
                    "percentage": round(batch.request_counts.completed / batch.request_counts.total * 100, 1) if batch.request_counts.total > 0 else 0,
                    "message": f"Groq processing: {batch.request_counts.completed}/{batch.request_counts.total} completed",
                    "failed": batch.request_counts.failed,
                }
            
            if batch.status == "completed":
                pipeline_state["status"] = "completed"
                pipeline_state["progress"]["message"] = "Batch completed! Run /pipeline/download"
            elif batch.status in ["failed", "expired", "cancelled"]:
                pipeline_state["status"] = "failed"
                pipeline_state["error"] = f"Batch {batch.status}"
                
        except Exception as e:
            pass  # Keep existing state
    
    return PipelineStatus(**pipeline_state)


@app.get("/pipeline/progress")
async def get_pipeline_progress():
    """Get detailed pipeline progress."""
    return {
        **pipeline_state,
        "uptime": None,
    }


class DownloadRequest(BaseModel):
    batch_id: Optional[str] = None  # Override batch_id if provided


@app.post("/pipeline/download")
async def download_results(request: DownloadRequest = None):
    """Download and parse batch results."""
    import requests as http_requests
    
    # Use provided batch_id or fall back to pipeline state
    batch_id = None
    if request and request.batch_id:
        batch_id = request.batch_id
    elif pipeline_state["batch_id"]:
        batch_id = pipeline_state["batch_id"]
    
    if not batch_id:
        raise HTTPException(status_code=400, detail="No batch job found. Provide batch_id in request body.")
    
    groq_client = get_groq_client()
    batch = groq_client.batches.retrieve(batch_id)
    
    if batch.status != "completed":
        raise HTTPException(status_code=400, detail=f"Batch not complete: {batch.status}")
    
    if not batch.output_file_id:
        raise HTTPException(status_code=400, detail="No output file available")
    
    # Download results using REST API (SDK doesn't have files.content)
    url = f"https://api.groq.com/openai/v1/files/{batch.output_file_id}/content"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    response = http_requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to download results: {response.text}")
    
    results_path = OUTPUT_DIR / "batch_results.jsonl"
    with open(results_path, 'wb') as f:
        f.write(response.content)
    
    # Parse and save to MongoDB
    saved = 0
    errors = 0
    
    with open(results_path, 'r') as f:
        for line in f:
            try:
                result = json.loads(line)
                custom_id = result.get("custom_id", "")
                category = custom_id.replace("cat_", "").replace("_", " ")
                
                if result.get("error"):
                    errors += 1
                    continue
                
                response_body = result.get("response", {}).get("body", {})
                choices = response_body.get("choices", [])
                
                if choices:
                    content = choices[0].get("message", {}).get("content", "{}")
                    dimensions_data = json.loads(content)
                    
                    if db is not None:
                        db.category_dimensions.update_one(
                            {"_id": category},
                            {"$set": {
                                "_id": category,
                                "category": category,
                                "dimensions": dimensions_data.get("dimensions", []),
                                "mined_at": datetime.utcnow(),
                                "model": GROQ_MODEL,
                            }},
                            upsert=True,
                        )
                    saved += 1
                    
            except:
                errors += 1
                continue
    
    pipeline_state["results"]["dimensions_mined"] = saved
    
    return {
        "message": "Results downloaded and parsed",
        "saved": saved,
        "errors": errors,
    }


@app.get("/categories")
async def list_categories():
    """List all mined categories."""
    if db is None:
        raise HTTPException(status_code=500, detail="MongoDB not connected")
    
    categories = []
    for doc in db.category_dimensions.find({}, {"_id": 1, "mined_at": 1}):
        categories.append({
            "category": doc["_id"],
            "mined_at": doc.get("mined_at", "").isoformat() if doc.get("mined_at") else None,
        })
    
    return {
        "total": len(categories),
        "categories": categories,
    }


@app.get("/categories/{category_name}")
async def get_category_dimensions(category_name: str):
    """Get dimensions for a specific category."""
    if db is None:
        raise HTTPException(status_code=500, detail="MongoDB not connected")
    
    # Try exact match first, then with underscores replaced
    doc = db.category_dimensions.find_one({"_id": category_name})
    if not doc:
        doc = db.category_dimensions.find_one({"_id": category_name.replace("_", " ")})
    
    if not doc:
        raise HTTPException(status_code=404, detail=f"Category '{category_name}' not found")
    
    return CategoryDimensions(
        category=doc["_id"],
        dimensions=doc.get("dimensions", []),
        mined_at=doc.get("mined_at", "").isoformat() if doc.get("mined_at") else None,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mongodb": "connected" if db is not None else "disconnected",
        "groq_key": "set" if GROQ_API_KEY else "not set",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
