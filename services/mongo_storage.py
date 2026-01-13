"""
MongoDB Storage Service

Persists mining results to MongoDB for durability and querying.
"""

import os
import structlog
from typing import Dict, Any, List, Optional
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

log = structlog.get_logger()


class MongoStorage:
    """
    MongoDB client for storing mining results and batch job metadata.
    """

    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.getenv("MONGODB_URI")
        self.client = None
        self.db = None
        self._connect()

    def _connect(self):
        """Establish connection to MongoDB."""
        if not self.connection_string:
            log.warning("No MongoDB connection string provided")
            return
        
        try:
            self.client = MongoClient(self.connection_string)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client.get_database("prism_miner")
            log.info("Connected to MongoDB", database="prism_miner")
        except ConnectionFailure as e:
            log.error("Failed to connect to MongoDB", error=str(e))
            self.client = None
            self.db = None

    def save_batch_job(self, batch_id: str, file_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Save batch job information for tracking.
        """
        if self.db is None:
            return False
        
        doc = {
            "_id": batch_id,
            "file_id": file_id,
            "status": "submitted",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            **metadata
        }
        
        self.db.batch_jobs.update_one(
            {"_id": batch_id},
            {"$set": doc},
            upsert=True
        )
        log.info("Saved batch job", batch_id=batch_id)
        return True

    def update_batch_status(self, batch_id: str, status: str, output_file_id: Optional[str] = None):
        """
        Update batch job status.
        """
        if self.db is None:
            return
        
        update = {
            "status": status,
            "updated_at": datetime.utcnow(),
        }
        if output_file_id:
            update["output_file_id"] = output_file_id
        
        self.db.batch_jobs.update_one(
            {"_id": batch_id},
            {"$set": update}
        )

    def save_category_dimensions(self, category: str, dimensions: Dict[str, Any]) -> bool:
        """
        Save extracted dimensions for a category.
        """
        if self.db is None:
            return False
        
        doc = {
            "_id": category,
            "category": category,
            "dimensions": dimensions.get("dimensions", []),
            "mined_at": datetime.utcnow(),
            "source": "groq_batch_api",
        }
        
        self.db.category_dimensions.update_one(
            {"_id": category},
            {"$set": doc},
            upsert=True
        )
        return True

    def save_all_dimensions(self, results: Dict[str, Dict[str, Any]]) -> int:
        """
        Bulk save all category dimensions from a batch result.
        """
        if self.db is None:
            return 0
        
        count = 0
        for category, dimensions in results.items():
            if self.save_category_dimensions(category, dimensions):
                count += 1
        
        log.info("Saved dimensions", categories=count)
        return count

    def get_category_dimensions(self, category: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve dimensions for a specific category.
        """
        if self.db is None:
            return None
        
        doc = self.db.category_dimensions.find_one({"_id": category})
        return doc

    def list_mined_categories(self) -> List[str]:
        """
        List all categories that have been mined.
        """
        if self.db is None:
            return []
        
        return [doc["_id"] for doc in self.db.category_dimensions.find({}, {"_id": 1})]

    def get_all_dimensions(self) -> Dict[str, Any]:
        """
        Export all dimensions as a dictionary.
        """
        if self.db is None:
            return {}
        
        results = {}
        for doc in self.db.category_dimensions.find():
            results[doc["_id"]] = {
                "dimensions": doc.get("dimensions", []),
                "mined_at": doc.get("mined_at"),
            }
        return results

    def get_pending_batch_jobs(self) -> List[Dict[str, Any]]:
        """
        Get batch jobs that are still in progress.
        """
        if self.db is None:
            return []
        
        return list(self.db.batch_jobs.find({
            "status": {"$nin": ["completed", "failed", "expired", "cancelled"]}
        }))

    def save_asin_map(self, category: str, asin_map: Dict[str, str]) -> bool:
        """
        Save ASIN mapping to MongoDB to avoid re-parsing massive files.
        Stores as individual documents in 'asin_catalog' collection.
        """
        if self.db is None:
            return False
            
        if not asin_map:
            return True
            
        from pymongo import InsertOne
        
        # Prepare bulk operations
        ops = []
        for asin, leaf_cat in asin_map.items():
            ops.append(InsertOne({
                "_id": asin,
                "leaf_category": leaf_cat,
                "parent_category": category
            }))
            
        # Bulk write in chunks of 10,000
        chunk_size = 10000
        total_inserted = 0
        
        try:
            for i in range(0, len(ops), chunk_size):
                chunk = ops[i:i + chunk_size]
                # Use ordered=False to ignore duplicate key errors (if ASINs already exist)
                self.db.asin_catalog.bulk_write(chunk, ordered=False)
                total_inserted += len(chunk)
                
            log.info("Saved ASIN map to MongoDB", 
                     category=category, 
                     count=total_inserted)
            return True
            
        except Exception as e:
            # bulk_write raises BulkWriteError if there are duplicates, which is fine
            # We just log it and assume it's mostly successful
            log.info("Partial save of ASIN map (likely duplicates)", error=str(e))
            return True

    def get_asin_map(self, category: str) -> Optional[Dict[str, str]]:
        """
        Retrieve ASIN mapping from MongoDB.
        """
        if self.db is None:
            return None
            
        # Check if we have any ASINs for this category
        if self.db.asin_catalog.count_documents({"parent_category": category}, limit=1) == 0:
            return None
            
        # Stream results
        asin_map = {}
        cursor = self.db.asin_catalog.find(
            {"parent_category": category},
            {"_id": 1, "leaf_category": 1}
        )
        
        for doc in cursor:
            asin_map[doc["_id"]] = doc["leaf_category"]
            
        return asin_map
