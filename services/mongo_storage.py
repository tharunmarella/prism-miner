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
        if not self.db:
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
        if not self.db:
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
        if not self.db:
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
        if not self.db:
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
        if not self.db:
            return None
        
        doc = self.db.category_dimensions.find_one({"_id": category})
        return doc

    def list_mined_categories(self) -> List[str]:
        """
        List all categories that have been mined.
        """
        if not self.db:
            return []
        
        return [doc["_id"] for doc in self.db.category_dimensions.find({}, {"_id": 1})]

    def get_all_dimensions(self) -> Dict[str, Any]:
        """
        Export all dimensions as a dictionary.
        """
        if not self.db:
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
        if not self.db:
            return []
        
        return list(self.db.batch_jobs.find({
            "status": {"$nin": ["completed", "failed", "expired", "cancelled"]}
        }))
