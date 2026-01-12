import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Groq API
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    
    # HuggingFace
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    # MongoDB
    MONGODB_URI = os.getenv("MONGODB_URI")
    
    # Processing
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))  # Reviews per LLM call
    MAX_REVIEWS_PER_CATEGORY = int(os.getenv("MAX_REVIEWS_PER_CATEGORY", "1000"))
    
    # Output
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/output")
    
    # Dataset
    AMAZON_DATASET = "McAuley-Lab/Amazon-Reviews-2023"
    TARGET_CATEGORY = os.getenv("TARGET_CATEGORY", "Clothing_Shoes_and_Jewelry")

config = Config()
