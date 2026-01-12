# Prism Miner

Large-scale product knowledge extraction and review mining service using SOTA NLP techniques.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRISM MINER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  HuggingFace │    │   Opinion    │    │  Groq Batch  │       │
│  │  Review Loader│───▶│   Extractor  │───▶│     API      │       │
│  │  (Streaming) │    │   (spaCy)    │    │  (GPT-OSS)   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              87M Reviews → 2,195 Categories          │       │
│  │              → Hidden Dimensions per Category        │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Opinion Unit Extraction**: Uses spaCy dependency parsing to split reviews into pure `(Aspect, Opinion)` pairs
- **Streaming Processing**: Downloads and processes reviews without loading everything into memory
- **Groq Batch API**: 50% cost reduction with 24h processing window
- **Category-Aware Mining**: Filters reviews by 2,195 leaf clothing categories

## Cost Estimate

| Reviews/Category | Total Reviews | Groq Batch Cost | Time |
|------------------|---------------|-----------------|------|
| 100 | 219,500 | ~$2.00 | 24h |
| 500 | 1,097,500 | ~$9.00 | 24h |
| 1,000 | 2,195,000 | ~$16.00 | 24h |

## Environment Variables

```bash
# Required
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token

# Optional
GROQ_MODEL=openai/gpt-oss-120b
MAX_REVIEWS_PER_CATEGORY=1000
TARGET_CATEGORY=Clothing_Shoes_and_Jewelry
WAIT_FOR_COMPLETION=false
```

## Usage

### Local Development

```bash
cd prism_miner
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run the pipeline
python -m prism_miner.main

# Check batch status
python -m prism_miner.main status <batch_id>

# Download results
python -m prism_miner.main download <batch_id>
```

### Deploy to Railway

1. Push to your repository
2. Connect Railway to the repo
3. Set environment variables in Railway dashboard
4. Deploy using the Dockerfile

## Output

The pipeline produces:

- `batch_info.json`: Batch job metadata
- `batch_results.jsonl`: Raw Groq responses
- `taxonomy_dimensions.json`: Final structured taxonomy

Example output:

```json
{
  "T-Shirts": {
    "dimensions": [
      {
        "name": "Fabric Softness",
        "importance": "High",
        "description": "How soft the fabric feels against the skin",
        "example_vocabulary": ["soft", "buttery", "comfortable", "scratchy", "rough"]
      },
      {
        "name": "Size Accuracy",
        "importance": "High", 
        "description": "Whether the sizing matches standard US sizes",
        "example_vocabulary": ["runs small", "true to size", "size up", "tight", "loose"]
      }
    ]
  }
}
```

## Pipeline Stages

### Stage 1: Opinion Unit Extraction (Local)
- Uses spaCy's dependency parser
- Splits "The fabric is soft but the zipper broke" into:
  - `fabric: soft`
  - `zipper: broke`

### Stage 2: Batch Mining (Groq)
- Groups opinion units by category
- Submits to Groq Batch API (50% cheaper)
- LLM synthesizes dimensions from patterns

### Stage 3: Taxonomy Integration (Future)
- Results integrated into Weaviate
- Powers product enrichment pipeline
