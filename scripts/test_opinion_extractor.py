from services.opinion_extractor import OpinionUnitExtractor

def test_extraction():
    extractor = OpinionUnitExtractor()
    
    test_reviews = [
        "The fabric is soft but the zipper is terrible. It fits small.",
        "Beautiful shoes but very uncomfortable for walking long distances.",
        "The sizing was ridiculous, I had to size up twice.",
        "Not warm enough for winter. Very thin material.",
        "I love the pink burgundy color, it looks exactly like the photo."
    ]
    
    print("=" * 60)
    print("🧪 TESTING OPINION UNIT EXTRACTION")
    print("=" * 60)
    
    for review in test_reviews:
        print(f"\nReview: {review}")
        units = extractor.extract(review)
        for i, unit in enumerate(units):
            print(f"  Unit {i+1}: {unit}")

if __name__ == "__main__":
    test_extraction()
