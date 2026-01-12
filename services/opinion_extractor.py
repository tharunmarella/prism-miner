"""
Opinion Unit Extractor (V2 - Improved)

Extracts structured opinion units from review text using spaCy dependency parsing.
An Opinion Unit is a (Aspect, Opinion) pair like "fabric: soft" or "zipper: broke".
"""

import spacy
from typing import List, Tuple, Set

class OpinionUnitExtractor:
    """
    Extracts structured opinion units from raw text.
    
    Patterns extracted:
    1. ADJ modifying NOUN: "soft fabric" → "fabric: soft"
    2. NOUN + copula + ADJ: "fabric is soft" → "fabric: soft"
    3. NOUN + VERB: "zipper broke" → "zipper: broke"
    4. ADJ + NOUN phrases: "thin material" → "material: thin"
    5. Comparative phrases: "runs small" → "sizing: runs small"
    """

    # Common aspect keywords that indicate product attributes
    ASPECT_KEYWORDS = {
        'fabric', 'material', 'quality', 'fit', 'size', 'sizing', 'color', 'colour',
        'zipper', 'button', 'buttons', 'stitching', 'seam', 'seams', 'hem',
        'waist', 'waistband', 'elastic', 'pocket', 'pockets', 'sleeve', 'sleeves',
        'collar', 'neckline', 'length', 'width', 'thickness', 'weight',
        'comfort', 'support', 'cushion', 'padding', 'lining', 'fleece',
        'price', 'value', 'shipping', 'delivery', 'packaging',
        'heel', 'toe', 'arch', 'sole', 'insole', 'ankle', 'strap',
        'design', 'style', 'look', 'appearance', 'print', 'pattern',
    }

    # Sizing-related verbs
    SIZING_VERBS = {'runs', 'fits', 'ran', 'fit'}
    
    # Quality-related verbs  
    QUALITY_VERBS = {'broke', 'ripped', 'tore', 'shrunk', 'faded', 'pilled', 'stretched', 'fell'}

    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            import subprocess
            subprocess.run(["python3", "-m", "spacy", "download", model])
            self.nlp = spacy.load(model)

    def extract(self, text: str) -> List[str]:
        """
        Extract all opinion units from the text.
        """
        if not text or not isinstance(text, str):
            return []

        doc = self.nlp(text)
        units: Set[str] = set()

        for token in doc:
            # Pattern 1: Adjective modifying noun directly ("soft fabric")
            if token.pos_ == "ADJ" and token.dep_ == "amod":
                aspect = self._get_full_noun(token.head)
                if aspect:
                    units.add(f"{aspect}: {token.text.lower()}")

            # Pattern 2: Predicate adjective ("fabric is soft")
            elif token.pos_ == "ADJ" and token.dep_ in ["acomp", "attr"]:
                subject = self._find_subject(token.head)
                if subject:
                    # Also get conjoined adjectives ("soft and comfortable")
                    adjs = [token.text.lower()]
                    for child in token.children:
                        if child.dep_ == "conj" and child.pos_ == "ADJ":
                            adjs.append(child.text.lower())
                    for adj in adjs:
                        units.add(f"{subject}: {adj}")

            # Pattern 3: Verb with subject ("zipper broke", "buttons fell off")
            elif token.pos_ == "VERB":
                if token.lemma_.lower() in self.QUALITY_VERBS:
                    subject = self._find_subject(token)
                    if subject:
                        units.add(f"{subject}: {token.lemma_.lower()}")
                
                # Sizing verbs ("runs small", "fits tight")
                elif token.lemma_.lower() in self.SIZING_VERBS:
                    modifier = self._find_modifier(token)
                    if modifier:
                        units.add(f"sizing: {token.text.lower()} {modifier}")

            # Pattern 4: Noun phrases with aspect keywords
            elif token.pos_ == "NOUN" and token.text.lower() in self.ASPECT_KEYWORDS:
                # Check for adjective children
                for child in token.children:
                    if child.pos_ == "ADJ":
                        units.add(f"{token.text.lower()}: {child.text.lower()}")
                
                # Check if this noun is subject of a predicate
                if token.dep_ == "nsubj":
                    head = token.head
                    if head.pos_ == "ADJ":
                        units.add(f"{token.text.lower()}: {head.text.lower()}")

        # Convert to sorted list
        return sorted(list(units))

    def _get_full_noun(self, token) -> str:
        """Get noun with any compound modifiers."""
        if token.pos_ not in ["NOUN", "PROPN"]:
            return ""
        
        compounds = []
        for child in token.children:
            if child.dep_ == "compound":
                compounds.append(child.text.lower())
        
        compounds.append(token.text.lower())
        return " ".join(compounds)

    def _find_subject(self, token) -> str:
        """Find the subject of a verb or adjective."""
        # Direct children
        for child in token.children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                return self._get_full_noun(child)
        
        # Check parent if this is a conjunction
        if token.dep_ == "conj":
            return self._find_subject(token.head)
        
        return ""

    def _find_modifier(self, token) -> str:
        """Find adverbial or adjectival modifiers."""
        for child in token.children:
            if child.dep_ in ["advmod", "acomp"] and child.pos_ in ["ADV", "ADJ"]:
                return child.text.lower()
        return ""
