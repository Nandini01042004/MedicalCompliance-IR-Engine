"""
Medical & Legal Domain Tokenizer with Conversational Filler Filtering.
Provides robust tokenization, normalization, and stemming while preserving
critical regulatory tokens (e.g. § section numbers, schedule codes, CFR parts).
"""

import re
from typing import List, Set

# Conversational speech filler words to strip from spoken queries
SPOKEN_FILLERS: Set[str] = {
    "um", "uh", "er", "ah", "like", "okay", "ok", "hey", "hello", "hi",
    "please", "can", "you", "tell", "me", "what", "are", "the", "is",
    "could", "would", "how", "do", "we", "know", "about", "i", "need",
    "want", "to", "find", "looking", "for", "give", "show", "check",
    "whether", "rules", "rule", "regulations", "requirements"
}

# Standard English stopwords (excluding domain-relevant prepositions if vital)
STANDARD_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "cannot", "could",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "would", "you", "your", "yours"
}


class PorterStemmerSimple:
    """
    Lightweight, deterministic suffix stemmer suitable for medical/legal terms.
    Preserves acronyms and digits (e.g. '164.514', 'h1', 'gmp').
    """
    def stem(self, word: str) -> str:
        word = word.lower()
        if len(word) <= 3 or word.isdigit() or "_" in word:
            return word
        
        # Strip common inflectional suffixes
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        if word.endswith("ss"):
            return word
        if word.endswith("s") and not word.endswith("us") and not word.endswith("is"):
            word = word[:-1]
        
        if word.endswith("ingly"):
            word = word[:-5]
        elif word.endswith("ing") and len(word) > 5:
            word = word[:-3]
        elif word.endswith("ation") and len(word) > 6:
            word = word[:-5] + "ate"
        elif word.endswith("tional") and len(word) > 7:
            word = word[:-2]
        elif word.endswith("ment") and len(word) > 6:
            word = word[:-4]
        elif word.endswith("able") and len(word) > 6:
            word = word[:-4]
        elif word.endswith("ed") and len(word) > 4:
            word = word[:-2]
            
        return word


class MedicalLegalTokenizer:
    """Tokenizer tailored for medical compliance documents and spoken queries."""
    
    def __init__(self, filter_fillers: bool = True, filter_stopwords: bool = True, stem: bool = True):
        self.filter_fillers = filter_fillers
        self.filter_stopwords = filter_stopwords
        self.stem = stem
        self.stemmer = PorterStemmerSimple()
        
        # Regex preserving section identifiers: e.g. 164.514, 21_cfr, part_11, sch_h1
        self.token_pattern = re.compile(r"[a-zA-Z0-9]+(?:[._\-][a-zA-Z0-9]+)*")

    def tokenize(self, text: str) -> List[str]:
        """Splits text into cleaned and normalized tokens."""
        if not text:
            return []
        
        # Normalize punctuation while keeping section dots/hyphens
        normalized = text.lower().replace("§", "section ").replace("/", " ")
        raw_matches = self.token_pattern.findall(normalized)
        
        tokens: List[str] = []
        for token in raw_matches:
            token = token.strip("._-")
            if not token:
                continue
            
            # Filter spoken fillers if requested
            if self.filter_fillers and token in SPOKEN_FILLERS:
                continue
                
            # Filter standard stopwords
            if self.filter_stopwords and token in STANDARD_STOPWORDS:
                continue
            
            # Apply stemming
            if self.stem:
                token = self.stemmer.stem(token)
                
            if len(token) >= 2 or token.isdigit():
                tokens.append(token)
                
        return tokens
