"""
Medical & Legal Phonetic Normalizer.
Detects and repairs Automatic Speech Recognition (ASR) transcription errors
on specialized medical, pharmaceutical, and regulatory acronyms and terminology.
"""

import re
from typing import Dict, List, Tuple


# Soundex algorithm for phonetic representation
def soundex(token: str) -> str:
    """Computes standard Soundex code for a word."""
    token = token.upper()
    if not token or not token.isalpha():
        return ""
    
    first_char = token[0]
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    encoded = [first_char]
    for char in token[1:]:
        digit = mapping.get(char, '0')
        if digit != '0' and digit != encoded[-1]:
            encoded.append(digit)
            
    # Pad or truncate to 4 characters
    code = "".join(encoded)
    return (code + "000")[:4]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes dynamic programming Levenshtein edit distance."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


class MedicalPhoneticNormalizer:
    """
    Repairs common ASR phonetically corrupted phrases into standardized regulatory terms.
    """
    # Direct dictionary of known ASR mishearing mappings
    KNOWN_ASR_MAPPINGS: Dict[str, str] = {
        "hippa": "hipaa",
        "heepa": "hipaa",
        "hippo": "hipaa",
        "hipa": "hipaa",
        "see dsco": "cdsco",
        "cd sco": "cdsco",
        "see disco": "cdsco",
        "b a a": "baa",
        "b.a.a.": "baa",
        "p h i": "phi",
        "p.h.i.": "phi",
        "schedule age": "schedule h",
        "schedule 8": "schedule h",
        "schedule h one": "schedule h1",
        "schedule age 1": "schedule h1",
        "schedule 8 1": "schedule h1",
        "schedule h-1": "schedule h1",
        "schedule ex": "schedule x",
        "schedule axe": "schedule x",
        "schedule 10": "schedule x",
        "schedule em": "schedule m",
        "part eleven": "part 11",
        "21 cfr": "21 cfr",
        "c f r": "cfr",
        "hepa filter": "hepa",
        "keeper filter": "hepa",
        "w f i": "wfi",
        "water for inject": "water for injection",
        "d identify": "de-identify",
        "day identify": "de-identify"
    }

    # Canonical compliance dictionary
    CANONICAL_TERMS: List[str] = [
        "hipaa", "cdsco", "cfr", "part 11", "schedule h", "schedule h1",
        "schedule x", "schedule m", "gmp", "phi", "baa", "hepa", "wfi",
        "de-identify", "safe harbor", "anonymization", "cleanroom",
        "effluent", "validation", "audit trail", "biometric", "encryption"
    ]

    def normalize(self, raw_transcript: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Normalizes spoken query text.
        Returns:
            - normalized_text: Cleaned text with corrections applied.
            - corrections_made: List of replacement logs for explainability.
        """
        text = raw_transcript.strip().lower()
        corrections: List[Dict[str, str]] = []

        # Step 1: Multi-word phrase regex replacements
        for wrong, correct in self.KNOWN_ASR_MAPPINGS.items():
            pattern = r'\b' + re.escape(wrong) + r'\b'
            if re.search(pattern, text):
                text = re.sub(pattern, correct, text)
                corrections.append({"original": wrong, "corrected": correct, "strategy": "exact_phrase"})

        # Step 2: Word-level phonetic matching for isolated corrupted tokens
        words = text.split()
        normalized_words = []
        for w in words:
            # Check Levenshtein against canonical terms
            best_match = None
            min_dist = 999
            w_soundex = soundex(w)
            
            for canon in self.CANONICAL_TERMS:
                if " " in canon:
                    continue  # already handled in phrase step
                dist = levenshtein_distance(w, canon)
                if dist <= 2 and (dist < min_dist) and (w_soundex == soundex(canon) or dist <= 1):
                    min_dist = dist
                    best_match = canon

            if best_match and best_match != w and min_dist <= 2 and len(w) > 3:
                normalized_words.append(best_match)
                corrections.append({"original": w, "corrected": best_match, "strategy": "phonetic_soundex"})
            else:
                normalized_words.append(w)

        return " ".join(normalized_words), corrections
