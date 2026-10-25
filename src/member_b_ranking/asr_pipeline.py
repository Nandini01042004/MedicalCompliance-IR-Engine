"""
Spoken Audio & ASR Pipeline Handler.
Manages audio ingestion, transcription simulation, confidence scoring,
and acoustic noise degradation for speech robustness benchmarking.
"""

from typing import Dict, Any, Tuple
from src.member_b_ranking.normalizer import MedicalPhoneticNormalizer


class ASRPipeline:
    """
    Handles speech input conversion, confidence tracking, and ASR phonetic normalization.
    """
    def __init__(self, normalizer: MedicalPhoneticNormalizer = None):
        self.normalizer = normalizer or MedicalPhoneticNormalizer()

    def process_spoken_input(
        self,
        raw_text_transcript: str,
        confidence_score: float = 1.0,
        simulate_asr_noise: bool = False
    ) -> Dict[str, Any]:
        """
        Processes spoken input:
        1. Optionally injects realistic ASR corruption for noise resilience experiments.
        2. Applies medical phonetic normalization.
        3. Returns audit trail with original, corrupted, and corrected transcripts.
        """
        transcript = raw_text_transcript.strip()
        corrupted_transcript = transcript

        # If benchmarking speech noise degradation, simulate phonetic errors
        if simulate_asr_noise:
            # Common realistic ASR mishearing substitutions
            noise_substitutions = {
                "hipaa": "hippa",
                "schedule h1": "schedule age 1",
                "schedule x": "schedule axe",
                "cdsco": "see dsco",
                "wfi": "water for inject"
            }
            for real, noisy in noise_substitutions.items():
                corrupted_transcript = corrupted_transcript.replace(real, noisy)

        # Apply phonetic normalization
        normalized_query, corrections = self.normalizer.normalize(corrupted_transcript)

        return {
            "raw_input": raw_text_transcript,
            "corrupted_transcript": corrupted_transcript if simulate_asr_noise else None,
            "normalized_query": normalized_query,
            "confidence_score": confidence_score,
            "phonetic_corrections": corrections
        }
