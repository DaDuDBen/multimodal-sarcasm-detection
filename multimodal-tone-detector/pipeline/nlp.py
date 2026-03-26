"""Natural language processing module for text-based tone understanding.

This module loads two Hugging Face text classification pipelines once at import time
and exposes ``analyze_text`` for downstream use in the multimodal system.
"""

from __future__ import annotations

from typing import Dict

from transformers import pipeline

# ---------------------------------------------------------------------------
# Load and cache models once on startup.
#
# These objects are module-level singletons, so callers can reuse them for many
# requests without paying repeated model load costs.
# ---------------------------------------------------------------------------
IRONY_MODEL_NAME = "cardiffnlp/twitter-roberta-base-irony"
EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

irony_classifier = pipeline(task="text-classification", model=IRONY_MODEL_NAME)
emotion_classifier = pipeline(task="text-classification", model=EMOTION_MODEL_NAME)


# Neutral/safe defaults for empty inputs.
NEUTRAL_RESULT = {
    "irony_label": "non_irony",
    "irony_score": 0.0,
    "emotion_label": "neutral",
    "emotion_score": 0.0,
}


def _normalize_irony_label(raw_label: str) -> str:
    """Normalize model output labels to either 'irony' or 'non_irony'."""
    normalized = (raw_label or "").strip().lower()

    # Many HF classifiers use generic labels such as LABEL_0/LABEL_1.
    # For this model, LABEL_1 is typically irony and LABEL_0 is non-irony.
    if normalized in {"label_1", "1", "irony", "sarcasm", "ironic"}:
        return "irony"

    return "non_irony"


def analyze_text(text: str) -> Dict[str, float | str]:
    """Analyze text for irony and emotion.

    Args:
        text: Input text to classify.

    Returns:
        Dictionary with normalized irony and emotion labels + confidence scores.
    """
    # Handle empty/whitespace-only text gracefully.
    if text is None or not str(text).strip():
        return dict(NEUTRAL_RESULT)

    clean_text = str(text).strip()

    # Run both models on the same input text.
    irony_output = irony_classifier(clean_text)[0]
    emotion_output = emotion_classifier(clean_text)[0]

    # Normalize irony labels to the required output contract.
    irony_label = _normalize_irony_label(irony_output.get("label", ""))

    # Emotion labels are returned directly by the emotion model.
    emotion_label = str(emotion_output.get("label", "neutral")).strip().lower() or "neutral"

    return {
        "irony_label": irony_label,
        "irony_score": float(irony_output.get("score", 0.0)),
        "emotion_label": emotion_label,
        "emotion_score": float(emotion_output.get("score", 0.0)),
    }


if __name__ == "__main__":
    # Quick sanity checks with three example sentences:
    # 1) sincere, 2) sarcastic, 3) neutral.
    examples = [
        "I really appreciate your help today. Thank you so much!",
        "Oh great, another Monday morning traffic jam. Exactly what I needed.",
        "The meeting is at 3 PM in conference room B.",
    ]

    for idx, sample in enumerate(examples, start=1):
        result = analyze_text(sample)
        print(f"Example {idx}: {sample}")
        print(result)
        print("-" * 60)
