"""Multimodal fusion module for combining audio, video, and text signals."""

from __future__ import annotations


def classify_tone(segment: dict) -> dict:
    """Classify final tone label and confidence from merged multimodal signals.

    Parameters
    ----------
    segment:
        Merged segment dictionary from ``sync.py`` with the following keys:
        ``text``, ``timestamp``, ``irony_label``, ``irony_score``,
        ``emotion_label``, ``emotion_score``, ``face_emotion``, ``face_confidence``.

    Returns
    -------
    dict
        A copy of the original segment with additional keys:
        ``tone_label`` and ``tone_confidence``.
    """
    # Define grouped emotion classes used by the rules.
    positive_nlp_emotions = {"joy", "surprise"}
    negative_face_emotions = {"angry", "sad", "fear", "disgust"}
    positive_face_emotions = {"happy", "surprise"}

    # Safely pull values from the merged segment while normalizing text labels.
    irony_label = str(segment.get("irony_label", "")).lower()
    irony_score = float(segment.get("irony_score", 0.0))
    emotion_label = str(segment.get("emotion_label", "")).lower()
    emotion_score = float(segment.get("emotion_score", 0.0))
    face_emotion = str(segment.get("face_emotion", "")).lower()
    face_confidence = float(segment.get("face_confidence", 0.0))

    # Helper for repeated average calculations in the rules.
    def avg(a: float, b: float) -> float:
        return (a + b) / 2.0

    # Apply rules in strict order. The first match determines the final output.
    if irony_label == "irony" and irony_score > 0.7 and face_emotion in negative_face_emotions:
        # Rule 1: Strong irony + negative facial cue -> sarcastic.
        tone_label = "Sarcastic"
        tone_confidence = avg(irony_score, face_confidence)
    elif irony_label == "irony" and irony_score > 0.6:
        # Rule 2: Moderate/strong irony alone can still indicate sarcasm.
        tone_label = "Sarcastic"
        tone_confidence = irony_score * 0.85
    elif (
        emotion_label in positive_nlp_emotions
        and face_emotion in negative_face_emotions
        and face_confidence > 0.5
    ):
        # Rule 3: Positive words + negative face mismatch suggests sarcasm.
        tone_label = "Sarcastic"
        tone_confidence = avg(emotion_score, face_confidence) * 0.9
    elif face_emotion in positive_face_emotions and emotion_label == "neutral":
        # Rule 4: Positive facial expression with neutral text -> playful/joking tone.
        tone_label = "Joking / Playful"
        tone_confidence = face_confidence
    elif emotion_label in positive_nlp_emotions and face_emotion in positive_face_emotions:
        # Rule 5: Positive text and positive face align.
        tone_label = "Sincere / Positive"
        tone_confidence = avg(emotion_score, face_confidence)
    else:
        # Rule 6: Fallback when no stronger multimodal signal is found.
        tone_label = "Neutral / Sincere"
        tone_confidence = 0.6

    # Return all original fields plus final decision; round confidence to 2 decimals.
    result = dict(segment)
    result["tone_label"] = tone_label
    result["tone_confidence"] = round(tone_confidence, 2)
    return result
