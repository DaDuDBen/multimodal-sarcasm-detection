"""Synchronization module for aligning multimodal inputs across time."""

from typing import Dict, List, Tuple


def get_dominant_face_emotion(matches: List[Dict]) -> Tuple[str, float]:
    """Return the highest-confidence face emotion from matched video results.

    Args:
        matches: Video inference dictionaries that fall within an audio segment's
            timestamp window.

    Returns:
        A tuple of (emotion_label, confidence_score). If no matches are provided,
        this function returns ("unknown", 0.0).
    """
    # If we have no matching face detections for this window, return defaults.
    if not matches:
        return "unknown", 0.0

    # Select the match with the largest confidence score.
    best_match = max(matches, key=lambda result: result.get("confidence", 0.0))

    # Read output fields safely with fallback defaults.
    emotion = best_match.get("emotion", "unknown")
    confidence = float(best_match.get("confidence", 0.0))
    return emotion, confidence


def sync_segments(audio_results: List[Dict], video_results: List[Dict]) -> List[Dict]:
    """Synchronize audio segments with nearby video emotion predictions.

    For each audio segment, this function searches for video predictions whose
    timestamps are within +/- 2.5 seconds of the audio timestamp. Among those
    matched predictions, the highest-confidence one is selected as the
    representative facial emotion.

    Args:
        audio_results: List of dictionaries with keys:
            text, timestamp, irony_label, irony_score, emotion_label,
            emotion_score.
        video_results: List of dictionaries with keys:
            emotion, confidence, timestamp.

    Returns:
        A list of merged dictionaries containing all original audio fields plus:
            face_emotion (str), face_confidence (float).
    """
    # Fixed temporal tolerance in seconds for matching audio and video outputs.
    window_seconds = 2.5

    merged_results: List[Dict] = []

    # Process each audio segment independently.
    for audio_segment in audio_results:
        audio_timestamp = float(audio_segment.get("timestamp", 0.0))

        # Gather all video results whose timestamps lie inside the +/- window.
        matches = [
            video_segment
            for video_segment in video_results
            if abs(float(video_segment.get("timestamp", 0.0)) - audio_timestamp)
            <= window_seconds
        ]

        # Pick the most confident face emotion among the matched video segments.
        face_emotion, face_confidence = get_dominant_face_emotion(matches)

        # Merge all audio fields and append synchronized face fields.
        merged_segment = {
            **audio_segment,
            "face_emotion": face_emotion,
            "face_confidence": face_confidence,
        }
        merged_results.append(merged_segment)

    return merged_results
