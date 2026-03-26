"""Video analysis module for deriving visual cues relevant to tone detection."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

import cv2
from deepface import DeepFace


def _parse_emotion_result(analysis: Any) -> tuple[str, float]:
    """Extract dominant emotion and confidence from DeepFace output.

    DeepFace can return either a dictionary or a list of dictionaries depending
    on version/options. This helper normalizes both formats.
    """
    if isinstance(analysis, list):
        if not analysis:
            raise ValueError("DeepFace returned an empty analysis list")
        analysis = analysis[0]

    dominant_emotion = analysis.get("dominant_emotion")
    emotion_scores = analysis.get("emotion", {})

    if dominant_emotion is None:
        raise ValueError("DeepFace analysis did not include dominant_emotion")

    confidence = float(emotion_scores.get(dominant_emotion, 0.0))
    return str(dominant_emotion), confidence


def run_video_loop(
    video_queue: queue.Queue,
    stop_event: threading.Event,
    camera_index: int = 0,
    show_preview: bool = True,
) -> None:
    """Capture webcam frames, run emotion analysis, and enqueue results.

    Args:
        video_queue: Thread-safe queue to store emotion predictions.
        stop_event: Event used to stop the loop cleanly from another thread.
        camera_index: Webcam index passed to cv2.VideoCapture.
        show_preview: If True, display webcam feed with emotion overlay.
    """
    # Open the webcam stream.
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    # Track timing so we only analyze one frame per second.
    start_time = time.time()
    last_sample_time = 0.0
    latest_emotion_text = "emotion: waiting..."

    try:
        while not stop_event.is_set():
            # Grab a frame from the camera.
            ret, frame = cap.read()
            if not ret:
                # If frame capture fails, wait briefly and retry.
                time.sleep(0.05)
                continue

            now = time.time()

            # Skip analysis unless at least one second passed.
            if now - last_sample_time >= 1.0:
                last_sample_time = now
                try:
                    # Analyze emotion only; don't enforce detection to be strict
                    # with tiny/partial faces, but still guard with exception.
                    analysis = DeepFace.analyze(
                        frame,
                        actions=["emotion"],
                        enforce_detection=True,
                    )
                    emotion, confidence = _parse_emotion_result(analysis)

                    # Put normalized result into the thread-safe queue.
                    video_queue.put(
                        {
                            "emotion": emotion,
                            "confidence": confidence,
                            # Use relative timestamp so it aligns with audio timestamps
                            # produced by pipeline.audio.capture_audio.
                            "timestamp": now - start_time,
                        }
                    )
                    latest_emotion_text = f"emotion: {emotion} ({confidence:.1f}%)"
                except Exception:
                    # If no face is detected or analysis fails, skip this frame
                    # gracefully and continue the loop without crashing.
                    latest_emotion_text = "emotion: no face detected"

            if show_preview:
                # Draw the latest detected emotion label on the frame.
                cv2.putText(
                    frame,
                    latest_emotion_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Tone Detection - Video", frame)

                # Allow the user to close preview with "q".
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break
    finally:
        # Always release hardware/UI resources.
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create shared queue/event objects for communication and shutdown.
    video_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    # Run the video loop in a background thread so main thread can wait for
    # user input (Enter) or KeyboardInterrupt.
    video_thread = threading.Thread(
        target=run_video_loop,
        args=(video_queue, stop_event),
        kwargs={"show_preview": True},
        daemon=True,
    )
    video_thread.start()

    try:
        input("Video loop running. Press Enter to stop...\n")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received; stopping video loop...")
    finally:
        stop_event.set()
        video_thread.join(timeout=5)
        print("Video loop stopped.")
