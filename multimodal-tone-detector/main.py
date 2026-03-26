"""CLI entry point for multimodal tone detection.

Modes
-----
- live: read microphone + webcam in background threads and fuse results in near real time.
- file: process a video file end-to-end (ffmpeg audio extraction + Whisper + DeepFace).
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import whisper
from deepface import DeepFace

from pipeline import fusion, nlp, sync


def _parse_deepface_emotion(analysis: Any) -> tuple[str, float]:
    """Normalize DeepFace output into (emotion, confidence)."""
    if isinstance(analysis, list):
        if not analysis:
            return "unknown", 0.0
        analysis = analysis[0]

    emotion_scores = analysis.get("emotion", {}) if isinstance(analysis, dict) else {}
    dominant = analysis.get("dominant_emotion", "unknown") if isinstance(analysis, dict) else "unknown"
    return str(dominant), float(emotion_scores.get(dominant, 0.0))


def run_live_mode() -> None:
    """Run real-time inference by draining audio/video queues every 5 seconds."""
    # Import inside mode branch so file mode can run without microphone dependencies.
    from pipeline.audio import audio_queue, capture_audio
    from pipeline.video import run_video_loop

    # Dedicated stop event managed by main loop.
    stop_event = threading.Event()
    video_queue: queue.Queue[dict[str, Any]] = queue.Queue()

    audio_thread = threading.Thread(target=capture_audio, name="AudioCapture", daemon=True)
    video_thread = threading.Thread(
        target=run_video_loop,
        args=(video_queue, stop_event),
        kwargs={"show_preview": True},
        name="VideoCapture",
        daemon=True,
    )

    print("[Live] Loading Whisper model 'base'...")
    whisper_model = whisper.load_model("base")

    print("[Live] Starting audio + video threads...")
    audio_thread.start()
    video_thread.start()

    video_history: list[dict[str, Any]] = []

    try:
        while True:
            # Process in 5-second windows as requested.
            time.sleep(5)

            drained_audio: list[dict[str, Any]] = []
            drained_video: list[dict[str, Any]] = []

            # Drain all available audio items from the shared queue.
            while True:
                try:
                    drained_audio.append(audio_queue.get_nowait())
                except queue.Empty:
                    break

            # Drain all available video items from the shared queue.
            while True:
                try:
                    drained_video.append(video_queue.get_nowait())
                except queue.Empty:
                    break

            if drained_video:
                video_history.extend(drained_video)

            # Transcribe each audio chunk, run NLP, then sync + fuse against video history.
            audio_results: list[dict[str, Any]] = []
            for item in drained_audio:
                result = whisper_model.transcribe(item["audio"], fp16=False)
                text = result.get("text", "").strip()
                if not text:
                    continue

                nlp_result = nlp.analyze_text(text)
                audio_results.append(
                    {
                        "text": text,
                        "timestamp": float(item.get("timestamp", 0.0)),
                        **nlp_result,
                    }
                )

            if not audio_results:
                continue

            synced = sync.sync_segments(audio_results, video_history)
            fused = [fusion.classify_tone(segment) for segment in synced]

            for segment in fused:
                print(
                    f"[Live][{segment['timestamp']:.2f}s] "
                    f"Tone={segment['tone_label']} "
                    f"Confidence={segment['tone_confidence']:.2f}"
                )
    except KeyboardInterrupt:
        print("\n[Live] KeyboardInterrupt received, shutting down...")
    finally:
        stop_event.set()
        # Also signal pipeline.audio loop to stop if available.
        try:
            from pipeline.audio import stop_event as audio_stop_event

            audio_stop_event.set()
        except Exception:
            pass

        audio_thread.join(timeout=5)
        video_thread.join(timeout=5)
        print("[Live] Shutdown complete.")


def _extract_audio_with_ffmpeg(input_video: str, output_wav: str) -> None:
    """Extract mono 16kHz WAV audio from input video using ffmpeg."""
    cmd = ["ffmpeg", "-y", "-i", input_video, "-ar", "16000", "-ac", "1", output_wav]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _extract_video_emotions_1fps(input_video: str) -> list[dict[str, Any]]:
    """Sample video at 1 FPS, run DeepFace per sampled frame, and return results."""
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(int(round(fps)), 1)  # one frame every ~1 second

    frame_index = 0
    video_results: list[dict[str, Any]] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_interval == 0:
                timestamp = frame_index / fps
                try:
                    analysis = DeepFace.analyze(
                        frame,
                        actions=["emotion"],
                        enforce_detection=True,
                    )
                    emotion, confidence = _parse_deepface_emotion(analysis)
                    video_results.append(
                        {
                            "timestamp": float(timestamp),
                            "emotion": emotion,
                            "confidence": confidence,
                        }
                    )
                except Exception:
                    # Ignore frames with no face detected or transient model errors.
                    pass

            frame_index += 1
    finally:
        cap.release()

    return video_results


def run_file_mode(input_video: str) -> None:
    """Run end-to-end processing for an existing video file and print timeline."""
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input file not found: {input_video}")

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = str(Path(temp_dir) / "audio.wav")

        # 1) Extract audio using the exact ffmpeg transform requested.
        _extract_audio_with_ffmpeg(input_video, audio_path)

        # 2) One-pass Whisper transcription over the full WAV file.
        print("[File] Loading Whisper model 'base'...")
        whisper_model = whisper.load_model("base")
        whisper_result = whisper_model.transcribe(audio_path, fp16=False)
        whisper_segments = whisper_result.get("segments", [])

        # 3) Extract 1 FPS visual features with OpenCV + DeepFace.
        video_results = _extract_video_emotions_1fps(input_video)

    # 4) NLP on every transcript segment.
    audio_results: list[dict[str, Any]] = []
    for seg in whisper_segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        # Use segment midpoint for better temporal matching with visual samples.
        start_ts = float(seg.get("start", 0.0))
        end_ts = float(seg.get("end", start_ts))
        midpoint = (start_ts + end_ts) / 2.0

        nlp_result = nlp.analyze_text(text)
        audio_results.append(
            {
                "text": text,
                "timestamp": midpoint,
                "start": start_ts,
                "end": end_ts,
                **nlp_result,
            }
        )

    # 5) Sync + fuse and print complete annotated timeline.
    synced = sync.sync_segments(audio_results, video_results)
    fused_timeline = [fusion.classify_tone(segment) for segment in synced]

    print("\n=== Annotated Tone Timeline ===")
    for row in fused_timeline:
        print(
            f"[{row.get('start', row['timestamp']):.2f}s - {row.get('end', row['timestamp']):.2f}s] "
            f"text={row['text']!r} | "
            f"nlp_irony={row['irony_label']}({row['irony_score']:.2f}) | "
            f"nlp_emotion={row['emotion_label']}({row['emotion_score']:.2f}) | "
            f"face={row['face_emotion']}({row['face_confidence']:.2f}) | "
            f"tone={row['tone_label']}({row['tone_confidence']:.2f})"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create and configure the CLI parser."""
    parser = argparse.ArgumentParser(description="Multimodal tone detection CLI")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["live", "file"],
        help="Run real-time capture (live) or process an existing video (file).",
    )
    parser.add_argument(
        "--input",
        help="Path to input video file (required when --mode file).",
    )
    return parser


def main() -> None:
    """Dispatch execution based on command-line mode."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "live":
        run_live_mode()
        return

    # mode == "file"
    if not args.input:
        parser.error("--input is required when --mode file")
    run_file_mode(args.input)


if __name__ == "__main__":
    main()
