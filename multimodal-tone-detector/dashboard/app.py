"""Streamlit dashboard entry module for presenting multimodal tone insights.

This app supports two operating modes:
1) Live Webcam mode (real-time audio + video threads)
2) Upload Video File mode (offline batch processing)
"""

from __future__ import annotations

import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import ffmpeg
import numpy as np
import pandas as pd
import streamlit as st
import whisper

from pipeline import audio as audio_pipeline
from pipeline import fusion
from pipeline import nlp
from pipeline import sync
from pipeline import video as video_pipeline

# -----------------------------------------------------------------------------
# Dashboard constants
# -----------------------------------------------------------------------------
TONE_COLORS = {
    "Sarcastic": "#FF6B00",
    "Sincere / Positive": "#2ECC71",
    "Joking / Playful": "#F1C40F",
    "Neutral / Sincere": "#95A5A6",
}

LIVE_REFRESH_SECONDS = 1.0


# -----------------------------------------------------------------------------
# Session state bootstrap
# -----------------------------------------------------------------------------
def _init_state() -> None:
    """Initialize all Streamlit session keys used by the app."""
    defaults: dict[str, Any] = {
        "live_running": False,
        "stop_event": None,
        "audio_capture_thread": None,
        "audio_transcribe_thread": None,
        "video_thread": None,
        "video_consumer_thread": None,
        "video_queue": queue.Queue(),
        "transcript_updates_queue": queue.Queue(),
        "emotion_updates_queue": queue.Queue(),
        "live_shared_state": {"latest_face_emotion": "unknown"},
        "live_shared_state_lock": threading.Lock(),
        "history": [],
        "latest_transcript": "Waiting for speech...",
        "latest_face_emotion": "unknown",
        "latest_tone_label": "Neutral / Sincere",
        "latest_confidence": 0.0,
        "live_whisper_model": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# -----------------------------------------------------------------------------
# Live mode workers
# -----------------------------------------------------------------------------
def _live_transcription_worker(
    stop_event: threading.Event,
    model: Any,
    shared_state: dict[str, Any],
    shared_state_lock: threading.Lock,
    transcript_updates_queue: queue.Queue,
) -> None:
    """Consume audio chunks, transcribe them, and update dashboard state.

    Notes:
    - Audio chunks are produced by `pipeline/audio.py::capture_audio`.
    - We load Whisper model once per dashboard session for efficiency.
    """
    while not stop_event.is_set() or not audio_pipeline.audio_queue.empty():
        try:
            item = audio_pipeline.audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        text = ""
        try:
            result = model.transcribe(item["audio"], fp16=False)
            text = str(result.get("text", "")).strip()
        except Exception:
            text = ""

        if text:
            nlp_result = nlp.analyze_text(text)
            with shared_state_lock:
                face_emotion = str(shared_state.get("latest_face_emotion", "unknown"))

            merged = {
                "text": text,
                "timestamp": time.time(),
                **nlp_result,
                "face_emotion": face_emotion,
                "face_confidence": 0.7 if face_emotion != "unknown" else 0.0,
            }
            tone_result = fusion.classify_tone(merged)

            transcript_updates_queue.put(
                {
                    "latest_transcript": text,
                    "latest_tone_label": tone_result["tone_label"],
                    "latest_confidence": float(tone_result["tone_confidence"]),
                    "history_row": {
                        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Transcript": text,
                        "Face Emotion": face_emotion,
                        "Tone Label": tone_result["tone_label"],
                        "Confidence": float(tone_result["tone_confidence"]),
                    },
                }
            )

        audio_pipeline.audio_queue.task_done()


def _live_video_consumer_worker(
    stop_event: threading.Event,
    video_queue: queue.Queue,
    shared_state: dict[str, Any],
    shared_state_lock: threading.Lock,
    emotion_updates_queue: queue.Queue,
) -> None:
    """Consume face-emotion predictions from the video queue and update state."""
    while not stop_event.is_set() or not video_queue.empty():
        try:
            item = video_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        latest_face_emotion = str(item.get("emotion", "unknown"))
        with shared_state_lock:
            shared_state["latest_face_emotion"] = latest_face_emotion
        emotion_updates_queue.put(latest_face_emotion)
        video_queue.task_done()


def _drain_live_updates() -> None:
    """Apply worker-thread updates onto Streamlit session state (main thread only)."""
    emotion_updates_queue: queue.Queue | None = st.session_state.emotion_updates_queue
    transcript_updates_queue: queue.Queue | None = st.session_state.transcript_updates_queue

    if emotion_updates_queue is not None:
        while True:
            try:
                st.session_state.latest_face_emotion = emotion_updates_queue.get_nowait()
                emotion_updates_queue.task_done()
            except queue.Empty:
                break

    if transcript_updates_queue is not None:
        while True:
            try:
                update = transcript_updates_queue.get_nowait()
            except queue.Empty:
                break

            st.session_state.latest_transcript = str(update.get("latest_transcript", ""))
            st.session_state.latest_tone_label = str(update.get("latest_tone_label", "Neutral / Sincere"))
            st.session_state.latest_confidence = float(update.get("latest_confidence", 0.0))
            history_row = update.get("history_row")
            if isinstance(history_row, dict):
                st.session_state.history.append(history_row)
            transcript_updates_queue.task_done()


# -----------------------------------------------------------------------------
# Live mode control helpers
# -----------------------------------------------------------------------------
def _start_live_mode() -> None:
    """Start all required live-mode threads (audio capture/transcribe + video)."""
    if st.session_state.live_running:
        return

    # Ensure previous stop flags are reset.
    audio_pipeline.stop_event.clear()
    stop_event = threading.Event()
    if st.session_state.live_whisper_model is None:
        st.session_state.live_whisper_model = whisper.load_model("base")

    video_queue: queue.Queue = queue.Queue()
    transcript_updates_queue: queue.Queue = queue.Queue()
    emotion_updates_queue: queue.Queue = queue.Queue()
    live_shared_state: dict[str, Any] = {"latest_face_emotion": "unknown"}
    live_shared_state_lock = threading.Lock()

    # Reset live display values for a fresh run.
    st.session_state.latest_transcript = "Listening..."
    st.session_state.latest_face_emotion = "unknown"
    st.session_state.latest_tone_label = "Neutral / Sincere"
    st.session_state.latest_confidence = 0.0

    # Build and start pipeline threads.
    audio_capture_thread = threading.Thread(
        target=audio_pipeline.capture_audio,
        name="LiveAudioCapture",
        daemon=True,
    )

    audio_transcribe_thread = threading.Thread(
        target=_live_transcription_worker,
        args=(
            audio_pipeline.stop_event,
            st.session_state.live_whisper_model,
            live_shared_state,
            live_shared_state_lock,
            transcript_updates_queue,
        ),
        name="LiveAudioTranscribe",
        daemon=True,
    )

    video_thread = threading.Thread(
        target=video_pipeline.run_video_loop,
        args=(video_queue, stop_event),
        kwargs={"show_preview": False},
        name="LiveVideoLoop",
        daemon=True,
    )

    video_consumer_thread = threading.Thread(
        target=_live_video_consumer_worker,
        args=(stop_event, video_queue, live_shared_state, live_shared_state_lock, emotion_updates_queue),
        name="LiveVideoConsumer",
        daemon=True,
    )

    audio_capture_thread.start()
    audio_transcribe_thread.start()
    video_thread.start()
    video_consumer_thread.start()

    # Persist references in session state for lifecycle management.
    st.session_state.stop_event = stop_event
    st.session_state.audio_capture_thread = audio_capture_thread
    st.session_state.audio_transcribe_thread = audio_transcribe_thread
    st.session_state.video_thread = video_thread
    st.session_state.video_consumer_thread = video_consumer_thread
    st.session_state.video_queue = video_queue
    st.session_state.transcript_updates_queue = transcript_updates_queue
    st.session_state.emotion_updates_queue = emotion_updates_queue
    st.session_state.live_shared_state = live_shared_state
    st.session_state.live_shared_state_lock = live_shared_state_lock
    st.session_state.live_running = True


def _stop_live_mode() -> None:
    """Signal all running live-mode threads to stop."""
    if not st.session_state.live_running:
        return

    # Signal both stop events (audio module event + local video event).
    audio_pipeline.stop_event.set()

    if st.session_state.stop_event is not None:
        st.session_state.stop_event.set()

    st.session_state.live_running = False


# -----------------------------------------------------------------------------
# File mode processing helpers
# -----------------------------------------------------------------------------
def _extract_streams(input_path: Path, work_dir: Path) -> tuple[Path, Path]:
    """Extract WAV audio and no-audio MP4 video using ffmpeg-python."""
    audio_out = work_dir / "extracted_audio.wav"
    video_out = work_dir / "extracted_video.mp4"

    (
        ffmpeg.input(str(input_path))
        .output(str(audio_out), ar=audio_pipeline.SAMPLE_RATE, ac=1)
        .overwrite_output()
        .run(quiet=True)
    )

    (
        ffmpeg.input(str(input_path))
        .output(str(video_out), an=None)
        .overwrite_output()
        .run(quiet=True)
    )

    return audio_out, video_out


def _process_audio_file(audio_path: Path) -> list[dict[str, Any]]:
    """Chunk extracted audio every ~5 seconds, transcribe, and run NLP analysis."""
    model = whisper.load_model("base")

    # Read mono WAV file into float32 in [-1, 1].
    pcm, _ = (
        ffmpeg.input(str(audio_path)).output("pipe:", format="f32le", ac=1, ar=audio_pipeline.SAMPLE_RATE).run(
            capture_stdout=True, capture_stderr=True, quiet=True
        )
    )
    audio_np = np.frombuffer(pcm, np.float32)
    sample_rate = audio_pipeline.SAMPLE_RATE

    window_size = int(sample_rate * audio_pipeline.RECORD_SECONDS)
    results: list[dict[str, Any]] = []

    for start in range(0, len(audio_np), window_size):
        chunk = audio_np[start : start + window_size]
        if chunk.size == 0:
            continue

        if np.abs(chunk).mean() < 0.01:
            continue

        timestamp = start / sample_rate
        text = model.transcribe(chunk, fp16=False).get("text", "").strip()
        if not text:
            continue

        results.append(
            {
                "text": text,
                "timestamp": float(timestamp),
                **nlp.analyze_text(text),
            }
        )

    return results


def _process_video_file(video_path: Path) -> list[dict[str, Any]]:
    """Sample one frame/second from extracted video and infer facial emotions."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_index = 0
    next_sample = 0
    results: list[dict[str, Any]] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index >= next_sample:
                next_sample += int(fps)
                timestamp = frame_index / fps

                try:
                    analysis = video_pipeline.DeepFace.analyze(
                        frame,
                        actions=["emotion"],
                        enforce_detection=True,
                    )
                    emotion, confidence = video_pipeline._parse_emotion_result(analysis)
                    results.append(
                        {
                            "emotion": emotion,
                            "confidence": confidence / 100.0,
                            "timestamp": float(timestamp),
                        }
                    )
                except Exception:
                    # If no face found in this sampled frame, skip and continue.
                    pass

            frame_index += 1
    finally:
        cap.release()

    return results


def _render_tone_badge(tone_label: str) -> None:
    """Render a large, color-coded tone label."""
    color = TONE_COLORS.get(tone_label, TONE_COLORS["Neutral / Sincere"])
    st.markdown(
        f"<h2 style='color:{color}; margin-top: 0.5rem;'>{tone_label}</h2>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Main app layout
# -----------------------------------------------------------------------------
def main() -> None:
    """Run Streamlit app."""
    st.set_page_config(page_title="Multimodal Tone Detector", layout="wide")
    _init_state()

    # Title required by the specification.
    st.title("Multimodal Tone Detector")

    # Sidebar mode selector required by the specification.
    mode = st.sidebar.radio("Select Mode", ["Live Webcam", "Upload Video File"])

    if mode == "Live Webcam":
        _drain_live_updates()
        st.subheader("Live Tone Monitoring")

        control_col1, control_col2 = st.columns(2)
        with control_col1:
            if st.button("Start", use_container_width=True):
                _start_live_mode()
        with control_col2:
            if st.button("Stop", use_container_width=True):
                _stop_live_mode()

        # Live display placeholders (st.empty) used for frequent updates.
        transcript_placeholder = st.empty()
        emotion_placeholder = st.empty()
        tone_placeholder = st.empty()
        confidence_placeholder = st.empty()

        transcript_placeholder.markdown(
            f"**Current Transcription (~5s):** {st.session_state.latest_transcript}"
        )
        emotion_placeholder.markdown(
            f"**Current Facial Emotion:** {st.session_state.latest_face_emotion}"
        )

        with tone_placeholder.container():
            st.markdown("**Tone Label**")
            _render_tone_badge(st.session_state.latest_tone_label)

        confidence_value = max(0.0, min(1.0, float(st.session_state.latest_confidence)))
        confidence_placeholder.markdown("**Confidence**")
        confidence_placeholder.progress(confidence_value)

        st.markdown("### History")
        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            height=280,
        )

        # Auto-refresh live UI while running.
        if st.session_state.live_running:
            time.sleep(LIVE_REFRESH_SECONDS)
            st.rerun()

    else:
        st.subheader("File-based Tone Analysis")

        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=["mp4", "mov", "avi"],
        )

        if st.button("Process", disabled=uploaded_file is None):
            if uploaded_file is None:
                st.warning("Please upload a file first.")
            else:
                with st.status("Processing uploaded file...", expanded=True) as status:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        tmp_path = Path(tmp_dir)
                        input_path = tmp_path / uploaded_file.name
                        input_path.write_bytes(uploaded_file.read())

                        st.write("1) Extracting audio/video streams with ffmpeg...")
                        audio_path, video_path = _extract_streams(input_path, tmp_path)

                        st.write("2) Running audio transcription + NLP...")
                        audio_results = _process_audio_file(audio_path)

                        st.write("3) Running video emotion inference...")
                        video_results = _process_video_file(video_path)

                        st.write("4) Synchronizing and fusing multimodal outputs...")
                        merged = sync.sync_segments(audio_results, video_results)
                        final_rows = [fusion.classify_tone(segment) for segment in merged]

                    status.update(label="Processing complete.", state="complete")

                if final_rows:
                    timeline_df = pd.DataFrame(
                        [
                            {
                                "Timestamp": f"{row.get('timestamp', 0.0):.2f}",
                                "Transcript": row.get("text", ""),
                                "Face Emotion": row.get("face_emotion", "unknown"),
                                "Tone Label": row.get("tone_label", "Neutral / Sincere"),
                                "Confidence": row.get("tone_confidence", 0.0),
                            }
                            for row in final_rows
                        ]
                    )
                    st.markdown("### Annotated Timeline")
                    st.dataframe(timeline_df, use_container_width=True)
                else:
                    st.info("No usable multimodal segments were produced from this file.")


if __name__ == "__main__":
    main()
