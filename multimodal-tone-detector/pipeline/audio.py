"""Audio capture and transcription pipeline for multimodal tone detection.

This module records microphone audio continuously with PyAudio, filters out silent
segments, and transcribes valid chunks using a local openai-whisper model.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

import numpy as np
import pyaudio
import whisper

# Audio capture configuration required by the pipeline.
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
RECORD_SECONDS = 5

# Shared queue used by producer (recording thread) and consumer (transcription thread).
audio_queue: queue.Queue[dict[str, Any]] = queue.Queue()

# Event used to request a clean shutdown across all threads.
stop_event = threading.Event()


def capture_audio() -> None:
    """Continuously capture microphone audio and push valid chunks into `audio_queue`."""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    # Track elapsed time so each chunk can be tagged with seconds since recording started.
    start_time = time.time()
    chunks_per_window = int((SAMPLE_RATE * RECORD_SECONDS) / CHUNK_SIZE)

    print("[Audio] Recording started...")

    try:
        while not stop_event.is_set():
            frames: list[bytes] = []

            # Collect enough CHUNK_SIZE reads to build one 5-second window.
            for _ in range(chunks_per_window):
                if stop_event.is_set():
                    break
                frames.append(stream.read(CHUNK_SIZE, exception_on_overflow=False))

            if not frames:
                continue

            # Combine raw bytes and convert int16 PCM to float32 in the range [-1, 1].
            raw_audio = b"".join(frames)
            audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

            # Skip chunks that are effectively silent based on mean absolute amplitude.
            if np.abs(audio_np).mean() < 0.01:
                continue

            timestamp = time.time() - start_time
            audio_queue.put({"audio": audio_np, "timestamp": timestamp})
    finally:
        # Always release audio resources, even on Ctrl+C or runtime exceptions.
        stream.stop_stream()
        stream.close()
        pa.terminate()
        print("[Audio] Recording stopped.")


def transcribe_audio() -> None:
    """Read audio chunks from `audio_queue` and print Whisper transcriptions."""
    print("[Whisper] Loading model 'base'...")
    model = whisper.load_model("base")
    print("[Whisper] Model loaded. Transcription thread is running.")

    # Keep processing until stop is requested and the queue has been fully drained.
    while not stop_event.is_set() or not audio_queue.empty():
        try:
            item = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        audio_np = item["audio"]
        timestamp = item["timestamp"]

        # Whisper transcribes numpy arrays directly when audio is mono 16kHz float32.
        result = model.transcribe(audio_np, fp16=False)
        text = result.get("text", "").strip()

        if text:
            print(f"[{timestamp:.2f}s] {text}")

        audio_queue.task_done()

    print("[Whisper] Transcription stopped.")


def main() -> None:
    """Start recording/transcription threads and stop on Enter or KeyboardInterrupt."""
    recorder_thread = threading.Thread(target=capture_audio, name="AudioCapture", daemon=True)
    transcriber_thread = threading.Thread(target=transcribe_audio, name="AudioTranscriber", daemon=True)

    recorder_thread.start()
    transcriber_thread.start()

    print("Press Enter to stop recording and transcription.")

    try:
        input()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Stopping...")
    finally:
        stop_event.set()
        recorder_thread.join()
        transcriber_thread.join()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
