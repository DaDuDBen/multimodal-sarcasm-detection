# Multimodal Tone Detector

Detects conversational tone (e.g., **Sarcastic**, **Sincere / Positive**, **Joking / Playful**) by combining:

- **Speech transcription** (Whisper)
- **Text understanding** (irony + emotion classifiers)
- **Facial emotion inference** (DeepFace + OpenCV)

The project includes:

- a **CLI** for live webcam/microphone mode and offline file processing
- a **Streamlit dashboard** for an interactive UI

---

## Project Structure

```text
multimodal-tone-detector/
├── main.py                  # CLI entry point
├── dashboard/
│   └── app.py               # Streamlit app
├── pipeline/
│   ├── audio.py             # Microphone capture + optional transcription worker
│   ├── nlp.py               # Irony/emotion text inference
│   ├── video.py             # Webcam video emotion inference
│   ├── sync.py              # Audio/video timestamp synchronization
│   └── fusion.py            # Rule-based multimodal tone classification
└── requirements.txt
```

---

## Requirements

- Python 3.10+ (recommended)
- `ffmpeg` installed and available on `PATH`
- Webcam + microphone for live mode

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## CLI Usage

From the repository root:

### 1) Live Mode (microphone + webcam)

```bash
python multimodal-tone-detector/main.py --mode live
```

What it does:
- Captures audio in ~5s windows.
- Samples webcam frames ~1 FPS.
- Runs Whisper + NLP + DeepFace.
- Prints fused tone predictions in real time.

Stop with `Ctrl+C`.

### 2) File Mode (offline video)

```bash
python multimodal-tone-detector/main.py --mode file --input /path/to/video.mp4
```

What it does:
- Extracts mono 16kHz audio with ffmpeg.
- Transcribes with Whisper.
- Runs text irony/emotion + facial emotion.
- Synchronizes modalities and prints an annotated timeline.

---

## Streamlit Dashboard

Run:

```bash
streamlit run multimodal-tone-detector/dashboard/app.py
```

Dashboard modes:

- **Live Webcam**: Start/Stop real-time monitoring and view rolling history.
- **Upload Video File**: Process an uploaded video and display a labeled timeline table.

---

## How Tone Is Decided

Final tone labels are produced in `pipeline/fusion.py` using deterministic rules over:

- NLP irony label/score
- NLP emotion label/score
- Dominant face emotion/confidence from synchronized video samples

Possible outputs:

- `Sarcastic`
- `Joking / Playful`
- `Sincere / Positive`
- `Neutral / Sincere`

---

## Troubleshooting

- **`ModuleNotFoundError` for dependencies**  
  Re-run `pip install -r requirements.txt`.

- **No audio/video in live mode**  
  Confirm microphone/camera permissions and that no other app is exclusively locking devices.

- **ffmpeg errors in file mode**  
  Ensure system ffmpeg is installed (`ffmpeg -version` should work in terminal).

- **Slow first run**  
  Whisper/transformer/deepface models may download on first execution.

---

## Notes

- Confidence values are heuristic and model-dependent.
- This project is a practical prototype and can be extended with learned multimodal fusion models, better diarization, and robust speaker/face tracking.
