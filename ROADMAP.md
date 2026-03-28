# Roadmap

## v0.1.0 — Released 2026-03-28
Initial release. Core CLI with 60+ metrics across 7 groups.

- Basic, loudness (EBU R128 / BS.1770), spectral, temporal, noise, speech, perceptual metric groups
- Console, JSON, and CSV output
- GPU acceleration (CUDA / Apple MPS)
- Multi-backend audio loading (WAV, FLAC, MP3, AAC, M4A, OGG, Opus, AIFF)
- Reference file support for intrusive metrics (PESQ, STOI, SI-SDR, SDR)
- Parallel batch analysis

---

## v0.2.0 — Speech Quality Metrics
Wire up the existing `metrics_speech_quality.py` and `speech_detector.py` modules that were written but left disconnected in v0.1.0.

- **Jitter metrics** — local jitter, RAP, PPQ5, DDP (pitch period perturbation)
- **Shimmer metrics** — local shimmer, APQ3, APQ5, DDA (amplitude perturbation)
- **Noise-to-Harmonics Ratio (NHR)** — complement to HNR
- **Speech detection gate** — automatically skip voice quality metrics on non-speech content (music, silence, noise)
- **Content-type classification** — tag each file as `speech`, `music`, `noise`, or `silence` in output

---

## v0.3.0 — Perceptual Quality Overhaul
Replace heuristic MOS proxies with model-based estimators.

- **DNSMOS integration** — Microsoft DNSMOS P.835 for non-intrusive speech quality (SIG, BAK, OVRL scores)
- **AECMOS support** — echo-aware quality scoring for call recordings
- **Improved P.563 proxy** — better single-ended quality estimation without a reference
- **Streaming normalization target advisor** — flag files against platform loudness targets (Spotify, Apple Music, YouTube, podcast standards)

---

## v0.4.0 — Richer Output & Reporting
Make results more useful for review and integration.

- **HTML report** — self-contained single-file report with waveform thumbnails, metric tables, and per-group pass/fail badges
- **Diff mode** — compare two analysis runs and surface regressions (`qualiax diff before.json after.json`)
- **Threshold rules** — define pass/fail rules via a config file; exit code reflects compliance
- **Markdown output format** — for embedding reports in docs or PR comments

---

## v0.5.0 — Streaming & Real-Time Analysis
Extend qualiax beyond static files.

- **stdin pipe support** — `ffmpeg -i stream.mp4 -f wav - | qualiax -`
- **Live microphone mode** — analyze microphone input in a rolling window
- **Segment mode** — split long files into N-second chunks and report per-segment metrics
- **Watch mode** — monitor a directory for new files and analyze on arrival

---

## v0.6.0 — Python API
Make qualiax usable as a library, not just a CLI.

- **Public Python API** — `from qualiax import analyze; result = analyze("file.wav")`
- **Async support** — `await analyze_async(path)` for integration into async pipelines
- **Typed result objects** — full type annotations on all return values
- **Plugin interface** — register custom metric groups without forking the package

---

## v0.7.0 — ML-Based Metrics & Extended Format Support
Bring in learned quality models and wider codec coverage.

- **CREPE F0 estimation** — replace autocorrelation-based pitch with the CREPE neural model for significantly better accuracy on noisy speech
- **Learned MOS (UTMOS / SHEET)** — state-of-the-art neural MOS prediction
- **Codec artifact detection** — identify MP3/AAC compression artifacts, pre-echo, quantization noise
- **Multi-channel support** — per-channel analysis and inter-channel metrics (stereo width, phase correlation, ITD/ILD)
- **Video container support** — extract and analyze audio tracks directly from `.mp4`, `.mkv`, `.mov`
