# Roadmap

## v0.1.0 — Released 2026-03-28
Initial release. Core CLI with 60+ metrics across 7 groups.

- Basic, loudness (EBU R128 / BS.1770), spectral, temporal, noise, speech, perceptual metric groups
- Console, JSON, and CSV output
- GPU acceleration (CUDA / Apple MPS)
- Multi-backend audio loading (WAV, FLAC, MP3, AAC, M4A, OGG, Opus, AIFF)
- Reference file support for intrusive metrics (PESQ, STOI, SI-SDR, SDR)
- Parallel batch analysis
- Auto-save `filename.json` alongside each analyzed file

---

## v0.2.0 — Prosody, Psychoacoustics & Speaker Analysis — Released 2026-03-28
Three new metric groups: 31 additional metrics.

- **prosody** group — per-frame F0 trajectory (autocorrelation), jitter, shimmer, tremor rate/depth, pitch variability CV, F0 slope, estimated speech rate
- **psychoacoustic** group — Vassilakis (2001) roughness, Sethares (1993) sensory dissonance, Zwicker spectral sharpness (acum), tonality, SFM, harmonicity
- **speaker** group — LPC formant tracking (F1–F4, order 12), spectral tilt, Cepstral Peak Prominence (CPP), breathiness index, creakiness/vocal fry ratio, heuristic gender estimation, heuristic age range
- Auto-save JSON behavior: `filename.wav` → `filename.json`
- GitHub URLs and version aligned to `0.1.0`
- LaTeX math formulas in README for all metric groups

---

## v0.3.0 — Speech Quality Metrics — Released 2026-03-29

- **Additional jitter measures** — RAP, PPQ5, DDP
- **Additional shimmer measures** — APQ3, APQ5, DDA
- **Noise-to-Harmonics Ratio (NHR)** — complement to HNR
- **Multi-cue speech detector** (`speech_detector.py`) — 7 weighted acoustic cues (F0, ZCR bimodality, syllabic modulation, spectral tilt, voicing continuity, speech-band energy, music discriminator); returns `content_type` and `speech_confidence`
- **Speech detection gate** — automatically skips prosody, speaker, and speech metric groups for non-speech content
- **Content-type classification** — each result tagged as `speech`, `music`, `noise`, `silence`, or `mixed`
- **Output schema versioning** — `schema_version` + `tool_version` emitted in every JSON and CSV output
- Consolidated `metrics_speech_quality.py` into existing metric modules

---

## v0.4.0 — Perceptual Quality Overhaul — Released 2026-03-30
Replace heuristic MOS proxies with model-based estimators.

- **DNSMOS integration** — Microsoft DNSMOS P.835 for non-intrusive speech quality (SIG, BAK, OVRL scores)
- **AECMOS support** — echo-aware quality scoring for call recordings
- **Improved P.563 proxy** — better single-ended quality estimation without a reference
- **Streaming normalization target advisor** — flag files against platform loudness targets (Spotify, Apple Music, YouTube, podcast standards)

---

## v0.4.1 — Speech Loudness & ASL — Released 2026-03-30
Follow-on patch release on the `v0.4` line.

- **Active Speech Level (ASL)** — P.56-inspired active speech RMS level in dBFS
- Tighten speech-level reporting for call-review and dialogue workflows

---

## v0.4.2 — Patch Continuation — Released 2026-03-30
Ongoing follow-on patch line on top of `v0.4.1`.

- Stabilize and polish the new speech-level / perceptual-quality reporting work
- Carry the `v0.4.x` line cleanly into the richer reporting work

---

## v0.5.0 — Richer Output & Reporting — Released 2026-04-02
Make results more useful for review and integration.

- **HTML report** — self-contained single-file report with waveform thumbnails, metric tables, and per-group pass/fail badges
- **Diff mode** — compare two analysis runs and surface regressions (`qualiax diff before.json after.json`)
- **Threshold rules** — define pass/fail rules via a config file; exit code reflects compliance
- **Markdown output format** — for embedding reports in docs or PR comments

---

## v0.5.1 — Diff Mode Kickoff — Released 2026-04-02
Follow-on patch release on top of the `v0.5` reporting line.

- **Diff mode groundwork** — compare analysis outputs and surface regressions between runs
- Finalize Markdown output and threshold-rule compliance on the `v0.5` line

---

## v0.6.0 — Streaming & Real-Time Analysis — Released 2026-04-02
Extend qualiax beyond static files.

- **stdin pipe support** — `ffmpeg -i stream.mp4 -f wav - | qualiax -`
- **Segment mode** — split long files into N-second chunks and report per-segment metrics
- **Live microphone capture mode** — record N-second microphone input and analyze it immediately
- **Watch mode** — monitor a directory for new files and analyze on arrival

---

## v0.6.1 — Segment Mode — Released 2026-04-02
Follow-on patch release on top of the `v0.6` streaming line.

- Ship fixed-length segment analysis via `--segment-seconds`
- Add per-segment metadata to JSON, CSV, HTML, and Markdown outputs
- Group segmented sidecars under the original source file

---

## v0.6.2 — Microphone & Watch Mode — Released 2026-04-02
Follow-on patch release on top of the completed `v0.6` streaming line.

- Add `--mic-seconds` and `--mic-sample-rate` for live microphone capture
- Add `--watch`, `--watch-interval`, and `--watch-limit` for arrival-based directory monitoring
- Keep aggregate output files updated as new watch-mode results arrive

---

## v0.7.0 — Python API — Released 2026-04-02
Make qualiax usable as a library, not just a CLI.

- **Public Python API** — `from qualiax import analyze; results = analyze("file.wav")`
- **Async support** — `await analyze_async(path)` for integration into async pipelines
- **Typed result objects** — full type annotations on all return values
- **Plugin interface** — register custom metric groups without forking the package

---

## v0.8.0 — ML-Based Metrics & Extended Format Support — Released 2026-04-02
Bring in learned quality models and wider codec coverage.

- **CREPE F0 estimation** — replace autocorrelation-based pitch with the CREPE neural model for significantly better accuracy on noisy speech
- **Learned MOS (UTMOS / SHEET)** — state-of-the-art neural MOS prediction
- **Codec artifact detection** — identify MP3/AAC compression artifacts, pre-echo, quantization noise
- **Multi-channel support** — per-channel analysis and inter-channel metrics (stereo width, phase correlation, ITD/ILD)
- **Video container support** — extract and analyze audio tracks directly from `.mp4`, `.mkv`, `.mov`

---

## v0.9.0 — Validation, Presets & Scorecards — Released 2026-04-03
Turn raw metrics into more trustworthy evaluation workflows.

- **Golden benchmark suite** — reference fixtures and regression checks for core metrics, reports, and model-backed outputs
- **Task presets** — built-in analysis/rule profiles for podcasts, call-center QA, speech enhancement, and music/mastering review via `--preset` and `analyze(..., preset=...)`
- **Aggregate scorecards** — batch-level summaries with percentiles, outlier lists, and per-metric rollups across large folders
- **Confidence & calibration notes** — clearer trust labeling for proxy, heuristic, and model-backed metrics in reports and APIs

---

## v0.9.1 — Packaging & Watch Hardening — Released 2026-04-03
Follow-on patch release on top of the completed `v0.9` line.

- **Stable full install** — `.[all]` now excludes the brittle CREPE dependency; install `crepe` manually only when you want the neural pitch backend
- **Event-driven watch mode** — uses `watchdog` when available instead of polling-only scans
- **Incremental JSONL watch output** — append one object per arrival with `.jsonl` / `.ndjson`
- **Expanded end-to-end coverage** — benchmark fixtures, scorecard tests, and real-result confidence checks

---

## v0.9.2 — Repo Hygiene & Installer Source-of-Truth — Released 2026-04-03
Follow-on patch release focused on packaging drift and local install safety.

- **Cleaner git hygiene** — ignore common report artifacts, coverage caches, and local output directories that should not live in the repo
- **No tracked egg-info drift** — remove stale `qualiax.egg-info` metadata from source control and rely on generated build metadata instead
- **Venv-first installer** — `install.sh` now creates or reuses a local virtualenv by default instead of mutating the global Python environment
- **No surprise sudo** — ffmpeg is now documented and detected, not auto-installed through system package managers
- **One install source of truth** — `install.sh` installs via the extras defined in `pyproject.toml` instead of hardcoding dependency lists

---

## v0.10.0 — API Stability, Async Flow & Richer Scorecards — Released 2026-04-11
Focused stabilization release driven by code-review findings in the public API, watch pipeline, and batch-summary layer.

- **Stable API return shape** — `analyze(...)` now always returns a list, while `analyze_one(...)` remains the exact-one helper
- **Real async orchestration** — `analyze_async(...)` now drives async per-file work instead of wrapping the entire sync call in a single background thread
- **Stricter library defaults** — Python API calls now default to `strict=True` so unexpected metric-group failures surface instead of degrading silently
- **Reprocessable watch inputs** — watch mode tracks processed file signatures so updated files can be analyzed again after they change
- **Richer scorecards** — categorical metrics and confidence-count distributions now appear in scorecard rollups alongside numeric summaries

---

## v0.11.0 — Contracts, Diagnostics & Ingest Robustness — Released 2026-05-10
Make qualiax easier to trust, automate, and run continuously in production-style workflows.

- **Formal output contracts** — publish JSON Schema definitions and built-in validation helpers for JSON / JSONL reports and scorecards
- **Structured diagnostics** — promote degraded execution details from free-form notes into machine-readable warnings, failure counters, and per-group health summaries
- **Hardened watch ingestion** — add debounce windows, retry/backoff, bounded queue policy, and explicit dropped-file reporting for long-running watch jobs
- **Rule/profile validation** — tighten presets and threshold-rule configs with stronger unmatched-name handling, dry-run linting, and clearer failure modes
- **Model/runtime provenance** — expose model asset versions, backend/runtime details, and calibration fingerprints in outputs for reproducibility
