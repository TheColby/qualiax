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

---

## v0.12.0 — Calibration Lab — Released 2026-09-18

- Per-label precision, recall, F1, support, and macro calibration reports
- Confidence intervals for benchmark samples
- Cross-codec and cross-variant consistency checks with explicit tolerances

---

## v0.13.0 — Dataset Intelligence — Released 2026-09-18

- Near-duplicate grouping from quality fingerprints
- Train/validation/test leakage detection
- Speaker, content-type, and defect-label distribution summaries
- Label-mismatch risks and prioritized remediation plans

---

## v0.14.0 — Operational Monitoring — Released 2026-09-18

- Persistent bounded drift history and rolling metric windows
- Cooldown-based alert suppression and JSON webhook delivery
- Prometheus text exposition for numeric quality metrics

---

## v0.15.0 — Safe Repair — Released 2026-09-18

- Defect-aware ffmpeg repair plans for noise, clipping, and loudness issues
- Source-overwrite protection and dry-run-first execution
- Before/after metric evaluation with explicit improvements and regressions

---

## v0.16.0 — Plugin Ecosystem — Released 2026-09-18

- Versioned plugin API for custom label providers and report renderers
- Python entry-point discovery
- Failure isolation and a built-in conformance report

---

## v0.17.0 — Composite Quality Control — Released 2026-09-18

- Quality fingerprints, defect labels, repair suggestions, drift checks, and CI gates
- Enrichment for existing reports plus compact pipeline summaries
- Flagged-segment snippet export and insight contract validation

---

## v0.18.0 — Review Workbench — Released 2026-09-18

- Persistent per-file and per-segment reviewer annotations
- Accept, reject, and needs-review decisions with reviewer attribution
- Portable JSON review-state export for downstream workflows

---

## v0.19.0 — Scale & Performance — Released 2026-09-18

- File-backed analysis cache with source-signature invalidation
- Incremental source selection and bounded audio chunk iteration
- Performance-regression budgets and executor-compatible distributed adapters

---

## v0.20.0 — Compatibility Hardening — Released 2026-09-18

- Report migration registry and current-schema normalization
- Stable automation exit-code enum and structured deprecation warnings
- Model-asset checksum locks and verification

---

## v1.0.0 — Stable Quality Platform — Released 2026-09-18

- Stable top-level Python API for the complete quality workflow
- Release-readiness gates, dependency inventory, and reproducibility manifests
- Python 3.9-3.12 support matrix and production/stable package metadata

---

## v1.1.0 — First PyPI Release — Implemented, unreleased

- **Post-1.0 hardening** ([#1](https://github.com/TheColby/qualiax/pull/1)) — BS.1770 loudness correction (double −0.691 dB offset, stereo channel summing), known-answer metric tests, roadmap-module gaps closed, and `--insights` verified end to end on a defect corpus
- **Version bump to 1.1.0** — behavior changes and new public exports since 1.0.0
- **First PyPI release** — `pip install qualiax` via Trusted Publishing (`release.yml`)
- **Docs catch-up** — document the post-1.0 behavior changes; move README math to fenced `math` blocks so equations don't render as raw TeX on PyPI

---

## v1.2.0 — Consistency & Open Decisions — Implemented, unreleased

- **Exit codes match `ExitCode`** — usage errors, undecodable audio and invalid rule configs return `INVALID_INPUT` (1) instead of sharing `QUALITY_GATE_FAILED` (2); crashes after the audio loads return `ANALYSIS_FAILED` (3), and `--strict` failures no longer escape as a traceback; `--validate-output` failures return `CONTRACT_VIOLATION` (4); watch mode reports the most severe batch outcome
- **Cepstral Peak Prominence in true dB** — dB power cepstrum of the dB power spectrum, analysed at 16 kHz; the breathiness index is anchored on measured clean speech (VoiceBank-DEMAND median CPP 19.5 dB) instead of reading ≈0.95 for every input
- **P.563 proxy retune** — the speech band starts at 80 Hz, so clean low-pitched speech is no longer scored at the MOS floor (sample.wav 1.0 → 3.95); sub-F0 rumble is still penalized
- **Plugins compose with `--insights`** — `enrich_results(plugins=...)`, `analyze(plugins=...)` and the opt-in `--plugins` flag run label providers so their labels feed repair suggestions, CI gates, and triage; plugin labels survive re-enrichment; `--metrics` accepts plugin-registered groups
- **Analysis cache wired in** — `analyze(cache=...)` and `--cache DIR` reuse results for unchanged files, keyed by file signature, analysis options, and qualiax version; `FileResult.from_dict()` round-trips results exactly
- **Cut candidates resolved** — `LocalExecutorAdapter`, `RollingMetricWindow`, and `qualiax.DistributedAdapter` are deprecated for removal in 2.0.0; `release_readiness` stays

---

## v1.3.0 — Proxy Validation & Model Assets — Implemented, unreleased

- **Official DNSMOS models** — a numpy port of Microsoft's `dnsmos_local.py` runs the real P.835 (SIG, BAK, OVRL) and P.808 models; scores match the reference exactly on 16 kHz input. The old "model" paths looked for files Microsoft never shipped and could not have worked
- **Model downloads** — `qualiax models download|verify|list` fetches the models pinned to a DNS-Challenge commit by size and SHA-256 into `$QUALIAX_MODEL_DIR` or `~/.cache/qualiax/models`; mismatched files are refused, and provenance records which verified weights produced each result (output schema 3.6)
- **Proxy benchmarks** — `benchmarks/proxy_benchmark.py` measures every MOS proxy against the official models on the VoiceBank-DEMAND test set (1,648 clips) with utterance-level bootstrap intervals; results in `docs/proxy-benchmark.md`. No proxy reaches r ≥ 0.7 at the lower 95% bound; the best is Estimated MOS (r 0.71, MAE 0.42), and the BAK proxy does not track the model at all (r 0.16)
- **Evidence-based trust labels** — each proxy's calibration note quotes its measured agreement, and a proxy is labelled `proxy` only when its correlation clears 0.7 at the lower bound, otherwise `heuristic`; a test fails if a proxy changes without re-running the benchmark
- **AECMOS scope** — Microsoft's AECMOS needs far-end and microphone signals, so the single-recording echo proxy is labelled `heuristic` and its fake model path is removed (as are the unverifiable UTMOS/SHEET file hooks)

---

## v1.3.1 — Streamline & PyPI Release — Implemented, unreleased

- **First PyPI release** — `pip install qualiax`; the release workflow now smoke-tests the installed wheel from outside the checkout (it previously imported the source tree), checks that the proxy calibration data ships, and creates a GitHub release after publishing
- **`python -m qualiax`** — runs the CLI like the `qualiax` command
- **Insights baselines and drift** — loudness, pitch, brightness, and zero-crossing rate regress when they move either way from the baseline (they were treated as higher-is-better); per-feature tolerances widen one-file baselines and replace the 1 Hz drift threshold for Hz-scale features; regressions rank by distance outside the band in tolerance units instead of raw mixed-unit deltas
- **CI gates** — a true peak between −1 and 0 dBTP is now a warning; only overs above 0 dBTP block `--ci`
- **Watch mode drift** — files analyzed in earlier batches keep their drift result instead of being recompared against later state
- **Echo proxy** — no longer warns on most clean speech (it averages 3.5 on echo-free recordings); the measured behaviour is in its calibration note
- **Cleanup** — README quick start for PyPI users, an options reference generated from `--help` and checked by a test, tests moved off Click's deprecated `isolated_filesystem`, and the obsolete root `test_synthetic.py` removed

---

## v1.4.0 — Web UI — Planned

- **Browser analysis** — drag-and-drop a file and get the HTML report, no local Python install
- **Hosted demo** — a public instance for trying qualiax before installing it
