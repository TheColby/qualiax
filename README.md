# qualiax

Perceptual speech and audio quality analyzer — command-line tool that computes 100+ metrics across 10 analysis groups. Zero required configuration. Point it at a file or folder and get a full quality report.

```
qualiax recording.wav
qualiax ./calls/ --format json --output results.json
qualiax interview.wav --reference clean.wav
ffmpeg -i stream.mp4 -f wav - | qualiax -
qualiax --mic-seconds 5 --output mic.json --silent
qualiax long_call.wav --segment-seconds 30 --output segments.json
qualiax ./incoming --watch --watch-limit 3 --output arrivals.json --silent
qualiax episode.wav --preset podcast --output report.json --silent
qualiax ./calls/ --output results.json --scorecard scorecard.md --silent
qualiax diff before.json after.json --format markdown
```

A sample file is included to try immediately:

```bash
qualiax sample.wav
```

> "The birch canoe slid on the smooth planks." — Harvard Sentence List 1, #1

To save a JSON sidecar next to the file:

```bash
qualiax sample.wav --save-sidecar
```

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Quick Install](#quick-install)
  - [Install Script](#install-script)
  - [Optional Dependencies](#optional-dependencies)
- [Usage](#usage)
  - [Basic](#basic)
  - [Directories](#directories)
  - [Output Formats](#output-formats)
  - [Diff Mode](#diff-mode)
  - [Threshold Rules](#threshold-rules)
  - [Task Presets](#task-presets)
  - [Stdin Pipes](#stdin-pipes)
  - [Microphone Mode](#microphone-mode)
  - [Segment Mode](#segment-mode)
  - [Watch Mode](#watch-mode)
  - [Reference Files](#reference-files)
  - [Filtering Metric Groups](#filtering-metric-groups)
  - [Parallel Processing](#parallel-processing)
  - [All Options](#all-options)
- [Python API](#python-api)
  - [Sync API](#sync-api)
  - [Async API](#async-api)
  - [Plugin Interface](#plugin-interface)
- [Metric Groups](#metric-groups)
  - [basic](#basic-1)
  - [loudness](#loudness)
  - [spectral](#spectral)
  - [temporal](#temporal)
  - [noise](#noise)
  - [speech](#speech)
  - [perceptual](#perceptual)
  - [prosody](#prosody)
  - [psychoacoustic](#psychoacoustic)
  - [speaker](#speaker)
- [Output](#output)
  - [Pretty (Console)](#pretty-console)
  - [JSON](#json)
  - [CSV](#csv)
  - [HTML](#html)
  - [Markdown](#markdown)
- [Hardware Acceleration](#hardware-acceleration)
- [Supported Formats](#supported-formats)
- [Requirements](#requirements)
- [References](#references)
- [License](#license)

---

## Features

- **100+ metrics** organized into 10 groups: basic signal properties, loudness (EBU R128 / BS.1770), spectral analysis, temporal structure, noise, speech characteristics, perceptual quality, prosody (F0 trajectory, jitter, shimmer, tremor), psychoacoustic (roughness, dissonance, sharpness, harmonicity), and speaker characteristics (formants, voice quality, and opt-in demographic heuristics)
- **Zero config** — works out of the box on any WAV file; print results to the terminal or opt into `--save-sidecar`
- **Intrusive and non-intrusive modes** — run standalone or supply a reference file for difference-based metrics (PESQ, STOI, SI-SDR, SDR)
- **Multiple output formats** — colored console, JSON, CSV, self-contained HTML, and Markdown reports
- **Batch scorecards** — write aggregate rollups with percentiles and outlier files via `--scorecard`
- **Diff mode** — compare two JSON exports and surface regressions with `qualiax diff before.json after.json`
- **Threshold rules** — apply metric compliance rules from JSON or TOML and return exit code `2` on violations
- **Rule/preset linting** — dry-run config validation with `--lint-rules` catches bad groups and contradictory thresholds before a run
- **Task presets** — built-in analysis profiles for podcasts, call-center QA, speech enhancement, and music mastering
- **Confidence & calibration notes** — proxy, heuristic, and model-backed caveats now surface in reports and JSON
- **Structured diagnostics & provenance** — machine-readable diagnostics, per-group health, backend/runtime details, and model asset fingerprints are emitted in JSON outputs
- **Built-in output contracts** — JSON Schema documents plus validation helpers for reports, JSONL streams, and scorecards
- **Public Python API** — call `analyze(...)`, `analyze_one(...)`, `analyze_many(...)`, or `await analyze_async(...)` and get typed `FileResult` / `MetricResult` objects back
- **Plugin interface** — register custom metric groups without forking the built-in registry
- **Optional neural metrics** — use CREPE for F0 and ONNX-backed learned MOS models when available, with proxy fallbacks when they are not
- **Color progress bar** — live `[████░░░░] N/total filename` bar during batch analysis
- **GPU acceleration** — automatically uses CUDA (NVIDIA) or MPS (Apple Silicon) when PyTorch is installed, falls back to CPU silently
- **Recursive directory analysis** — point at a folder to analyze every audio file in the tree
- **stdin pipe support** — analyze piped WAV data with `qualiax -`
- **Microphone mode** — capture live microphone audio for a fixed duration with `--mic-seconds`
- **Segment mode** — split long files into fixed-length chunks and report per-segment metrics with explicit segment metadata
- **Watch mode** — monitor directories for newly arrived files and analyze them automatically
- **Parallel workers** — thread-based parallelism for batch jobs
- **Robust audio loading** — multi-backend fallback chain handles WAV, FLAC, MP3, AAC, M4A, OGG, Opus, AIFF, and audio tracks inside `.mp4`, `.mkv`, and `.mov`

---

## Installation

### Quick Install

From the project root:

```bash
# Minimal — WAV support only
pip install -e .

# Recommended — adds MP3, FLAC, OGG, AAC support
pip install -e ".[audio]"

# Full — stable extras for formats, reports, learned-MOS hooks, watch mode, and microphone capture
pip install -e ".[all]"
```

After installation, `qualiax` is available in the environment you installed it into.

### Install Script

```bash
./install.sh
```

The install script is a thin wrapper around the extras defined in [pyproject.toml](/Users/cleider/dev/qualiax/pyproject.toml). It creates or reuses a local `.venv` by default, installs `qualiax` from the current checkout, and avoids automatic `sudo` or package-manager changes.

Make it executable first if needed:

```bash
chmod +x install.sh && ./install.sh
```

Common variants:

```bash
./install.sh                  # default: local .venv + .[all]
./install.sh --minimal        # base package only
./install.sh --audio --watch  # choose specific extras from pyproject.toml
./install.sh --with-torch     # also install torch + torchaudio
./install.sh --with-crepe     # manually opt into CREPE
./install.sh --global         # explicit opt-in to install into the current env
```

The script detects `ffmpeg` and tells you how to install it if needed, but it does not modify your system package manager for you.

### Optional Dependencies

| Extra | Packages | What it enables |
|-------|----------|-----------------|
| `audio` | `soundfile`, `pydub` | FLAC, OGG, AIFF, MP3, AAC, M4A, and video-container audio via ffmpeg |
| `perceptual` | `pesq`, `pystoi` | True PESQ P.862 scores, true STOI intelligibility |
| `live` | `sounddevice` | Live microphone capture mode (`--mic-seconds`) |
| `watch` | `watchdog` | Event-driven watch mode instead of polling-only directory scans |
| `ml` | `onnxruntime` | Learned MOS model hooks (UTMOS/SHEET) |
| `all` | all of the above + `librosa`, `sounddevice`, `watchdog`, `onnxruntime` | Stable full install for format support, microphone capture, event-driven watch mode, and ML hooks |

**ffmpeg** is required for MP3 and AAC/M4A decoding via pydub:

```bash
brew install ffmpeg        # macOS
apt install ffmpeg         # Ubuntu/Debian
```

**PyTorch** is optional but enables GPU acceleration on CUDA and Apple Silicon:

```bash
pip install torch
```

**sounddevice** is optional and enables microphone capture mode:

```bash
pip install sounddevice
```

**onnxruntime** is optional and enables learned-MOS inference:

```bash
pip install onnxruntime
```

**CREPE** remains supported, but its upstream packaging is brittle on some environments. Install it manually only if you specifically want the neural F0 backend:

```bash
pip install crepe
```

---

## Usage

### Basic

Analyze a single file. Results are printed to the terminal:

```bash
qualiax recording.wav
```

### Directories

Recursively analyze every supported audio file in a directory:

```bash
qualiax ./recordings/
```

Multiple paths can be provided:

```bash
qualiax file1.wav file2.mp3 ./folder/
```

### Output Formats

Write all results to a specific file:

```bash
qualiax ./calls/ --output results.json
qualiax ./calls/ --output results.csv
qualiax ./calls/ --output report.html
qualiax ./calls/ --output report.md
qualiax ./incoming --watch --output arrivals.jsonl --silent
qualiax recording.wav --format json     # explicit format, print to stdout
```

Suppress console output when writing to a file:

```bash
qualiax ./calls/ --output results.json --silent
qualiax ./calls/ --output results.json --scorecard scorecard.md --silent
```

Write per-file JSON sidecars next to the analyzed sources:

```bash
qualiax ./calls/ --save-sidecar
```

Existing output files are protected by default. Use `--force` to overwrite an existing aggregate report or sidecar JSON file.

`jsonl` / `ndjson` outputs are also supported. They are especially useful with `--watch`, because qualiax can append one JSON object per newly arrived file instead of rewriting a large aggregate report every cycle.

### Diff Mode

Compare two previously generated JSON reports and surface regressions, improvements, and added or removed files:

```bash
qualiax diff before.json after.json
qualiax diff before.json after.json --output diff.md --silent
```

Diff mode supports `pretty`, `json`, and `markdown` output, and it can compare both aggregate `.json` exports and line-oriented `.jsonl` / `.ndjson` reports.

### Threshold Rules

Apply pass/fail rules from a JSON or TOML config file while analyzing audio:

```bash
qualiax recording.wav --rules rules.json --output report.json --silent
```

Example `rules.json`:

```json
{
  "rules": [
    {
      "metric": "True Peak",
      "group": "loudness",
      "max": -1.0,
      "message": "Streaming ceiling exceeded"
    }
  ]
}
```

Rule violations are attached to the affected metric as warnings, summarized in `notes`, and cause exit code `2` when the analysis itself succeeds. Matching is case-insensitive, and explicit rule-file metric names that never match any analyzed metric are surfaced in `notes` instead of disappearing silently.

Use `--lint-rules` when you want to validate a preset or rules file without analyzing audio:

```bash
qualiax --rules rules.json --lint-rules
qualiax --preset podcast --lint-rules
```

Lint mode reports duplicate rules, invalid groups, and contradictory thresholds like `min > max`. During normal analysis, explicit `--rules` files now treat unmatched metric names as compliance violations instead of note-only drift.

### Task Presets

Use a built-in preset when you want sensible default metric groups plus threshold rules for a common workflow:

```bash
qualiax episode.wav --preset podcast --output report.json --silent
qualiax contact_center.wav --preset call-center-qa
qualiax enhanced.wav --preset speech-enhancement
qualiax master.wav --preset music-mastering
```

Available presets:

- `podcast`
- `call-center-qa`
- `speech-enhancement`
- `music-mastering`

Presets supply default metric groups and built-in threshold rules. If you also pass `--metrics`, the explicit metric selection wins, and the preset's rules are still applied to any matching metrics that are present.

### Stdin Pipes

Analyze piped WAV audio by passing `-` as the input path:

```bash
ffmpeg -i stream.mp4 -f wav - | qualiax -
ffmpeg -i stream.mp4 -f wav - | qualiax - --output report.md --format markdown
```

### Microphone Mode

Capture a short microphone recording and analyze it immediately:

```bash
qualiax --mic-seconds 5
qualiax --mic-seconds 10 --mic-sample-rate 16000 --output mic.json --silent
```

Microphone mode requires the optional `sounddevice` package. Captured results are labeled as `microphone` in the output, and `--save-sidecar` is intentionally disabled for this mode because there is no stable source file path to write next to.

### Segment Mode

Split long recordings into fixed-length chunks and analyze each segment independently:

```bash
qualiax long_call.wav --segment-seconds 30 --output segments.json
qualiax ./calls/ --segment-seconds 15 --output report.md --format markdown
```

Segmented output includes `source_file`, `segment_index`, `total_segments`, `segment_start_s`, and `segment_end_s`. When `--save-sidecar` is used with segment mode, qualiax writes one JSON sidecar per original source file containing all of that file's segment results.

### Watch Mode

Monitor one or more directories for newly arrived audio files and analyze them as they appear:

```bash
qualiax ./incoming --watch
qualiax ./incoming --watch --watch-limit 5 --output arrivals.json --silent
qualiax ./incoming --watch --watch-debounce 0.5 --watch-retries 3 --watch-backoff 1.5
```

Watch mode skips files that already exist when the command starts and only processes newly discovered supported audio files. Newly arrived files are held until they appear stable on disk, which reduces false starts on partially written recordings. Qualiax now tracks processed file signatures instead of just paths, so a file that changes after its first analysis can be picked up again on a later pass. Watch mode also supports debounce windows, bounded pending queues, and retry/backoff for failed analyses before a file is explicitly dropped. When `--output` is used, qualiax rewrites the aggregate report as new results arrive.

If `watchdog` is installed via `pip install -e ".[watch]"` or `pip install watchdog`, qualiax uses filesystem events instead of pure polling. With `--output arrivals.jsonl`, watch mode appends incrementally.

### Reference Files

Supply a clean reference recording to enable intrusive metrics (PESQ, STOI, SI-SDR, SDR, log-spectral distance, spectral correlation, cepstral distance):

```bash
qualiax degraded.wav --reference clean.wav
```

Without a reference, perceptual metrics fall back to non-intrusive proxies.

When `--workers` is greater than 1, the reference file is loaded once and shared across worker threads.

### Filtering Metric Groups

```bash
qualiax recording.wav --metrics basic
qualiax recording.wav --metrics basic,loudness,spectral
qualiax recording.wav --metrics all      # default
```

Available groups: `basic`, `loudness`, `spectral`, `temporal`, `noise`, `speech`, `perceptual`, `prosody`, `psychoacoustic`, `speaker`

`--metrics` is validated strictly. Unknown group names now fail with a CLI error instead of being ignored, so typos like `--metrics speach` are surfaced immediately.

### Parallel Processing

```bash
qualiax ./calls/ --workers 4
```

### All Options

```
Usage: qualiax [OPTIONS] PATHS...

Options:
  --silent                        Suppress console output (useful with --output)
  --save-sidecar                  Write per-file JSON sidecars next to sources
  --force                         Allow overwriting existing output files
  -o, --output PATH               Write results to file (.json, .jsonl, .csv, .html, .md)
  -f, --format [pretty|json|jsonl|csv|html|markdown]
                                  Output format (default: pretty)
  -r, --reference PATH            Reference audio for intrusive metrics
  -m, --metrics TEXT              Comma-separated metric groups (default: all)
  --no-color                      Disable colored output
  -v, --verbose                   Show analysis progress and warnings
  -w, --workers INTEGER           Parallel worker threads (default: 1)
  --preset [podcast|call-center-qa|speech-enhancement|music-mastering]
                                  Built-in analysis preset
  --scorecard PATH                Write an aggregate scorecard (.json, .html, .md)
  --rules PATH                    Threshold rules config (.json or .toml)
  --lint-rules                    Validate preset/rule configuration and exit
  --segment-seconds FLOAT         Split each input into fixed-length segments
  --mic-seconds FLOAT             Capture microphone input for N seconds
  --mic-sample-rate INTEGER       Microphone capture sample rate
  --watch                         Watch directories for new audio files
  --watch-interval FLOAT          Polling interval for --watch
  --watch-limit INTEGER           Stop --watch after N new files
  --watch-debounce FLOAT          Stable time before watch analysis
  --watch-retries INTEGER         Retry count for watch failures
  --watch-backoff FLOAT           Base backoff between watch retries
  --watch-max-pending INTEGER     Max pending watch candidates
  --validate-output               Validate JSON outputs against built-in schema
  --strict                        Fail on unexpected metric-group failures
  --include-demographics          Include heuristic speaker age/gender outputs
  --help                          Show this message and exit
```

---

## Python API

The package can now be used directly from Python without shelling out to the CLI.

### Sync API

Analyze one file and get back a list of typed [FileResult](/Users/cleider/dev/qualiax/qualiax/models.py) objects:

```python
from qualiax import analyze

results = analyze("recording.wav", metrics=["basic", "loudness"])
print(results[0].sample_rate)
print(results[0].metrics[0].name, results[0].metrics[0].value)
```

`analyze(...)` now always returns a list. Use `analyze_one(...)` when you want an exactly-one contract:

```python
from qualiax import analyze_many, analyze_one

one = analyze_one("recording.wav", metrics=["basic"])
many = analyze_many(["call_a.wav", "call_b.wav"], segment_seconds=30)
```

The Python API now defaults to `strict=True`, so unexpected metric-group failures raise instead of quietly degrading. Pass `strict=False` if you want best-effort library behavior.

Analyze multiple paths and get a list of results:

```python
from qualiax import analyze

results = analyze(["call_a.wav", "call_b.wav"], segment_seconds=30)
for item in results:
    print(item.path, item.duration_s)
```

Use a built-in preset from Python the same way:

```python
from qualiax import analyze

results = analyze("episode.wav", preset="podcast")
print(results[0].notes)
```

Build an aggregate scorecard from API results:

```python
from qualiax import analyze, build_scorecard, render_scorecard

results = analyze(["call_a.wav", "call_b.wav"], preset="call-center-qa")
scorecard = build_scorecard(results)
print(render_scorecard(scorecard, "markdown"))
```

### Async API

Use the async API when you want to integrate with an existing event loop. It now schedules per-file analysis asynchronously instead of wrapping the whole sync call in one background thread:

```python
import asyncio
from qualiax import analyze_async

async def main():
    results = await analyze_async("recording.wav", metrics="speech")
    print(results[0].content_type, results[0].speech_confidence)

asyncio.run(main())
```

### Plugin Interface

Register a custom metric group and then request it through the same analyzer pipeline:

```python
from qualiax import MetricResult, analyze, register_metric_group, unregister_metric_group

def compute_custom(audio, sr, ref_audio=None, ref_sr=None):
    return [MetricResult(name="Custom Score", value=7, group="custom")]

register_metric_group("custom", compute_custom)
try:
    results = analyze("recording.wav", metrics=["custom"])
    print(results[0].metrics[0].name, results[0].metrics[0].value)
finally:
    unregister_metric_group("custom")
```

`available_metric_groups()` returns the active registry, including any custom groups registered at runtime. The API validates metric-group names strictly, just like the CLI.

`available_presets()` returns the built-in preset names, and `get_preset(name)` returns the full preset definition.

For downstream automation, qualiax now also exports JSON Schema and validation helpers:

```python
from qualiax import get_json_schema, validate_report_payload

schema = get_json_schema("report")
issues = validate_report_payload(results_json_payload)
assert not issues
```

---

## Metric Groups

---

### basic

Core signal properties derived directly from the waveform. Let $x[n]$ denote the discrete audio sample at index $n$, and $N$ the total number of samples.

$$x_{rms} = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} x[n]^2}$$

This equation computes the root-mean-square amplitude, which summarizes the typical signal magnitude over the full clip.

Where: $x_{rms}$ is RMS amplitude, $x[n]$ is the sample value at index $n$, and $N$ is the total number of samples.

$$L_P = 20\log_{10}\left(\max_n \lvert x[n] \rvert\right)$$

This equation computes peak level in dBFS by converting the largest absolute sample magnitude into a logarithmic full-scale measurement.

Where: $L_P$ is peak level in dBFS, $\max_n \lvert x[n] \rvert$ is the largest absolute sample amplitude in the file, and dBFS uses full scale as the 0 dB reference.

$$L_{rms} = 20\log_{10}(x_{rms})$$

This equation converts RMS amplitude into RMS level in dBFS, which is a simple loudness-like summary of the file.

Where: $L_{rms}$ is RMS level in dBFS and $x_{rms}$ is the RMS amplitude defined above.

$$C = 20\log_{10}\left(\frac{\max_n \lvert x[n] \rvert}{x_{rms}}\right)$$

This equation computes crest factor, which measures how far the signal peaks rise above its typical level.

Where: $C$ is crest factor in dB, $\max_n \lvert x[n] \rvert$ is peak amplitude, and $x_{rms}$ is RMS amplitude.

$$\mu = \frac{1}{N}\sum_{n=0}^{N-1} x[n]$$

This equation computes DC offset, which indicates whether the waveform is biased above or below zero.

Where: $\mu$ is the mean sample value, $x[n]$ is the sample at index $n$, and $N$ is the total number of samples.

$$ZCR = \frac{1}{N-1}\sum_{n=1}^{N-1}\mathbf{1}\left[sgn(x[n]) \neq sgn(x[n-1])\right]$$

This equation computes zero-crossing rate, which counts how often the waveform changes sign between adjacent samples.

Where: $ZCR$ is zero-crossing rate in crossings per sample, $\mathbf{1}[\cdot]$ is 1 when the condition is true and 0 otherwise, and $sgn(\cdot)$ returns the sign of its argument.

| Metric | Unit | Description |
|--------|------|-------------|
| Duration | s | $T = N / f_s$, where $f_s$ is the sample rate |
| Sample Rate | Hz | $f_s$ |
| Channels | — | Mono / stereo channel count |
| Peak Amplitude | — | $\max_n \lvert x[n] \rvert$ |
| Peak Level | dBFS | $L_P$ — 0 dBFS is full scale |
| RMS Level | dBFS | $L_{rms}$ |
| Crest Factor | dB | $C$ — high = dynamic or sparse signal |
| DC Offset | — | $\mu$ — non-zero indicates a DC bias |
| Silence Ratio | % | Fraction of samples with $\lvert x[n] \rvert < 10^{-3}$ |
| Dynamic Range (simple) | dB | $L_P - L_{rms}$ |
| Clipping Detected | 0/1 | 1 if any sample has $\lvert x[n] \rvert \geq 0.999$ |
| Zero Crossing Rate | crossings/sample | ZCR — higher = noisier or fricative-heavy |
| Channel N RMS / Peak Level | dBFS | Per-channel level metrics for multi-channel files |
| Stereo Phase Correlation | — | Correlation of the first stereo pair; negative values warn about mono collapse |
| Stereo Width | — | Side-to-mid energy ratio mapped to 0–1 |
| Interaural Level Difference (ILD) | dB | Level offset between the first stereo pair |
| Interaural Time Difference (ITD) | ms | Peak cross-correlation lag between the first stereo pair |

---

### loudness

Broadcast-standard loudness measurements per ITU-R BS.1770-4 / EBU R128. The K-weighting filter $H_K(s)$ is a two-stage shelving filter applied before integration.

$$L_K = -0.691 + 10\log_{10}\!\left(\frac{1}{T}\int_0^T \lvert x_K(t) \rvert^2 \, dt\right)$$

This equation computes integrated loudness in LUFS after K-weighting, which approximates perceived program loudness according to BS.1770.

Where: $L_K$ is integrated loudness in LUFS, $x_K(t)$ is the K-weighted time-domain signal, $T$ is signal duration in seconds, and $-0.691$ is the BS.1770 calibration offset. Gating uses an absolute threshold of $-70$ LUFS and a relative threshold of $\bar{L} - 10$ LU, where $\bar{L}$ is the ungated integrated loudness.

$$L_{TP} = 20\log_{10}\left(\max_n \lvert x_{\uparrow 4}[n] \rvert\right)$$

This equation computes true-peak level by measuring the highest amplitude after 4x oversampling so inter-sample peaks are not missed.

Where: $L_{TP}$ is true-peak level in dBTP, $x_{\uparrow 4}[n]$ is the waveform after 4x upsampling, and $\max_n \lvert x_{\uparrow 4}[n] \rvert$ is the largest oversampled absolute amplitude.

| Metric | Unit | Description |
|--------|------|-------------|
| Integrated Loudness | LUFS | Gated $L_K$. Streaming target: $-16$ to $-14$ LUFS |
| Loudness Range (LRA) | LU | $L_{hi,95} - L_{lo,10}$ over gated 3 s blocks (EBU R128) |
| Max Short-Term Loudness | LUFS | $\max_t L_K(t)$ over a sliding 3 s window |
| True Peak | dBTP | $L_{TP}$ — 4× oversampled inter-sample peak. Streaming limit: $-1$ dBTP |
| Loudness Delta vs Platform Targets | LU | Delta from Spotify, Apple Music, YouTube, podcast, and broadcast targets. Positive = too loud; negative = too quiet |

---

### spectral

Frequency-domain features computed via STFT. Notation: $X_t[k]$ = complex STFT coefficient at frame $t$ and bin $k$; $P_t[k] = \lvert X_t[k] \rvert^2$ = power; $f_k$ = center frequency of bin $k$ in Hz; $K$ = number of bins; $\langle \cdot \rangle_t$ = mean over all frames.

**Centroid and Bandwidth:**

$$C_t = \frac{\sum_k f_k \, P_t[k]}{\sum_k P_t[k]}, \qquad C = \langle C_t \rangle_t$$

These equations compute spectral centroid per frame and then average it across time, so the final value represents the spectrum's energy-weighted center of mass.

Where: $C_t$ is the centroid for frame $t$, $C$ is the mean centroid across frames, $f_k$ is the frequency of bin $k$, $P_t[k]$ is the power in bin $k$ at frame $t$, and $\langle \cdot \rangle_t$ denotes averaging over frames.

$$B_t = \sqrt{\frac{\sum_k (f_k - C_t)^2 \, P_t[k]}{\sum_k P_t[k]}}, \qquad B = \langle B_t \rangle_t$$

These equations compute spectral bandwidth per frame and then average it, so the final value describes how widely energy spreads around the centroid.

Where: $B_t$ is the bandwidth for frame $t$, $B$ is the mean bandwidth across frames, $f_k$ is the frequency of bin $k$, $C_t$ is the centroid for frame $t$, and $P_t[k]$ is the bin power.

**Spectral Rolloff** (95th-percentile energy frequency):

$$R_t = \min\left\{ f : \sum_{k:\, f_k \leq f} P_t[k] \;\geq\; 0.95 \sum_k P_t[k] \right\}, \qquad R = \langle R_t \rangle_t$$

These equations compute the frequency below which 95% of the frame energy lies, then average that frequency over time.

Where: $R_t$ is rolloff frequency for frame $t$, $R$ is mean rolloff, $f$ is a candidate cutoff frequency, $f_k$ is the frequency of bin $k$, and $P_t[k]$ is the bin power.

**Spectral Flatness** (geometric-to-arithmetic power mean ratio):

$$F_t = \frac{\exp\!\left(\frac{1}{K}\sum_k \ln P_t[k]\right)}{\frac{1}{K}\sum_k P_t[k]}, \qquad F_{dB} = 10\log_{10}\langle F_t \rangle_t$$

These equations compare the geometric and arithmetic means of spectral power. Flat, noise-like spectra produce larger values, while tonal spectra produce smaller values.

Where: $F_t$ is flatness for frame $t$, $F_{dB}$ is the average flatness expressed in dB, $K$ is the number of spectral bins, and $P_t[k]$ is the power in bin $k$ at frame $t$.

$F_{dB} = 0$ dB corresponds to white noise; more negative values indicate tonal content.

**Spectral Flux:**

$$\Phi_t = \sqrt{\sum_k \left(\lvert X_t[k] \rvert - \lvert X_{t-1}[k] \rvert\right)^2}, \qquad \Phi = \langle \Phi_t \rangle_t$$

These equations compute how much the spectrum changes from one frame to the next and then average that change over time.

Where: $\Phi_t$ is spectral flux for frame $t$, $\Phi$ is mean spectral flux, and $\lvert X_t[k] \rvert$ and $\lvert X_{t-1}[k] \rvert$ are magnitude spectra for adjacent frames.

**Spectral Entropy:**

$$H_t = -\sum_k p_{t,k} \log_2 p_{t,k}, \quad p_{t,k} = \frac{P_t[k]}{\sum_j P_t[j]}, \qquad H = \langle H_t \rangle_t \quad \text{bits}$$

These equations normalize each frame spectrum into a probability distribution and then measure how evenly energy is spread across bins.

Where: $H_t$ is entropy for frame $t$, $H$ is mean entropy, $p_{t,k}$ is the normalized power fraction in bin $k$ for frame $t$, and $P_t[k]$ is the bin power.

**Estimated F0** (autocorrelation):

$$\hat{f}_0 = \frac{f_s}{\hat{\tau}}, \qquad R_{xx}[\hat{\tau}] = \max_{\tau \in [\tau_{min},\, \tau_{max}]} R_{xx}[\tau]$$

These equations estimate the fundamental frequency by finding the autocorrelation lag with the strongest periodic match and converting that lag into frequency.

Where: $\hat{f}_0$ is the estimated fundamental frequency, $f_s$ is the sample rate, $\hat{\tau}$ is the selected lag in samples, $\tau_{min}$ and $\tau_{max}$ bound the allowed lag search range, and $R_{xx}[\tau] = \sum_n x[n]\,x[n+\tau]$ is the autocorrelation at lag $\tau$.

**Band Energy** in band $B$:

$$E_B = \frac{\sum_{k:\, f_k \in B} P[k]}{\sum_k P[k]} \times 100\%$$

This equation computes the percentage of total spectral power that falls inside a named frequency band.

Where: $E_B$ is band energy as a percentage, $B$ is the selected frequency band, $f_k$ is the frequency of bin $k$, and $P[k]$ is the power in bin $k$.

| Metric | Unit | Description |
|--------|------|-------------|
| Spectral Centroid | Hz | $C$ — center of mass of the power spectrum |
| Spectral Bandwidth | Hz | $B$ — weighted std dev around centroid |
| Spectral Rolloff (95%) | Hz | $R$ — frequency below which 95% of energy falls |
| Spectral Flatness | dB | $F_{dB}$ — 0 dB = white noise; more negative = tonal |
| Spectral Flux | — | $\Phi$ — mean frame-to-frame magnitude change |
| Spectral Skewness | — | Third standardized moment of $P_t[k]$ over $k$ |
| Spectral Entropy | bits | $H$ — 0 = single tone; $\log_2 K$ = white noise |
| Effective Bandwidth | Hz | Frequency span with energy above $-30$ dB of peak |
| Estimated F0 | Hz | $\hat{f}_0$ via CREPE when available, otherwise autocorrelation |
| Band Energy: Sub-Bass (20–80 Hz) | % | $E_B$ per band |
| Band Energy: Bass (80–300 Hz) | % | $E_B$ per band |
| Band Energy: Low-Mid (300–2000 Hz) | % | $E_B$ per band |
| Band Energy: Presence (2–6 kHz) | % | $E_B$ per band |
| Band Energy: Air (6–20 kHz) | % | $E_B$ per band |
| High-Frequency Content (HFC) | — | $\langle \sum_k f_k \cdot P_t[k] \rangle_t$ — energy weighted toward high frequencies |

---

### temporal

Time-domain structure, speech activity, and pause patterns. Frame energy at frame $m$:

$$E_m = \frac{1}{L}\sum_{n=0}^{L-1} x[mH + n]^2$$

This equation computes short-time frame energy, which is the basis for activity detection, pause finding, and temporal dynamics.

Where: $E_m$ is frame energy for frame $m$, $L$ is frame length in samples, $H$ is hop size in samples, $x[\cdot]$ is the waveform, and $m$ is the frame index. In practice, $L$ corresponds to 25 ms and $H$ corresponds to 10 ms. We also use $E_m^{dB} = 10\log_{10}(E_m)$ and a VAD threshold $\theta = \max(P_{30}(E^{dB}), -50 \text{ dBFS})$, where $P_{30}$ is the 30th percentile of frame-energy values and $M$ is the total number of frames.

**Temporal Centroid:**

$$\bar{t} = \frac{\sum_m t_m \, E_m}{\sum_m E_m}$$

This equation computes the energy-weighted center of time, so earlier energy pulls the value left and later energy pulls it right.

Where: $\bar{t}$ is temporal centroid in seconds, $t_m = mH / f_s$ is the time stamp of frame $m$, $E_m$ is frame energy, $H$ is hop size in samples, and $f_s$ is the sample rate.

| Metric | Unit | Description |
|--------|------|-------------|
| Speech/Activity Ratio | % | $\frac{1}{M}\sum_m \mathbf{1}[E_m^{dB} > \theta] \times 100$ |
| Attack Time | ms | Time to reach $0.9 \cdot \max_m E_m$ from start |
| Temporal Centroid | s | $\bar{t}$ — energy-weighted mean time |
| Num Pauses (>100 ms) | — | Count of contiguous inactive runs longer than 100 ms |
| Mean Pause Duration | s | Mean length of detected pauses |
| Max Pause Duration | s | Length of the longest detected pause |
| Energy Variance | dB² | $Var(E^{dB})$ — high = dynamic, low = monotone |
| ZCR Mean | crossings/sample | $\langle ZCR_m \rangle_m$ |
| ZCR Variance | — | $Var(ZCR_m)$ |

---

### noise

Noise floor and signal quality estimates. The noise floor $\hat{N}$ is the mean energy of the quietest 10% of frames.

$$\text{SNR} = 10\log_{10}\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right) \;\text{dB}$$

This equation estimates signal-to-noise ratio by comparing high-activity frame power against low-activity frame power.

Where: $\text{SNR}$ is signal-to-noise ratio in dB, $P_{signal}$ is the mean frame power of the most active 50% of frames, and $P_{noise}$ is the mean frame power of the quietest 10% of frames.

$$\text{HNR} = 10\log_{10}\left(\frac{P_{\text{harmonic}}}{P_{\text{aperiodic}}}\right) \;\text{dB}$$

This equation estimates harmonic-to-noise ratio by comparing periodic harmonic energy against residual aperiodic energy.

Where: $\text{HNR}$ is harmonic-to-noise ratio in dB, $P_{harmonic}$ is power concentrated at harmonic multiples of the estimated fundamental frequency, and $P_{aperiodic}$ is the remaining non-harmonic power.

| Metric | Unit | Description |
|--------|------|-------------|
| Estimated Noise Floor | dBFS | $\hat{N} = \langle E_m^{dB} \rangle$ over quietest 10% of frames |
| Estimated SNR | dB | $\langle E_m^{dB} \rangle_{active} - \hat{N}$ |
| Spectral SNR | dB | $10\log_{10}(P_{active} / P_{quiet})$ in the frequency domain |
| Harmonic-to-Noise Ratio (HNR) | dB | HNR — higher = cleaner voiced speech |
| Near-Clipped Samples | count | Samples within 1 dB of full scale |
| Detected Dropouts | count | Sudden near-silence drops in otherwise active audio |

---

### speech

Speech-specific features computed on a mono signal. MFCCs are computed from the log Mel filterbank via DCT-II:

$$c_k = \sqrt{\frac{2}{M}}\sum_{j=1}^{M} m_j \cos\!\left(\frac{\pi k (j - 0.5)}{M}\right), \quad k = 1, \ldots, 13$$

This equation computes the first 13 MFCCs from the log Mel spectrum, which compactly summarize speech spectral shape.

Where: $c_k$ is MFCC coefficient $k$, $M$ is the number of Mel filters, $m_j = \log(\mathbf{f}_j^\top \mathbf{p} + \varepsilon)$ is the log energy in Mel filter $j$, $\mathbf{f}_j$ is the $j$th Mel filter, $\mathbf{p}$ is the frame power spectrum, and $\varepsilon$ is a small constant used to avoid $\log(0)$. In practice, $M = 40$ spanning roughly 80 Hz to 8 kHz.

MFCC statistics over $T$ frames:

$$\bar{c}_k = \frac{1}{T}\sum_{t=1}^T c_k(t), \qquad \sigma_k = \sqrt{\frac{1}{T}\sum_{t=1}^T \left(c_k(t) - \bar{c}_k\right)^2}$$

These equations summarize each MFCC over time using its mean and standard deviation.

Where: $\bar{c}_k$ is the mean of coefficient $k$ over all frames, $\sigma_k$ is the standard deviation of coefficient $k$, $c_k(t)$ is coefficient $k$ at frame $t$, and $T$ is the number of analyzed frames.

**Active Speech Level** (ASL, P.56-inspired):

$$ASL = 20\log_{10}\left(\sqrt{\frac{1}{|A|}\sum_{m \in A} r_m^2}\right)$$

This equation reports the RMS level of speech-active frames only, so pauses and long silences do not drag the speech level downward.

Where: $ASL$ is active speech level in dBFS, $A$ is the set of speech-active frames selected by an activity gate, $|A|$ is the number of active frames, and $r_m$ is the RMS amplitude of frame $m$.

| Metric | Unit | Description |
|--------|------|-------------|
| MFCC 1–13 Mean | — | $\bar{c}_k$ per coefficient |
| MFCC 1–13 Std | — | $\sigma_k$ per coefficient |
| F1-Region Energy (300–1000 Hz) | % | $E_B$ in first formant band |
| F2-Region Energy (1–2.5 kHz) | % | $E_B$ in second formant band |
| F3-Region Energy (2.5–3.5 kHz) | % | $E_B$ in third formant band |
| Voiced/Unvoiced Ratio | — | Voiced frames (low ZCR + high energy) divided by unvoiced frames |
| Active Speech Level (ASL) | dBFS | P.56-inspired RMS level over speech-active frames |
| Speech-Band SNR (300 Hz – 3.4 kHz) | dB | SNR restricted to the telephone/speech band |

---

### perceptual

Perceptual quality scores. Non-intrusive proxies are used by default; supply `--reference` to enable true intrusive metrics.

**SI-SDR** (scale-invariant signal-to-distortion ratio):

$$\text{SI-SDR} = 10\log_{10}\!\left(\frac{\lVert \alpha \mathbf{r} \rVert^2}{\lVert \hat{\mathbf{x}} - \alpha \mathbf{r} \rVert^2}\right), \qquad \alpha = \frac{\hat{\mathbf{x}}^\top \mathbf{r}}{\lVert \mathbf{r} \rVert^2}$$

This equation computes the distortion level after optimally rescaling the reference to match the degraded signal, which makes the score invariant to overall gain.

Where: $\text{SI-SDR}$ is scale-invariant signal-to-distortion ratio in dB, $\hat{\mathbf{x}}$ is the zero-mean degraded signal vector, $\mathbf{r}$ is the zero-mean reference vector, $\alpha$ is the least-squares projection scalar, and $\hat{\mathbf{x}} - \alpha \mathbf{r}$ is the residual distortion vector.

**SDR:**

$$\text{SDR} = 10\log_{10}\!\left(\frac{\lVert \mathbf{r} \rVert^2}{\lVert \mathbf{r} - \hat{\mathbf{x}} \rVert^2}\right)$$

This equation computes a simpler signal-to-distortion ratio without scale normalization, so gain mismatches directly affect the result.

Where: $\text{SDR}$ is signal-to-distortion ratio in dB, $\mathbf{r}$ is the reference signal vector, and $\hat{\mathbf{x}}$ is the degraded signal vector.

**Log-Spectral Distance:**

$$\text{LSD} = \frac{1}{KT}\sum_{t,k} \left\lvert \log \lvert X_t[k] \rvert^2 - \log \lvert R_t[k] \rvert^2 \right\rvert$$

This equation computes the average absolute gap between the degraded and reference log spectra across all bins and frames.

Where: $\text{LSD}$ is log-spectral distance, $K$ is the number of frequency bins, $T$ is the number of frames, $X_t[k]$ is the degraded STFT coefficient at frame $t$ and bin $k$, and $R_t[k]$ is the reference STFT coefficient at frame $t$ and bin $k$.

**Cepstral Distance:**

$$\text{CD} = \frac{1}{T}\sum_{t=1}^T \sqrt{\sum_{k=1}^{13}\left(c_k(t) - c_k^{(r)}(t)\right)^2}$$

This equation computes the Euclidean distance between degraded and reference MFCC vectors and then averages that distance over time.

Where: $\text{CD}$ is cepstral distance, $T$ is the number of frames, $c_k(t)$ is degraded MFCC coefficient $k$ at frame $t$, and $c_k^{(r)}(t)$ is reference MFCC coefficient $k$ at frame $t$.

| Metric | Unit | Notes |
|--------|------|-------|
| DNSMOS P.835 SIG / BAK / OVRL | MOS 1–5 | ONNX if model files are present, otherwise speech-quality proxies |
| AECMOS | MOS 1–5 | Echo-aware quality score via ONNX if available, otherwise autocorrelation proxy |
| Estimated MOS (non-intrusive) | 1–5 | Heuristic from SNR and spectral shape |
| UTMOS / SHEET MOS | MOS 1–5 | ONNX if model files are present, otherwise learned-MOS-style proxies |
| Estimated Codec Bandwidth Cutoff | Hz | Heuristic upper bandwidth before codec-style low-pass loss dominates |
| Spectral Hole Ratio | % | Mid/high-band notch density relative to a smoothed spectral envelope |
| Pre-echo Risk | — | Transient smear heuristic for codec ringing or pre-echo |
| Codec Artifact Risk | % | Combined heuristic risk for codec artifacts |
| P.563 Proxy (NB Quality Estimate) | 1–4.5 | Improved narrowband non-intrusive quality proxy |
| PESQ (ITU-T P.862) | MOS-LQO | Requires `--reference`. True P.862 if `pesq` installed |
| STOI | 0–1 | Requires `--reference`. True STOI if `pystoi` installed |
| SI-SDR | dB | Requires `--reference` |
| Log-Spectral Distance | dB | Requires `--reference` |

---

### prosody

Per-frame F0 trajectory, perturbation measures, and speech rate. qualiax uses CREPE when it is installed and falls back to normalized autocorrelation otherwise.

**Normalized autocorrelation F0 detection:**

$$r_{xx}[\tau] = \frac{\sum_n x[n]\, x[n+\tau]}{\sum_n x[n]^2}, \qquad \hat{f}_0 = \frac{f_s}{\hat{\tau}}, \qquad r_{xx}[\hat{\tau}] = \max_{\tau \in [\tau_{min},\, \tau_{max}]} r_{xx}[\tau]$$

These equations compute normalized autocorrelation, choose the strongest periodic lag in the allowed pitch range, and convert that lag into a fundamental-frequency estimate.

Where: $r_{xx}[\tau]$ is normalized autocorrelation at lag $\tau$, $\hat{f}_0$ is the estimated fundamental frequency, $f_s$ is the sample rate, $\hat{\tau}$ is the selected lag in samples, $\tau_{min} = \lfloor f_s / f_{0,max} \rfloor$, $\tau_{max} = \lfloor f_s / f_{0,min} \rfloor$, $f_{0,min} = 60$ Hz, and $f_{0,max} = 500$ Hz. A frame is treated as voiced when $r_{xx}[\hat{\tau}] > 0.40$.

**Jitter** (local F0 period perturbation, Baken & Orlikoff 2000):

$$J = \frac{1}{\bar{T}(N-1)} \sum_{i=1}^{N-1} \lvert T_i - T_{i-1} \rvert \times 100\%, \qquad \bar{T} = \frac{1}{N}\sum_{i=1}^{N} T_i$$

This equation computes local jitter by measuring how much consecutive voiced periods fluctuate relative to their mean period.

Where: $J$ is local jitter in percent, $T_i = 1 / f_{0,i}$ is the period of voiced frame $i$, $\bar{T}$ is the mean voiced period, $f_{0,i}$ is the fundamental frequency of voiced frame $i$, and $N$ is the number of voiced frames used in the estimate.

**Shimmer** (local amplitude perturbation):

$$S = \frac{1}{\bar{A}(N-1)} \sum_{i=1}^{N-1} \lvert A_i - A_{i-1} \rvert \times 100\%, \qquad \bar{A} = \frac{1}{N}\sum_{i=1}^{N} A_i$$

This equation computes local shimmer by measuring frame-to-frame amplitude variation relative to the mean voiced amplitude.

Where: $S$ is local shimmer in percent, $A_i$ is the RMS amplitude of voiced frame $i$, $\bar{A}$ is the mean voiced amplitude, and $N$ is the number of voiced frames used in the estimate.

| Metric | Unit | Description |
|--------|------|-------------|
| Voiced Frame Ratio | % | Fraction of frames with $r_{xx}[\hat{\tau}] > 0.40$ |
| F0 Estimator Backend | — | `crepe` when available, otherwise `autocorrelation` |
| F0 Mean | Hz | Mean F0 over voiced frames |
| F0 Std | Hz | Standard deviation of voiced F0 |
| F0 Min / Max | Hz | Extremes of voiced F0 |
| F0 Range | Hz | $f_{0,\max} - f_{0,\min}$ |
| Pitch Variability (CV) | % | $\sigma_{f_0} / \bar{f}_0 \times 100$ — higher = more expressive |
| F0 Slope | Hz/s | Linear regression slope of $f_0(t)$ over voiced frames |
| Jitter (Local) | % | $J$ — normal speech below 1%; higher = dysphonia or roughness |
| Shimmer (Local) | % | $S$ — normal speech below 3%; higher = breathiness or hoarseness |
| Tremor Rate | Hz | Dominant spectral peak of F0 modulation in the 2–15 Hz band |
| Tremor Depth | Hz² | Power of the dominant F0 modulation component |
| Estimated Speech Rate | syll/s | Energy-envelope peak count / duration. Typical: 3–7 syll/s |

---

### psychoacoustic

Perceptual features based on auditory models using Bark-scale critical-band analysis. The Bark scale $z$ is (Traunmüller 1990):

$$z = \frac{26.81 \, f}{1960 + f} - 0.53$$

This equation maps frequency in Hz onto the Bark scale, which better reflects auditory critical-band spacing than linear frequency.

Where: $z$ is Bark-scale position and $f$ is frequency in Hz.

**Roughness** (Vassilakis 2001) — amplitude modulation between spectral partial pairs:

$$R = \sum_{i < j} \left(\frac{A_i A_j}{A_i^2 + A_j^2}\right)^{3.11} (A_i A_j)^{0.1} \cdot x^2 e^{-(x/0.25)^2}, \qquad x = \frac{\lvert f_j - f_i \rvert}{\text{CBW}(f_i)}$$

This equation estimates auditory roughness by summing beating interactions between partial pairs, with the strongest contribution when their spacing is a quarter of a critical bandwidth.

Where: $R$ is roughness, $A_i$ and $A_j$ are normalized amplitudes of partials $i$ and $j$, $f_i$ and $f_j$ are their frequencies in Hz with $f_i < f_j$, $x$ is normalized frequency spacing, and $CBW(f_i) = 25 + 75\left(1 + 1.4(f_i/1000)^2\right)^{0.69}$ is the critical bandwidth around $f_i$.

**Sensory Dissonance** (Sethares 1993):

$$D = \sum_{i < j} A_i A_j \left(e^{-b_1 s \lvert f_j - f_i \rvert} - e^{-b_2 s \lvert f_j - f_i \rvert}\right), \qquad s = \frac{0.24}{0.0207 f_i + 18.96}$$

This equation estimates sensory dissonance by summing how strongly each partial pair is expected to beat within the auditory system.

Where: $D$ is sensory dissonance, $A_i$ and $A_j$ are normalized partial amplitudes, $f_i$ and $f_j$ are partial frequencies, $b_1 = 3.5$ and $b_2 = 5.75$ are empirical constants, and $s = 0.24 / (0.0207 f_i + 18.96)$ is the lower-partial scaling factor.

**Sharpness** (Zwicker & Fastl 1990; Von Bismarck 1974):

$$S_{acum} = 0.11 \frac{\sum_z N'(z)\, g(z)\, z}{\sum_z N'(z)}, \qquad g(z) = \begin{cases} 1 & z \leq 15 \\ 0.066\, e^{0.171 z} & z > 15 \end{cases}$$

This equation computes sharpness by weighting specific loudness toward higher Bark bands, so bright high-frequency energy contributes more strongly.

Where: $S_{acum}$ is sharpness in acum, $z$ is Bark-band position, $N'(z)$ is specific loudness in band $z$, and $g(z)$ is the high-frequency weighting function defined piecewise above.

**Spectral Flatness and Tonality:**

$$\text{SFM} = \frac{\exp\!\left(\frac{1}{K}\sum_k \ln P[k]\right)}{\frac{1}{K}\sum_k P[k]}, \qquad \text{Tonality} = 1 - \text{SFM}$$

These equations compute spectral flatness and then derive tonality as its complement, so tonal signals push tonality upward while noise-like signals push it downward.

Where: $\text{SFM}$ is spectral flatness measure, $\text{Tonality}$ is the derived tonality score, $K$ is the number of spectral bins, and $P[k]$ is the power in bin $k$.

| Metric | Unit | Description |
|--------|------|-------------|
| Roughness | asper (rel.) | $R$ — Vassilakis AM roughness. High = grating or harsh |
| Sensory Dissonance | (rel.) | $D$ — Sethares beating model. Low = consonant spectrum |
| Sharpness | acum | $S_{acum}$ — 1 acum = reference 1 kHz narrow-band noise |
| Spectral Flatness (SFM) | 0–1 | 0 = pure sine tone; 1 = white noise |
| Tonality | 0–1 | $1 - \text{SFM}$. High = tonal; low = noise-like |

---

### speaker

Voice characteristics and speaker demographics estimated from acoustic features. All estimates are heuristic approximations, not biometric classifiers.

**LPC Formant Tracking** — order-$p$ LPC via Levinson-Durbin on the Yule-Walker system:

$$\mathbf{R}\,\mathbf{a} = -\mathbf{r}_{1:p}, \qquad \mathbf{R}_{ij} = r[\lvert i - j \rvert]$$

This equation solves the LPC normal equations, which estimate an all-pole vocal-tract model from short-time speech autocorrelation.

Where: $\mathbf{R}$ is the $p \times p$ symmetric Toeplitz autocorrelation matrix, $\mathbf{a} = [a_1, \ldots, a_p]^T$ is the LPC coefficient vector, $r[k] = \frac{1}{N}\sum_n x[n]\, x[n+k]$ is biased autocorrelation at lag $k$, and $p = 12$ is the LPC order used here.

Formant frequencies and bandwidths from LPC polynomial roots $z_k$ in the upper half of the complex plane:

$$F_k = \frac{\phi_k f_s}{2\pi}, \qquad BW_k = \frac{-\ln \lvert z_k \rvert \cdot f_s}{\pi}$$

These equations convert LPC roots into formant frequency and bandwidth estimates.

Where: $F_k$ is the frequency of formant candidate $k$, $BW_k$ is its bandwidth, $\phi_k$ is the phase angle of root $z_k$ in radians, $f_s$ is the sample rate, and $\lvert z_k \rvert$ is the magnitude of root $z_k$. The LPC polynomial is $A(z) = 1 + a_1 z^{-1} + \cdots + a_p z^{-p}$. Only roots with $50 < F_k < 5500$ Hz and $BW_k < 600$ Hz are retained.

**Cepstral Peak Prominence** (Hillenbrand et al. 1994):

$$\text{CPP} = \max_{q \in [q_{\min},\, q_{\max}]} \left[ c[q] - \hat{c}[q] \right]$$

This equation measures how strongly the dominant cepstral pitch peak rises above its smooth baseline, which is a proxy for periodic voice clarity.

Where: $\text{CPP}$ is cepstral peak prominence in dB, $q$ is quefrency in samples, $q_{min}$ and $q_{max}$ define the searched quefrency range, $c[q] = \lvert \mathcal{F}^{-1}\{\log \lvert X \rvert^2\} \rvert$ is the real cepstrum, and $\hat{c}[q]$ is the linear-regression baseline over the pitch-relevant quefrency interval.

**Gender estimation** (Traunmüller & Eriksson 1995 empirical distributions):

| F0 Mean | Estimate |
|---------|----------|
| below 145 Hz | male (adult) |
| 145–180 Hz | ambiguous overlap zone |
| 180–260 Hz | female (adult) |
| above 260 Hz | child / high soprano |

**Age estimation** — heuristic scoring from jitter, shimmer, HNR, spectral tilt, and pitch variability (Linville 2001; Xue & Deliyski 2001). Accuracy approximately ±15 years.

| Metric | Unit | Description |
|--------|------|-------------|
| F1–F4 Formant Frequency | Hz | Median $F_k$ over voiced frames (LPC, $p = 12$) |
| Spectral Tilt | dB/oct | Power spectrum slope 100 Hz – Nyquist. Typical speech: $-6$ to $-12$ dB/oct |
| Cepstral Peak Prominence (CPP) | dB | Higher = clearer periodic voice. Above 5 dB = modal; below 3 dB = breathy |
| Breathiness Index | 0–1 | CPP-derived. 0 = modal voice; 1 = highly breathy |
| Creakiness (Vocal Fry) Ratio | % | Fraction of active frames with autocorrelation peak in 20–80 Hz |
| Estimated Gender | — | Heuristic from F0 mean. 145–180 Hz = ambiguous overlap zone |
| Gender Confidence | 0–1 | Distance from overlap zone as proxy for certainty |
| Estimated Age Range | — | Heuristic from jitter, shimmer, HNR, tilt, pitch variability (approx. ±15 years) |

---

## Output

### Pretty (Console)

Default output. Metrics are grouped by category with color coding. Warnings are highlighted when values fall outside reference ranges. A colored progress bar is shown during analysis:

```
  [████████████████████████████] 3/3  interview_003.wav
```

Disable color:

```bash
qualiax recording.wav --no-color
```

### JSON

Use `--save-sidecar` to write `filename.json` alongside each source file. To write a single aggregate file instead:

```bash
qualiax recording.wav --output custom.json
```

Structure:

```json
[
  {
    "schema_version": "3.4",
    "tool_version": "0.11.0",
    "file": "recording.wav",
    "source_file": null,
    "segment_index": null,
    "total_segments": null,
    "segment_start_s": null,
    "segment_end_s": null,
    "duration_s": 4.23,
    "sample_rate": 44100,
    "channels": 1,
    "bit_depth": null,
    "content_type": "speech",
    "speech_confidence": 0.91,
    "notes": [
      "perceptual: fell back to proxy metric because optional dependency was unavailable"
    ],
    "confidence_notes": [
      "Includes proxy-derived metrics; use them for directional review rather than absolute scoring."
    ],
    "diagnostics": [
      {
        "code": "metric_group_warning",
        "severity": "warn",
        "source": "metric_group",
        "message": "proxy fallback used",
        "group": "perceptual",
        "metric": null,
        "count": 1,
        "context": {}
      }
    ],
    "group_health": [
      {
        "group": "loudness",
        "status": "ok",
        "metric_count": 1,
        "warning_count": 0,
        "missing_count": 0,
        "diagnostic_count": 0
      }
    ],
    "provenance": {
      "compute_backend": "cpu",
      "python_version": "3.14.3",
      "platform": "macOS-...",
      "runtime_fingerprint": "abc123def4567890",
      "model_runtime": "none",
      "model_assets": []
    },
    "error": null,
    "metrics": [
      {
        "name": "Integrated Loudness (LUFS)",
        "value": -16.2,
        "unit": "LUFS",
        "description": "ITU-R BS.1770 / EBU R128 integrated loudness (gated)",
        "group": "loudness",
        "higher_is_better": null,
        "warning": null,
        "confidence": "measured",
        "calibration_note": null
      }
    ]
  }
]
```

`schema_version` is the compatibility marker for downstream parsers. `tool_version` records the producing qualiax release. `content_type` now comes from the shared speech detector and may be `speech`, `music`, `noise`, `silence`, `mixed`, or `unknown`. `speech_confidence` is that detector's 0–1 score. For `mixed` files with substantial detected speech, qualiax retains speech-oriented groups and records that decision in `notes`; otherwise speech-only groups are skipped and the skip reason is recorded there as well. `notes` captures degraded-but-successful execution paths, while `confidence_notes` summarizes trust and calibration caveats across the file. `diagnostics` promotes those runtime events into machine-readable entries with severity, source, code, and context. `group_health` summarizes per-group status for automation and scorecards. `provenance` records backend/runtime details and model asset fingerprints for reproducibility. Metric objects now also expose `confidence` and `calibration_note` so downstream consumers can distinguish measured, model-backed, proxy, and heuristic outputs. In segment mode, `source_file`, `segment_index`, `total_segments`, `segment_start_s`, and `segment_end_s` identify the parent recording and the segment boundaries. `error` is reserved for whole-file failures.

### CSV

Flat table with one row per result and one column per metric. The `schema_version`, `tool_version`, `content_type`, `speech_confidence`, `notes`, `confidence_notes`, `diagnostics`, `group_health`, `runtime_fingerprint`, and segment metadata columns mirror the JSON metadata so batch consumers can detect contract changes, degraded analyses, provenance changes, and segment boundaries:

```bash
qualiax ./calls/ --output report.csv
```

### HTML

Self-contained single-file report with file summary cards, grouped metric tables, notes, and per-group status badges:

```bash
qualiax ./calls/ --output report.html
```

Use `--format html` to print the same report to stdout when needed.

### Markdown

Markdown output is designed for docs, changelogs, and PR comments:

```bash
qualiax recording.wav --output report.md
qualiax diff before.json after.json --format markdown
```

The Markdown renderer includes file-level status, notes, and grouped metric tables that copy cleanly into GitHub or documentation pages.

### Scorecards

Use `--scorecard` to emit a batch-level summary with per-metric rollups, percentiles, warning counts, and outlier files:

```bash
qualiax ./calls/ --output results.json --scorecard scorecard.md --silent
qualiax ./calls/ --scorecard scorecard.html --silent
```

Scorecards now aggregate by original source file when segment metadata is present, so long recordings analyzed with `--segment-seconds` do not outweigh shorter files just because they produce more segments. They also summarize categorical outputs and confidence-count distributions, so non-numeric metrics and trust labels remain visible in the batch view.

---

## Hardware Acceleration

qualiax automatically selects the best available compute backend:

1. **CUDA** — NVIDIA GPU via PyTorch
2. **MPS** — Apple Silicon (M1/M2/M3/M4) via PyTorch Metal
3. **CPU** — NumPy/SciPy fallback (always available)

Install PyTorch to enable GPU acceleration:

```bash
pip install torch
```

Accelerated operations: STFT, MFCC, autocorrelation, spectral features, K-weighting filter, true peak.

To force a device in code:

```python
from qualiax.gpu import set_device
set_device("cpu")   # or "cuda" / "mps"
```

---

## Supported Formats

| Format | Extension | Requires |
|--------|-----------|----------|
| WAV | `.wav` | stdlib PCM loader (always available; supports 8/16/24/32-bit PCM) |
| FLAC | `.flac` | `soundfile` |
| OGG Vorbis | `.ogg` | `soundfile` |
| AIFF | `.aiff`, `.aif` | `soundfile` |
| MP3 | `.mp3` | `pydub` + ffmpeg |
| AAC | `.aac`, `.m4a` | `pydub` + ffmpeg |
| Opus | `.opus` | `pydub` + ffmpeg |
| MP4 / MOV / MKV | `.mp4`, `.mov`, `.mkv` | `pydub` + ffmpeg |

---

## Requirements

- Python 3.9–3.14
- `click >= 8.0`
- `numpy >= 1.23`
- `scipy >= 1.9`

Optional:

- `soundfile >= 0.12` — FLAC, OGG, AIFF
- `pydub >= 0.25` + ffmpeg — MP3, AAC, M4A, Opus
- `pesq >= 0.0.4` — True PESQ scores
- `pystoi >= 0.3` — True STOI scores
- `librosa >= 0.10` — Universal audio loader fallback
- `torch` — GPU acceleration (CUDA or Apple MPS)

---

## References

### Standards & Specifications

- ITU-R BS.1770-4 (2015). *Algorithms to measure audio programme loudness and true-peak audio level.* International Telecommunication Union.
- EBU R128 (2020). *Loudness normalisation and permitted maximum level of audio signals.* European Broadcasting Union.
- ITU-T P.862 (2001). *Perceptual evaluation of speech quality (PESQ).* International Telecommunication Union.
- ITU-T P.563 (2004). *Single-ended method for objective speech quality assessment.* International Telecommunication Union.

### Perceptual Quality

- Rix, A. W., et al. (2001). Perceptual evaluation of speech quality (PESQ). *ICASSP 2001*, 749–752.
- Taal, C. H., et al. (2011). An algorithm for intelligibility prediction of time–frequency weighted noisy speech. *IEEE Trans. Audio, Speech, Lang. Process.*, 19(7), 2125–2136.
- Le Roux, J., et al. (2019). SDR — half-baked or well done? *ICASSP 2019*, 626–630.

### Psychoacoustics

- Plomp, R., & Levelt, W. J. M. (1965). Tonal consonance and critical bandwidth. *JASA*, 38(4), 548–560.
- Sethares, W. A. (1993). Local consonance and the relationship between timbre and scale. *JASA*, 94(3), 1218–1228.
- Vassilakis, P. N. (2001). *Perceptual and physical properties of amplitude fluctuation.* PhD dissertation, UCLA.
- Zwicker, E., & Fastl, H. (1990). *Psychoacoustics: Facts and models.* Springer.
- Von Bismarck, G. (1974). Sharpness as an attribute of the timbre of steady sounds. *Acustica*, 30(3), 159–172.
- Zwicker, E. (1961). Subdivision of the audible frequency range into critical bands. *JASA*, 33(2), 248.
- Traunmüller, H. (1990). Analytical expressions for the tonotopic sensory scale. *JASA*, 88(1), 97–100.

### Prosody & F0

- Boersma, P. (1993). Accurate short-term analysis of the fundamental frequency and the HNR. *Proc. Institute of Phonetic Sciences*, 17, 97–110.
- De Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency estimator for speech and music. *JASA*, 111(4), 1917–1930.
- Mermelstein, P. (1975). Automatic segmentation of speech into syllabic units. *JASA*, 58(4), 880–883.
- Baken, R. J., & Orlikoff, R. F. (2000). *Clinical measurement of speech and voice* (2nd ed.). Singular Publishing.

### Voice Quality & Speaker Characteristics

- Hillenbrand, J., et al. (1994). Acoustic correlates of breathy vocal quality. *J. Speech Hear. Res.*, 37(4), 769–778.
- Linville, S. E. (2001). *Vocal aging.* Singular Publishing Group.
- Xue, S. A., & Deliyski, D. D. (2001). Effects of aging on selected acoustic voice parameters. *Educational Gerontology*, 27(2), 159–168.
- Traunmüller, H., & Eriksson, A. (1995). *The frequency range of the voice fundamental in the speech of male and female adults.* Manuscript, Stockholm University.

### LPC, Cepstrum & MFCCs

- Markel, J. D., & Gray, A. H. (1976). *Linear prediction of speech.* Springer.
- Levinson, N. (1947). The Wiener error criterion in filter design and prediction. *J. Math. Phys.*, 25, 261–278.
- Noll, A. M. (1967). Cepstrum pitch determination. *JASA*, 41(2), 293–309.
- Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition. *IEEE Trans. ASSP*, 28(4), 357–366.

### Software

- McFee, B., et al. (2015). librosa: Audio and music signal analysis in Python. *SciPy 2015*, 18–25.
- Boersma, P., & Weenink, D. (2024). *Praat: Doing phonetics by computer* (v6.4). praat.org.

---

## License

MIT
