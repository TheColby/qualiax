# qualiax

Perceptual speech and audio quality analyzer — command-line tool that computes 60+ metrics across 7 analysis groups. Zero required configuration. Point it at a file or folder and get a full quality report.

```
qualiax recording.wav
qualiax ./calls/ --format json --output results.json
qualiax interview.wav --reference clean.wav
```

A sample file is included to try immediately:

```bash
qualiax sample.wav
```

> "The birch canoe slid on the smooth planks." — Harvard Sentence List 1, #1

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
  - [Reference Files](#reference-files)
  - [Filtering Metric Groups](#filtering-metric-groups)
  - [Parallel Processing](#parallel-processing)
  - [All Options](#all-options)
- [Metric Groups](#metric-groups)
  - [basic](#basic-1)
  - [loudness](#loudness)
  - [spectral](#spectral)
  - [temporal](#temporal)
  - [noise](#noise)
  - [speech](#speech)
  - [perceptual](#perceptual)
- [Output](#output)
  - [Pretty (Console)](#pretty-console)
  - [JSON](#json)
  - [CSV](#csv)
- [Hardware Acceleration](#hardware-acceleration)
- [Supported Formats](#supported-formats)
- [Requirements](#requirements)
- [License](#license)

---

## Features

- **60+ metrics** organized into 7 groups: basic signal properties, loudness (EBU R128 / BS.1770), spectral analysis, temporal structure, noise, speech characteristics, and perceptual quality
- **Zero config** — works out of the box on any WAV file with no arguments beyond the path
- **Intrusive and non-intrusive modes** — run standalone or supply a reference file for difference-based metrics (PESQ, STOI, SI-SDR, SDR)
- **Multiple output formats** — colored console, JSON, CSV
- **GPU acceleration** — automatically uses CUDA (NVIDIA) or MPS (Apple Silicon) when PyTorch is installed, falls back to CPU silently
- **Recursive directory analysis** — point at a folder to analyze every audio file in the tree
- **Parallel workers** — thread-based parallelism for batch jobs
- **Robust audio loading** — multi-backend fallback chain handles WAV, FLAC, MP3, AAC, M4A, OGG, Opus, AIFF

---

## Installation

### Quick Install

From the project root:

```bash
# Minimal — WAV support only
pip install -e .

# Recommended — adds MP3, FLAC, OGG, AAC support
pip install -e ".[audio]"

# Full — everything including PESQ, STOI, librosa
pip install -e ".[all]"
```

After installation, `qualiax` is available globally in your environment.

### Install Script

```bash
./install.sh
```

Runs `pip install -e ".[all]"` relative to the project directory. Make it executable first if needed:

```bash
chmod +x install.sh
./install.sh
```

### Optional Dependencies

| Extra | Packages | What it enables |
|-------|----------|-----------------|
| `audio` | `soundfile`, `pydub` | FLAC, OGG, AIFF, MP3, AAC, M4A (requires ffmpeg for MP3/AAC) |
| `perceptual` | `pesq`, `pystoi` | True PESQ P.862 scores, true STOI intelligibility |
| `all` | all of the above + `librosa` | Full format support + perceptual metrics + universal fallback loader |

**ffmpeg** is required for MP3 and AAC/M4A decoding via pydub:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
apt install ffmpeg
```

**PyTorch** is optional but enables GPU acceleration on CUDA and Apple Silicon:

```bash
pip install torch
```

---

## Usage

### Basic

Analyze a single file and print results to the terminal:

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

Write results to a file. The format is inferred from the extension:

```bash
# JSON
qualiax ./calls/ --output results.json

# CSV
qualiax ./calls/ --output results.csv

# Explicit format override
qualiax recording.wav --format json
qualiax recording.wav --format csv
```

Suppress console output when writing to a file:

```bash
qualiax ./calls/ --output results.json --silent
```

### Reference Files

Supply a clean reference recording to enable intrusive metrics (PESQ, STOI, SI-SDR, SDR, log-spectral distance, spectral correlation, cepstral distance):

```bash
qualiax degraded.wav --reference clean.wav
```

Without a reference, perceptual metrics fall back to non-intrusive proxies.

### Filtering Metric Groups

Run only specific groups to speed up analysis or focus the output:

```bash
# Single group
qualiax recording.wav --metrics basic

# Multiple groups
qualiax recording.wav --metrics basic,loudness,spectral

# All groups (default)
qualiax recording.wav --metrics all
```

Available groups: `basic`, `loudness`, `spectral`, `temporal`, `noise`, `speech`, `perceptual`

### Parallel Processing

Use multiple threads for batch analysis:

```bash
qualiax ./calls/ --workers 4
```

### All Options

```
Usage: qualiax [OPTIONS] PATHS...

Options:
  --silent                        Suppress console output (useful with --output)
  -o, --output PATH               Write results to file (.json or .csv)
  -f, --format [pretty|json|csv]  Output format (default: pretty)
  -r, --reference PATH            Reference audio for intrusive metrics
  -m, --metrics TEXT              Comma-separated metric groups (default: all)
  --no-color                      Disable colored output
  -v, --verbose                   Show progress and warnings
  -w, --workers INTEGER           Parallel worker threads (default: 1)
  --help                          Show this message and exit
```

---

## Metric Groups

### basic

Core signal properties derived directly from the waveform.

| Metric | Unit | Description |
|--------|------|-------------|
| Duration | s | Total file duration |
| Sample Rate | Hz | Audio sample rate |
| Channels | — | Mono / stereo channel count |
| Peak Amplitude | — | Maximum absolute sample value (0–1 scale) |
| Peak Level | dBFS | Peak level relative to full scale |
| RMS Level | dBFS | Root-mean-square energy level |
| Crest Factor | dB | Peak-to-RMS ratio; high values indicate dynamic or sparse signal |
| DC Offset | — | Mean sample value; non-zero indicates a DC component |
| Silence Ratio | % | Fraction of samples below −60 dB |
| Dynamic Range (simple) | dB | Difference between peak and RMS level |
| Clipping Detected | 0/1 | 1 if any sample reaches ≥ 0.999 full scale |
| Zero Crossing Rate | crossings/sample | Rate of sign changes; correlated with pitch and noisiness |

### loudness

Broadcast-standard loudness measurements per ITU-R BS.1770 / EBU R128.

| Metric | Unit | Description |
|--------|------|-------------|
| Integrated Loudness (LUFS) | LUFS | Gated integrated loudness (BS.1770-4). Streaming target: −16 to −14 LUFS |
| Loudness Range (LRA) | LU | EBU R128 loudness range; measures dynamic variation |
| Max Short-Term Loudness | LUFS | Maximum 3-second sliding window loudness |
| True Peak | dBTP | Inter-sample peak level (4× oversampled). Streaming limit: −1 dBTP |

### spectral

Frequency-domain characteristics computed via STFT.

| Metric | Unit | Description |
|--------|------|-------------|
| Spectral Centroid | Hz | Weighted mean frequency; perceptual "brightness" |
| Spectral Bandwidth | Hz | Weighted standard deviation around centroid |
| Spectral Rolloff (95%) | Hz | Frequency below which 95% of spectral energy falls |
| Spectral Flatness | dB | Ratio of geometric to arithmetic mean power; low = tonal, high = noise-like |
| Spectral Flux | — | Mean frame-to-frame spectral change; high values indicate rapid variation |
| Spectral Skewness | — | Asymmetry of the spectral energy distribution |
| Spectral Entropy | bits | Entropy of the normalized power spectrum |
| Estimated F0 | Hz | Fundamental frequency estimate via autocorrelation |
| Sub-bass Energy | dB | Energy in 20–80 Hz band |
| Bass Energy | dB | Energy in 80–250 Hz band |
| Low-mid Energy | dB | Energy in 250–2000 Hz band |
| Presence Energy | dB | Energy in 2000–6000 Hz band |
| Air Energy | dB | Energy in 6000–20000 Hz band |
| High-Frequency Content (HFC) | — | Frequency-weighted power sum; sensitive to high-frequency content |

### temporal

Time-domain structure including speech activity and pause patterns.

| Metric | Unit | Description |
|--------|------|-------------|
| Voice Activity Ratio | % | Fraction of frames with energy above noise floor |
| Attack Time | s | Time to reach peak energy from a low-energy region |
| Temporal Centroid | s | Energy-weighted center of mass in time |
| Pause Count | — | Number of silence gaps longer than 200 ms |
| Mean Pause Duration | s | Average duration of detected silence gaps |
| Energy Variance | — | Variance of per-frame RMS energy |
| ZCR Std | — | Standard deviation of the zero-crossing rate |

### noise

Noise floor and signal quality estimates.

| Metric | Unit | Description |
|--------|------|-------------|
| Noise Floor | dBFS | Estimated background noise level |
| SNR (estimated) | dB | Signal-to-noise ratio estimate |
| Spectral SNR | dB | SNR computed in the frequency domain |
| HNR | dB | Harmonic-to-noise ratio; key voice quality indicator |
| Near-Clipped Samples | — | Count of samples within 1 dB of full scale |
| Dropout Count | — | Number of detected signal interruptions |

### speech

Speech-specific features. Computed on mono signal.

| Metric | Unit | Description |
|--------|------|-------------|
| MFCC Mean (1–13) | — | Mean of each of the first 13 Mel-frequency cepstral coefficients |
| MFCC Std (1–13) | — | Standard deviation of each MFCC coefficient |
| Formant Band Energy (F1–F3) | dB | Energy in estimated formant frequency bands |
| Voiced/Unvoiced Ratio | — | Ratio of voiced to unvoiced frames |
| Speaking Rate | syllables/s | Estimated speaking rate based on energy modulation |
| Speech-Band SNR | dB | SNR in the 300–3400 Hz speech band |

### perceptual

Perceptual quality scores. Non-intrusive proxies are used by default; supply `--reference` to enable true intrusive metrics.

| Metric | Unit | Notes |
|--------|------|-------|
| Pseudo-MOS | 1–5 | Non-intrusive MOS estimate based on acoustic heuristics |
| P.563 Proxy | 1–5 | Proxy for ITU-T P.563 single-ended quality measure |
| PESQ | MOS-LQO | Requires `--reference`. True PESQ (P.862) if `pesq` library installed, else spectral proxy |
| STOI | 0–1 | Requires `--reference`. True STOI if `pystoi` installed, else proxy |
| SI-SDR | dB | Requires `--reference`. Scale-invariant signal-to-distortion ratio |
| SDR | dB | Requires `--reference`. Signal-to-distortion ratio |
| Log-Spectral Distance | dB | Requires `--reference`. Log-domain spectral distortion |
| Spectral Correlation | 0–1 | Requires `--reference`. Frame-by-frame spectral similarity |
| Cepstral Distance | — | Requires `--reference`. Cepstral-domain distortion |

---

## Output

### Pretty (Console)

Default output. Metrics are grouped by category with color coding. Warnings are highlighted when values fall outside reference ranges.

```
┌─ basic ─────────────────────────────────────────────────────────┐
│ Duration              4.23 s
│ Sample Rate           44100 Hz
│ Peak Level           -1.2 dBFS
│ RMS Level           -18.4 dBFS
│ Clipping Detected       0      ✓
└─────────────────────────────────────────────────────────────────┘
```

Disable color for plain-text output:

```bash
qualiax recording.wav --no-color
```

### JSON

Structured output with full metric metadata per file:

```json
[
  {
    "path": "recording.wav",
    "duration_s": 4.23,
    "sample_rate": 44100,
    "channels": 1,
    "metrics": [
      {
        "name": "Integrated Loudness (LUFS)",
        "value": -16.2,
        "unit": "LUFS",
        "group": "loudness",
        "higher_is_better": null,
        "warning": null
      }
    ]
  }
]
```

### CSV

Flat table with one row per file and one column per metric. Useful for batch analysis and downstream processing:

```bash
qualiax ./calls/ --output report.csv
```

---

## Hardware Acceleration

qualiax automatically detects and uses the best available compute backend:

1. **CUDA** — NVIDIA GPU via PyTorch
2. **MPS** — Apple Silicon (M1/M2/M3/M4) via PyTorch Metal
3. **CPU** — NumPy/SciPy fallback (always available)

No configuration needed. Install PyTorch to enable GPU acceleration:

```bash
pip install torch
```

Accelerated operations include: STFT, MFCC, autocorrelation, spectral feature extraction, K-weighting filter, and true peak detection.

To force a specific device in code:

```python
from qualiax.gpu import set_device
set_device("cpu")   # or "cuda" / "mps"
```

---

## Supported Formats

| Format | Extension | Requires |
|--------|-----------|----------|
| WAV | `.wav` | stdlib (always available) |
| FLAC | `.flac` | `soundfile` |
| OGG Vorbis | `.ogg` | `soundfile` |
| AIFF | `.aiff`, `.aif` | `soundfile` |
| MP3 | `.mp3` | `pydub` + ffmpeg |
| AAC | `.aac`, `.m4a` | `pydub` + ffmpeg |
| Opus | `.opus` | `pydub` + ffmpeg |

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

## License

MIT
