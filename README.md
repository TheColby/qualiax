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

Results are automatically saved to `sample.json` alongside the source file.

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
- **Zero config** — works out of the box on any WAV file; results automatically saved to `filename.json`
- **Intrusive and non-intrusive modes** — run standalone or supply a reference file for difference-based metrics (PESQ, STOI, SI-SDR, SDR)
- **Multiple output formats** — colored console, JSON, CSV
- **Color progress bar** — live `[████░░░░] N/total filename` bar during batch analysis
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
chmod +x install.sh && ./install.sh
```

### Optional Dependencies

| Extra | Packages | What it enables |
|-------|----------|-----------------|
| `audio` | `soundfile`, `pydub` | FLAC, OGG, AIFF, MP3, AAC, M4A (requires ffmpeg for MP3/AAC) |
| `perceptual` | `pesq`, `pystoi` | True PESQ P.862 scores, true STOI intelligibility |
| `all` | all of the above + `librosa` | Full format support + perceptual metrics + universal fallback loader |

**ffmpeg** is required for MP3 and AAC/M4A decoding via pydub:

```bash
brew install ffmpeg        # macOS
apt install ffmpeg         # Ubuntu/Debian
```

**PyTorch** is optional but enables GPU acceleration on CUDA and Apple Silicon:

```bash
pip install torch
```

---

## Usage

### Basic

Analyze a single file. Results are printed to the terminal and automatically saved to `recording.json`:

```bash
qualiax recording.wav
```

### Directories

Recursively analyze every supported audio file in a directory. Each file gets its own `.json` alongside it:

```bash
qualiax ./recordings/
```

Multiple paths can be provided:

```bash
qualiax file1.wav file2.mp3 ./folder/
```

### Output Formats

Write all results to a specific file (overrides per-file JSON default):

```bash
qualiax ./calls/ --output results.json
qualiax ./calls/ --output results.csv
qualiax recording.wav --format json     # explicit format, print to stdout
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

```bash
qualiax recording.wav --metrics basic
qualiax recording.wav --metrics basic,loudness,spectral
qualiax recording.wav --metrics all      # default
```

Available groups: `basic`, `loudness`, `spectral`, `temporal`, `noise`, `speech`, `perceptual`

### Parallel Processing

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
  -v, --verbose                   Show analysis progress and warnings
  -w, --workers INTEGER           Parallel worker threads (default: 1)
  --help                          Show this message and exit
```

---

## Metric Groups

### basic

Core signal properties derived directly from the waveform.

$$x_{\text{rms}} = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} x[n]^2}$$

| Metric | Unit | Formula / Description |
|--------|------|-----------------------|
| Duration | s | $T = N / f_s$ |
| Sample Rate | Hz | $f_s$ |
| Channels | — | Mono / stereo channel count |
| Peak Amplitude | — | $P = \max_n \|x[n]\|$ |
| Peak Level | dBFS | $L_P = 20\log_{10}(P)$ |
| RMS Level | dBFS | $L_{\text{rms}} = 20\log_{10}(x_{\text{rms}})$ |
| Crest Factor | dB | $C = 20\log_{10}(P / x_{\text{rms}})$ — high values indicate dynamic or sparse signal |
| DC Offset | — | $\mu = \frac{1}{N}\sum_n x[n]$ — non-zero indicates a DC component |
| Silence Ratio | % | $S = \frac{1}{N}\|\{n : \|x[n]\| < 10^{-3}\}\| \times 100$ |
| Dynamic Range (simple) | dB | $L_P - L_{\text{rms}}$ |
| Clipping Detected | 0/1 | $\mathbf{1}[\exists\, n : \|x[n]\| \geq 0.999]$ |
| Zero Crossing Rate | crossings/sample | $\text{ZCR} = \frac{1}{N-1}\sum_{n=1}^{N-1}\mathbf{1}[\text{sgn}(x[n]) \neq \text{sgn}(x[n-1])]$ |

---

### loudness

Broadcast-standard loudness measurements per ITU-R BS.1770-4 / EBU R128.

The K-weighting filter $H_K(s)$ is a two-stage shelving filter applied before loudness integration:

$$L_K = -0.691 + 10\log_{10}\!\left(\frac{1}{T}\int_0^T \left|x_K(t)\right|^2 dt\right) \quad \text{LUFS}$$

Gating is applied per BS.1770-4: absolute gate at $-70$ LUFS, relative gate at $\bar{L} - 10$ LU.

| Metric | Unit | Description |
|--------|------|-------------|
| Integrated Loudness | LUFS | Gated integrated loudness $L_K$. Streaming target: $-16$ to $-14$ LUFS |
| Loudness Range (LRA) | LU | $L_{\text{hi},95} - L_{\text{lo},10}$ over gated 3 s blocks (EBU R128) |
| Max Short-Term Loudness | LUFS | $\max_t L_K(t)$ over a 3 s sliding window |
| True Peak | dBTP | $L_{\text{TP}} = 20\log_{10}\!\left(\max_n \|x_{\uparrow 4}[n]\|\right)$ — 4× oversampled inter-sample peak. Streaming limit: $-1$ dBTP |

---

### spectral

Frequency-domain characteristics computed via STFT. Let $X_t[k]$ denote the magnitude spectrum at frame $t$, $f_k$ the corresponding frequency, and $P_t[k] = |X_t[k]|^2$ the power.

| Metric | Unit | Formula |
|--------|------|---------|
| Spectral Centroid | Hz | $C = \left\langle\dfrac{\sum_k f_k P_t[k]}{\sum_k P_t[k]}\right\rangle_t$ |
| Spectral Bandwidth | Hz | $B = \left\langle\sqrt{\dfrac{\sum_k (f_k - C)^2 P_t[k]}{\sum_k P_t[k]}}\right\rangle_t$ |
| Spectral Rolloff (95%) | Hz | $R = \left\langle\min f : \sum_{k: f_k \leq f} P_t[k] \geq 0.95\sum_k P_t[k]\right\rangle_t$ |
| Spectral Flatness | dB | $F = 10\log_{10}\!\left\langle\dfrac{\exp\!\left(\frac{1}{K}\sum_k \ln P_t[k]\right)}{\frac{1}{K}\sum_k P_t[k]}\right\rangle_t$ — $0$ dB = white noise |
| Spectral Flux | — | $\Phi = \left\langle\sqrt{\sum_k (|X_t[k]| - |X_{t-1}[k]|)^2}\right\rangle_t$ |
| Spectral Skewness | — | Third standardized moment of $P_t[k]$ over $k$ |
| Spectral Entropy | bits | $H = \left\langle -\sum_k p_k \log_2 p_k\right\rangle_t$, $p_k = P_t[k] / \sum_j P_t[j]$ |
| Effective Bandwidth | Hz | Frequency range with energy $> -30$ dB of peak |
| Estimated F0 | Hz | $\hat{f}_0 = f_s / \hat{\tau}$, $\hat{\tau} = \arg\max_{\tau} R_{xx}[\tau]$ via autocorrelation |
| Band Energy: Sub-Bass (20–80 Hz) | % | $\sum_{k \in B} P[k] \;/\; \sum_k P[k]$ |
| Band Energy: Bass (80–300 Hz) | % | Same formula per band |
| Band Energy: Low-Mid (300–2000 Hz) | % | Same formula per band |
| Band Energy: Presence (2–6 kHz) | % | Same formula per band |
| Band Energy: Air (6–20 kHz) | % | Same formula per band |
| High-Frequency Content (HFC) | — | $\text{HFC} = \left\langle\sum_k f_k \cdot P_t[k]\right\rangle_t$ |

---

### temporal

Time-domain structure including speech activity and pause patterns. Frame energy:

$$E_m = \frac{1}{L}\sum_{n=0}^{L-1} x[mH + n]^2$$

where $L$ is frame length (25 ms) and $H$ is hop size (10 ms).

| Metric | Unit | Description |
|--------|------|-------------|
| Speech/Activity Ratio | % | $\frac{1}{M}\sum_m \mathbf{1}[E_m^{dB} > \theta] \times 100$, $\theta = \max(\text{P}_{30}(E^{dB}),\, -50\,\text{dB})$ |
| Attack Time | ms | Time to reach $0.9 \cdot \max_m E_m$ from start |
| Temporal Centroid | s | $\bar{t} = \sum_m t_m E_m \;/\; \sum_m E_m$ |
| Num Pauses (>100 ms) | — | Count of contiguous inactive frames spanning $> 100$ ms |
| Mean Pause Duration | s | Mean length of detected pauses |
| Max Pause Duration | s | Length of the longest detected pause |
| Energy Variance | dB² | $\text{Var}(E^{dB})$ — high = dynamic, low = monotone |
| ZCR Mean | crossings/sample | $\left\langle\text{ZCR}_m\right\rangle_m$ |
| ZCR Variance | — | $\text{Var}(\text{ZCR}_m)$ |

---

### noise

Noise floor and signal quality estimates.

$$\text{SNR} = 10\log_{10}\!\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right) \quad \text{dB}$$

Noise floor is estimated as the mean energy of the quietest 10% of frames.

| Metric | Unit | Formula / Description |
|--------|------|-----------------------|
| Estimated Noise Floor | dBFS | $\hat{N} = \left\langle E_m^{dB}\right\rangle_{m \in \text{quietest 10\%}}$ |
| Estimated SNR | dB | $\text{SNR} = \left\langle E_m^{dB}\right\rangle_{\text{active}} - \hat{N}$ |
| Spectral SNR | dB | $10\log_{10}(P_{\text{active}} / P_{\text{quiet}})$ in frequency domain |
| Harmonic-to-Noise Ratio (HNR) | dB | $\text{HNR} = 10\log_{10}(P_{\text{harmonic}} / P_{\text{noise}})$ — high = cleaner voiced speech |
| Near-Clipped Samples | count | $\|\{n : \|x[n]\| \geq 10^{-1/20}\}\|$ (within 1 dB of full scale) |
| Detected Dropouts | count | Sudden near-silence drops in otherwise active audio |

---

### speech

Speech-specific features computed on mono signal.

MFCCs are computed from the log Mel filterbank $\mathbf{m}$ via DCT:

$$c_k = \sqrt{\frac{2}{M}}\sum_{j=1}^{M} m_j \cos\!\left(\frac{\pi k (j - 0.5)}{M}\right), \quad k = 1, \ldots, 13$$

| Metric | Unit | Description |
|--------|------|-------------|
| MFCC-1 through MFCC-13 Mean | — | $\bar{c}_k = \frac{1}{T}\sum_t c_k(t)$ |
| MFCC-1 through MFCC-13 Std | — | $\sigma_k = \sqrt{\frac{1}{T}\sum_t (c_k(t) - \bar{c}_k)^2}$ |
| F1-Region Energy (300–1000 Hz) | % | Energy fraction in first formant band |
| F2-Region Energy (1–2.5 kHz) | % | Energy fraction in second formant band |
| F3-Region Energy (2.5–3.5 kHz) | % | Energy fraction in third formant band |
| Voiced/Unvoiced Ratio | — | Frames with $\text{ZCR} < \theta_{\text{zcr}}$ and $E > \theta_E$ divided by unvoiced frames |
| Speech-Band SNR (300 Hz – 3.4 kHz) | dB | $\text{SNR}$ restricted to the telephone/speech band |

---

### perceptual

Perceptual quality scores. Non-intrusive proxies are used by default; supply `--reference` to enable true intrusive metrics.

**SI-SDR** (scale-invariant signal-to-distortion ratio):

$$\text{SI-SDR} = 10\log_{10}\!\left(\frac{\|\alpha\mathbf{r}\|^2}{\|\hat{\mathbf{x}} - \alpha\mathbf{r}\|^2}\right), \quad \alpha = \frac{\hat{\mathbf{x}}^\top\mathbf{r}}{\|\mathbf{r}\|^2}$$

**SDR**:

$$\text{SDR} = 10\log_{10}\!\left(\frac{\|\mathbf{r}\|^2}{\|\mathbf{r} - \hat{\mathbf{x}}\|^2}\right)$$

**Log-Spectral Distance**:

$$\text{LSD} = \frac{1}{KT}\sum_{t,k} \left|\log|X_t[k]|^2 - \log|R_t[k]|^2\right|$$

**Cepstral Distance**:

$$\text{CD} = \frac{1}{T}\sum_t \sqrt{\sum_{k=1}^{13}(c_k(t) - c_k^{(r)}(t))^2}$$

| Metric | Unit | Notes |
|--------|------|-------|
| Estimated MOS (non-intrusive proxy) | 1–5 | Heuristic estimate based on SNR and spectral shape |
| P.563 Proxy (NB Quality Estimate) | 1–4.5 | Narrowband non-intrusive quality proxy |
| PESQ (ITU-T P.862) | MOS-LQO | Requires `--reference`. True P.862 if `pesq` installed, else spectral proxy |
| STOI (Short-Time Objective Intelligibility) | 0–1 | Requires `--reference`. True STOI if `pystoi` installed |
| SI-SDR | dB | Requires `--reference` |
| Log-Spectral Distance | dB | Requires `--reference` |

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

Results are automatically saved to `filename.json` alongside each source file. To override the path:

```bash
qualiax recording.wav --output custom.json
```

Structure:

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

Flat table with one row per file and one column per metric:

```bash
qualiax ./calls/ --output report.csv
```

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
