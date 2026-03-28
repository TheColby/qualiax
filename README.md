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
  - [prosody](#prosody)
  - [psychoacoustic](#psychoacoustic)
  - [speaker](#speaker)
- [Output](#output)
  - [Pretty (Console)](#pretty-console)
  - [JSON](#json)
  - [CSV](#csv)
- [Hardware Acceleration](#hardware-acceleration)
- [Supported Formats](#supported-formats)
- [Requirements](#requirements)
- [References](#references)
- [License](#license)

---

## Features

- **90+ metrics** organized into 10 groups: basic signal properties, loudness (EBU R128 / BS.1770), spectral analysis, temporal structure, noise, speech characteristics, perceptual quality, prosody (F0 trajectory, jitter, shimmer, tremor), psychoacoustic (roughness, dissonance, sharpness, harmonicity), and speaker characteristics (formants, gender, age, voice quality)
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

Installs ffmpeg (brew/apt/dnf/pacman), all Python dependencies, and PyTorch. Make it executable first if needed:

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

Available groups: `basic`, `loudness`, `spectral`, `temporal`, `noise`, `speech`, `perceptual`, `prosody`, `psychoacoustic`, `speaker`

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

---

### basic

Core signal properties derived directly from the waveform. Let $x[n]$ denote the discrete audio sample at index $n$, and $N$ the total number of samples.

$$x_\text{rms} = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} x[n]^2}$$

$$L_P = 20\log_{10}\!\left(\max_n \lvert x[n] \rvert\right) \qquad \text{Peak Level (dBFS)}$$

$$L_\text{rms} = 20\log_{10}(x_\text{rms}) \qquad \text{RMS Level (dBFS)}$$

$$C = 20\log_{10}\!\left(\frac{\max_n \lvert x[n] \rvert}{x_\text{rms}}\right) \qquad \text{Crest Factor (dB)}$$

$$\mu = \frac{1}{N}\sum_{n=0}^{N-1} x[n] \qquad \text{DC Offset}$$

$$\text{ZCR} = \frac{1}{N-1}\sum_{n=1}^{N-1}\mathbf{1}\bigl[\operatorname{sgn}(x[n]) \neq \operatorname{sgn}(x[n-1])\bigr] \qquad \text{(crossings/sample)}$$

| Metric | Unit | Description |
|--------|------|-------------|
| Duration | s | $T = N / f_s$, where $f_s$ is the sample rate |
| Sample Rate | Hz | $f_s$ |
| Channels | — | Mono / stereo channel count |
| Peak Amplitude | — | $\max_n \lvert x[n] \rvert$ |
| Peak Level | dBFS | $L_P$ — 0 dBFS is full scale |
| RMS Level | dBFS | $L_\text{rms}$ |
| Crest Factor | dB | $C$ — high = dynamic or sparse signal |
| DC Offset | — | $\mu$ — non-zero indicates a DC bias |
| Silence Ratio | % | Fraction of samples with $\lvert x[n] \rvert < 10^{-3}$ |
| Dynamic Range (simple) | dB | $L_P - L_\text{rms}$ |
| Clipping Detected | 0/1 | 1 if any sample has $\lvert x[n] \rvert \geq 0.999$ |
| Zero Crossing Rate | crossings/sample | ZCR — higher = noisier or fricative-heavy |

---

### loudness

Broadcast-standard loudness measurements per ITU-R BS.1770-4 / EBU R128. The K-weighting filter $H_K(s)$ is a two-stage shelving filter applied before integration.

$$L_K = -0.691 + 10\log_{10}\!\left(\frac{1}{T}\int_0^T \lvert x_K(t) \rvert^2 \, dt\right) \qquad \text{LUFS}$$

where $x_K(t)$ is the signal after K-weighting, $T$ is duration in seconds, and $-0.691$ aligns the scale to LUFS. Gating per BS.1770-4: absolute gate at $-70$ LUFS; relative gate at $\bar{L} - 10$ LU, where $\bar{L}$ is the ungated integrated loudness.

$$L_\text{TP} = 20\log_{10}\!\left(\max_n \lvert x_{\uparrow 4}[n] \rvert\right) \qquad \text{dBTP}$$

where $x_{\uparrow 4}$ is the signal upsampled 4× to capture inter-sample peaks.

| Metric | Unit | Description |
|--------|------|-------------|
| Integrated Loudness | LUFS | Gated $L_K$. Streaming target: $-16$ to $-14$ LUFS |
| Loudness Range (LRA) | LU | $L_{\text{hi},95} - L_{\text{lo},10}$ over gated 3 s blocks (EBU R128) |
| Max Short-Term Loudness | LUFS | $\max_t L_K(t)$ over a sliding 3 s window |
| True Peak | dBTP | $L_\text{TP}$ — 4× oversampled inter-sample peak. Streaming limit: $-1$ dBTP |

---

### spectral

Frequency-domain features computed via STFT. Notation: $X_t[k]$ = complex STFT coefficient at frame $t$ and bin $k$; $P_t[k] = \lvert X_t[k] \rvert^2$ = power; $f_k$ = center frequency of bin $k$ in Hz; $K$ = number of bins; $\langle \cdot \rangle_t$ = mean over all frames.

**Centroid and Bandwidth:**

$$C_t = \frac{\sum_k f_k \, P_t[k]}{\sum_k P_t[k]}, \qquad C = \langle C_t \rangle_t$$

$$B_t = \sqrt{\frac{\sum_k (f_k - C_t)^2 \, P_t[k]}{\sum_k P_t[k]}}, \qquad B = \langle B_t \rangle_t$$

**Spectral Rolloff** (95th-percentile energy frequency):

$$R_t = \min\Bigl\{ f : \sum_{k:\, f_k \leq f} P_t[k] \;\geq\; 0.95 \sum_k P_t[k] \Bigr\}, \qquad R = \langle R_t \rangle_t$$

**Spectral Flatness** (geometric-to-arithmetic power mean ratio):

$$F_t = \frac{\exp\!\bigl(\tfrac{1}{K}\sum_k \ln P_t[k]\bigr)}{\tfrac{1}{K}\sum_k P_t[k]}, \qquad F_\text{dB} = 10\log_{10}\langle F_t \rangle_t$$

$F_\text{dB} = 0$ dB corresponds to white noise; more negative values indicate tonal content.

**Spectral Flux:**

$$\Phi_t = \sqrt{\sum_k \bigl(\lvert X_t[k] \rvert - \lvert X_{t-1}[k] \rvert\bigr)^2}, \qquad \Phi = \langle \Phi_t \rangle_t$$

**Spectral Entropy:**

$$H_t = -\sum_k p_{t,k} \log_2 p_{t,k}, \quad p_{t,k} = \frac{P_t[k]}{\sum_j P_t[j]}, \qquad H = \langle H_t \rangle_t \quad \text{bits}$$

**Estimated F0** (autocorrelation):

$$\hat{f}_0 = \frac{f_s}{\hat{\tau}}, \qquad \hat{\tau} = \operatorname*{arg\,max}_{\tau \in [\tau_\min, \tau_\max]} R_{xx}[\tau]$$

where $R_{xx}[\tau] = \sum_n x[n]\,x[n+\tau]$ is the autocorrelation at lag $\tau$.

**Band Energy** in band $B$:

$$E_B = \frac{\sum_{k:\, f_k \in B} P[k]}{\sum_k P[k]} \times 100\%$$

| Metric | Unit | Description |
|--------|------|-------------|
| Spectral Centroid | Hz | $C$ — center of mass of the power spectrum |
| Spectral Bandwidth | Hz | $B$ — weighted std dev around centroid |
| Spectral Rolloff (95%) | Hz | $R$ — frequency below which 95% of energy falls |
| Spectral Flatness | dB | $F_\text{dB}$ — 0 dB = white noise; more negative = tonal |
| Spectral Flux | — | $\Phi$ — mean frame-to-frame magnitude change |
| Spectral Skewness | — | Third standardized moment of $P_t[k]$ over $k$ |
| Spectral Entropy | bits | $H$ — 0 = single tone; $\log_2 K$ = white noise |
| Effective Bandwidth | Hz | Frequency span with energy above $-30$ dB of peak |
| Estimated F0 | Hz | $\hat{f}_0$ via autocorrelation peak |
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

where $L$ = frame length (25 ms), $H$ = hop size (10 ms), $M$ = total number of frames, and $E_m^\text{dB} = 10\log_{10}(E_m)$. The VAD threshold is $\theta = \max\!\bigl(\text{P}_{30}(E^\text{dB}),\, -50\,\text{dBFS}\bigr)$, where $\text{P}_{30}$ is the 30th percentile of all frame energies.

**Temporal Centroid:**

$$\bar{t} = \frac{\sum_m t_m \, E_m}{\sum_m E_m}$$

where $t_m = m H / f_s$ is the time of frame $m$ in seconds.

| Metric | Unit | Description |
|--------|------|-------------|
| Speech/Activity Ratio | % | $\frac{1}{M}\sum_m \mathbf{1}[E_m^\text{dB} > \theta] \times 100$ |
| Attack Time | ms | Time to reach $0.9 \cdot \max_m E_m$ from start |
| Temporal Centroid | s | $\bar{t}$ — energy-weighted mean time |
| Num Pauses (>100 ms) | — | Count of contiguous inactive runs longer than 100 ms |
| Mean Pause Duration | s | Mean length of detected pauses |
| Max Pause Duration | s | Length of the longest detected pause |
| Energy Variance | dB² | $\operatorname{Var}(E^\text{dB})$ — high = dynamic, low = monotone |
| ZCR Mean | crossings/sample | $\langle \text{ZCR}_m \rangle_m$ |
| ZCR Variance | — | $\operatorname{Var}(\text{ZCR}_m)$ |

---

### noise

Noise floor and signal quality estimates. The noise floor $\hat{N}$ is the mean energy of the quietest 10% of frames.

$$\text{SNR} = 10\log_{10}\!\left(\frac{P_\text{signal}}{P_\text{noise}}\right) \;\text{dB}$$

where $P_\text{signal}$ is the mean frame power of the most active 50% of frames and $P_\text{noise}$ is the mean frame power of the quietest 10%.

$$\text{HNR} = 10\log_{10}\!\left(\frac{P_\text{harmonic}}{P_\text{aperiodic}}\right) \;\text{dB}$$

where $P_\text{harmonic}$ is the power at harmonic multiples of the estimated F0, and $P_\text{aperiodic}$ is the residual noise power.

| Metric | Unit | Description |
|--------|------|-------------|
| Estimated Noise Floor | dBFS | $\hat{N} = \langle E_m^\text{dB} \rangle$ over quietest 10% of frames |
| Estimated SNR | dB | $\langle E_m^\text{dB} \rangle_\text{active} - \hat{N}$ |
| Spectral SNR | dB | $10\log_{10}(P_\text{active} / P_\text{quiet})$ in the frequency domain |
| Harmonic-to-Noise Ratio (HNR) | dB | HNR — higher = cleaner voiced speech |
| Near-Clipped Samples | count | Samples within 1 dB of full scale |
| Detected Dropouts | count | Sudden near-silence drops in otherwise active audio |

---

### speech

Speech-specific features computed on a mono signal. MFCCs are computed from the log Mel filterbank via DCT-II:

$$c_k = \sqrt{\frac{2}{M}}\sum_{j=1}^{M} m_j \cos\!\left(\frac{\pi k (j - 0.5)}{M}\right), \quad k = 1, \ldots, 13$$

where $m_j = \log(\mathbf{f}_j^\top \mathbf{p} + \varepsilon)$ is the log energy in Mel filter $j$, $M = 40$ is the number of Mel filters spanning 80 Hz – 8 kHz, $\mathbf{p}$ is the frame power spectrum, and $\varepsilon$ prevents $\log(0)$.

MFCC statistics over $T$ frames:

$$\bar{c}_k = \frac{1}{T}\sum_{t=1}^T c_k(t), \qquad \sigma_k = \sqrt{\frac{1}{T}\sum_{t=1}^T \bigl(c_k(t) - \bar{c}_k\bigr)^2}$$

| Metric | Unit | Description |
|--------|------|-------------|
| MFCC 1–13 Mean | — | $\bar{c}_k$ per coefficient |
| MFCC 1–13 Std | — | $\sigma_k$ per coefficient |
| F1-Region Energy (300–1000 Hz) | % | $E_B$ in first formant band |
| F2-Region Energy (1–2.5 kHz) | % | $E_B$ in second formant band |
| F3-Region Energy (2.5–3.5 kHz) | % | $E_B$ in third formant band |
| Voiced/Unvoiced Ratio | — | Voiced frames (low ZCR + high energy) divided by unvoiced frames |
| Speech-Band SNR (300 Hz – 3.4 kHz) | dB | SNR restricted to the telephone/speech band |

---

### perceptual

Perceptual quality scores. Non-intrusive proxies are used by default; supply `--reference` to enable true intrusive metrics.

**SI-SDR** (scale-invariant signal-to-distortion ratio):

$$\text{SI-SDR} = 10\log_{10}\!\left(\frac{\lVert \alpha \mathbf{r} \rVert^2}{\lVert \hat{\mathbf{x}} - \alpha \mathbf{r} \rVert^2}\right), \qquad \alpha = \frac{\hat{\mathbf{x}}^\top \mathbf{r}}{\lVert \mathbf{r} \rVert^2}$$

where $\hat{\mathbf{x}}$ is the zero-mean test (degraded) signal vector, $\mathbf{r}$ is the zero-mean reference vector, $\alpha$ is the optimal projection scalar (minimizes distortion energy), and $\hat{\mathbf{x}} - \alpha \mathbf{r}$ is the residual distortion.

**SDR:**

$$\text{SDR} = 10\log_{10}\!\left(\frac{\lVert \mathbf{r} \rVert^2}{\lVert \mathbf{r} - \hat{\mathbf{x}} \rVert^2}\right)$$

**Log-Spectral Distance:**

$$\text{LSD} = \frac{1}{KT}\sum_{t,k} \left\lvert \log \lvert X_t[k] \rvert^2 - \log \lvert R_t[k] \rvert^2 \right\rvert$$

where $X_t[k]$ and $R_t[k]$ are the STFT coefficients of the test and reference signals, respectively.

**Cepstral Distance:**

$$\text{CD} = \frac{1}{T}\sum_{t=1}^T \sqrt{\sum_{k=1}^{13}\bigl(c_k(t) - c_k^{(r)}(t)\bigr)^2}$$

where $c_k(t)$ and $c_k^{(r)}(t)$ are the $k$-th MFCC of the test and reference signals at frame $t$.

| Metric | Unit | Notes |
|--------|------|-------|
| Estimated MOS (non-intrusive) | 1–5 | Heuristic from SNR and spectral shape |
| P.563 Proxy (NB Quality Estimate) | 1–4.5 | Narrowband non-intrusive quality proxy |
| PESQ (ITU-T P.862) | MOS-LQO | Requires `--reference`. True P.862 if `pesq` installed |
| STOI | 0–1 | Requires `--reference`. True STOI if `pystoi` installed |
| SI-SDR | dB | Requires `--reference` |
| Log-Spectral Distance | dB | Requires `--reference` |

---

### prosody

Per-frame F0 trajectory, perturbation measures, and speech rate. All computed via normalized autocorrelation — no external dependencies.

**Normalized autocorrelation F0 detection:**

$$r_{xx}[\tau] = \frac{\sum_n x[n]\, x[n+\tau]}{\sum_n x[n]^2}, \qquad \hat{f}_0 = \frac{f_s}{\hat{\tau}}, \qquad \hat{\tau} = \operatorname*{arg\,max}_{\tau \in [\tau_\min,\, \tau_\max]} r_{xx}[\tau]$$

where $\tau$ is lag in samples, $\tau_\min = \lfloor f_s / f_{0,\max} \rfloor$ and $\tau_\max = \lfloor f_s / f_{0,\min} \rfloor$ with $f_{0,\min} = 60$ Hz and $f_{0,\max} = 500$ Hz. A frame is voiced when $r_{xx}[\hat{\tau}] > 0.40$.

**Jitter** (local F0 period perturbation, Baken & Orlikoff 2000):

$$J = \frac{\dfrac{1}{N-1}\sum_{i=1}^{N-1} \lvert T_i - T_{i-1} \rvert}{\dfrac{1}{N}\sum_{i=1}^{N} T_i} \times 100\%$$

where $T_i = 1 / f_{0,i}$ is the fundamental period of the $i$-th consecutive voiced frame in seconds, and $N$ is the count of voiced frames.

**Shimmer** (local amplitude perturbation):

$$S = \frac{\dfrac{1}{N-1}\sum_{i=1}^{N-1} \lvert A_i - A_{i-1} \rvert}{\dfrac{1}{N}\sum_{i=1}^{N} A_i} \times 100\%$$

where $A_i$ is the RMS amplitude of voiced frame $i$.

| Metric | Unit | Description |
|--------|------|-------------|
| Voiced Frame Ratio | % | Fraction of frames with $r_{xx}[\hat{\tau}] > 0.40$ |
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

where $f$ is frequency in Hz.

**Roughness** (Vassilakis 2001) — amplitude modulation between spectral partial pairs:

$$R = \sum_{i < j} \left(\frac{A_i A_j}{A_i^2 + A_j^2}\right)^{3.11} (A_i A_j)^{0.1} \cdot x^2 e^{-(x/0.25)^2}, \qquad x = \frac{\lvert f_j - f_i \rvert}{\text{CBW}(f_i)}$$

where $A_i, A_j$ are normalized amplitudes of the partial pair, $f_i < f_j$ are their frequencies in Hz, $\text{CBW}(f_i) = 25 + 75\bigl(1 + 1.4(f_i/1000)^2\bigr)^{0.69}$ is the critical bandwidth at $f_i$ (Zwicker 1961), and the roughness curve peaks at $x = 0.25$ (Plomp & Levelt 1965).

**Sensory Dissonance** (Sethares 1993):

$$D = \sum_{i < j} A_i A_j \left(e^{-b_1 s \lvert f_j - f_i \rvert} - e^{-b_2 s \lvert f_j - f_i \rvert}\right), \qquad s = \frac{0.24}{0.0207 f_i + 18.96}$$

where $b_1 = 3.5$ and $b_2 = 5.75$ are empirical constants from Plomp & Levelt (1965), and $s$ is a frequency-dependent scaling factor for the lower partial $f_i$.

**Sharpness** (Zwicker & Fastl 1990; Von Bismarck 1974):

$$S_\text{acum} = 0.11 \frac{\sum_z N'(z)\, g(z)\, z}{\sum_z N'(z)}, \qquad g(z) = \begin{cases} 1 & z \leq 15 \\ 0.066\, e^{0.171 z} & z > 15 \end{cases}$$

where $z$ is Bark band index (0–24 Bark), $N'(z)$ is specific loudness in band $z$ (proportional to the square root of mean power in the band), and $g(z)$ is Von Bismarck's (1974) high-frequency weighting function.

**Spectral Flatness and Tonality:**

$$\text{SFM} = \frac{\exp\!\bigl(\tfrac{1}{K}\sum_k \ln P[k]\bigr)}{\tfrac{1}{K}\sum_k P[k]}, \qquad \text{Tonality} = 1 - \text{SFM}$$

| Metric | Unit | Description |
|--------|------|-------------|
| Roughness | asper (rel.) | $R$ — Vassilakis AM roughness. High = grating or harsh |
| Sensory Dissonance | (rel.) | $D$ — Sethares beating model. Low = consonant spectrum |
| Sharpness | acum | $S_\text{acum}$ — 1 acum = reference 1 kHz narrow-band noise |
| Spectral Flatness (SFM) | 0–1 | 0 = pure sine tone; 1 = white noise |
| Tonality | 0–1 | $1 - \text{SFM}$. High = tonal; low = noise-like |

---

### speaker

Voice characteristics and speaker demographics estimated from acoustic features. All estimates are heuristic approximations, not biometric classifiers.

**LPC Formant Tracking** — order-$p$ LPC via Levinson-Durbin on the Yule-Walker system:

$$\mathbf{R}\,\mathbf{a} = -\mathbf{r}_{1:p}, \qquad \mathbf{R}_{ij} = r[\lvert i - j \rvert]$$

where $\mathbf{R}$ is the $p \times p$ symmetric Toeplitz matrix, $\mathbf{a} = [a_1, \ldots, a_p]^\top$ is the LPC coefficient vector, $r[k] = \frac{1}{N}\sum_n x[n]\, x[n+k]$ is the biased autocorrelation at lag $k$, and $p = 12$.

Formant frequencies and bandwidths from LPC polynomial roots $z_k$ with $\operatorname{Im}(z_k) \geq 0$:

$$F_k = \frac{\angle z_k \cdot f_s}{2\pi}, \qquad \text{BW}_k = \frac{-\ln \lvert z_k \rvert \cdot f_s}{\pi}$$

where $A(z) = 1 + a_1 z^{-1} + \cdots + a_p z^{-p}$ is the LPC polynomial, $\angle z_k$ is the argument of root $z_k$, and $\lvert z_k \rvert$ is its modulus. Only roots with $50 < F_k < 5500$ Hz and $\text{BW}_k < 600$ Hz are retained.

**Cepstral Peak Prominence** (Hillenbrand et al. 1994):

$$\text{CPP} = \max_{q \in [q_\min,\, q_\max]} \bigl[ c[q] - \hat{c}[q] \bigr]$$

where $c[q] = \lvert \mathcal{F}^{-1}\{\log \lvert X \rvert^2\} \rvert$ is the real cepstrum at quefrency $q$ (samples), $\hat{c}[q]$ is a linear regression baseline over the quefrency range $[f_s / 500,\; f_s / 50]$ (F0 range 50–500 Hz), and CPP is the peak prominence above that baseline.

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
