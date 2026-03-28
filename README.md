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

### basic

Core signal properties derived directly from the waveform.

$$x_{\text{rms}} = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} x[n]^2}$$

where $x[n]$ is the discrete audio sample at index $n$, and $N$ is the total number of samples in the signal.

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

where $x_K(t)$ is the audio signal after passing through the K-weighting filter $H_K(s)$, and $T$ is the total duration in seconds. The $-0.691$ offset aligns the scale to LUFS (Loudness Units relative to Full Scale).

Gating is applied per BS.1770-4: absolute gate at $-70$ LUFS, relative gate at $\bar{L} - 10$ LU, where $\bar{L}$ is the ungated integrated loudness.

| Metric | Unit | Description |
|--------|------|-------------|
| Integrated Loudness | LUFS | Gated integrated loudness $L_K$. Streaming target: $-16$ to $-14$ LUFS |
| Loudness Range (LRA) | LU | $L_{\text{hi},95} - L_{\text{lo},10}$ over gated 3 s blocks (EBU R128) |
| Max Short-Term Loudness | LUFS | $\max_t L_K(t)$ over a 3 s sliding window |
| True Peak | dBTP | $L_{\text{TP}} = 20\log_{10}\!\left(\max_n \|x_{\uparrow 4}[n]\|\right)$ — 4× oversampled inter-sample peak. Streaming limit: $-1$ dBTP |

---

### spectral

Frequency-domain characteristics computed via STFT. Let $X_t[k]$ denote the magnitude spectrum at frame $t$, $f_k$ the center frequency of bin $k$ in Hz, $P_t[k] = |X_t[k]|^2$ the power at bin $k$ and frame $t$, $K$ the total number of frequency bins, and $\langle \cdot \rangle_t$ the mean over all STFT frames.

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

where $m$ is the frame index, $L$ is the frame length (25 ms = 400 samples at 16 kHz), $H$ is the hop size (10 ms = 160 samples at 16 kHz), and $M$ is the total number of frames. The per-frame energy in dB is $E_m^{dB} = 10\log_{10}(E_m)$. The VAD threshold is $\theta = \max(\text{P}_{30}(E^{dB}),\,-50\,\text{dBFS})$, where $\text{P}_{30}$ is the 30th percentile of all frame energies.

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

where $P_{\text{signal}}$ is the mean frame energy of the most active 50% of frames, and $P_{\text{noise}}$ is the mean frame energy of the quietest 10% of frames (the noise floor estimate $\hat{N}$).

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

where $m_j = \log(\mathbf{f}_j^\top \mathbf{p} + \varepsilon)$ is the log energy in Mel filter $j$, $\mathbf{f}_j$ is the $j$-th triangular Mel filter (spanning 80 Hz–8 kHz), $\mathbf{p}$ is the power spectrum of the frame, $M = 40$ is the number of Mel filters, $\varepsilon$ is a small constant for numerical stability, and $\sqrt{2/M}$ is the orthonormal DCT-II normalization factor.

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

where $\hat{\mathbf{x}}$ is the zero-mean test (degraded) signal vector, $\mathbf{r}$ is the zero-mean reference signal vector, $\alpha\mathbf{r}$ is the optimal target projection of $\hat{\mathbf{x}}$ onto $\mathbf{r}$, and $\hat{\mathbf{x}} - \alpha\mathbf{r}$ is the residual distortion component.

**SDR**:

$$\text{SDR} = 10\log_{10}\!\left(\frac{\|\mathbf{r}\|^2}{\|\mathbf{r} - \hat{\mathbf{x}}\|^2}\right)$$

where $\mathbf{r}$ is the reference signal vector and $\hat{\mathbf{x}}$ is the estimated/test signal vector, so $\mathbf{r} - \hat{\mathbf{x}}$ is the distortion residual.

**Log-Spectral Distance**:

$$\text{LSD} = \frac{1}{KT}\sum_{t,k} \left|\log|X_t[k]|^2 - \log|R_t[k]|^2\right|$$

where $X_t[k]$ is the STFT magnitude of the test signal at frame $t$ and bin $k$, $R_t[k]$ is the STFT magnitude of the reference signal, $K$ is the number of frequency bins, and $T$ is the number of frames.

**Cepstral Distance**:

$$\text{CD} = \frac{1}{T}\sum_t \sqrt{\sum_{k=1}^{13}(c_k(t) - c_k^{(r)}(t))^2}$$

where $c_k(t)$ is the $k$-th MFCC of the test signal at frame $t$, $c_k^{(r)}(t)$ is the $k$-th MFCC of the reference signal at frame $t$, and $T$ is the number of frames. The inner sum is the Euclidean distance in 13-dimensional cepstral space.

| Metric | Unit | Notes |
|--------|------|-------|
| Estimated MOS (non-intrusive proxy) | 1–5 | Heuristic estimate based on SNR and spectral shape |
| P.563 Proxy (NB Quality Estimate) | 1–4.5 | Narrowband non-intrusive quality proxy |
| PESQ (ITU-T P.862) | MOS-LQO | Requires `--reference`. True P.862 if `pesq` installed, else spectral proxy |
| STOI (Short-Time Objective Intelligibility) | 0–1 | Requires `--reference`. True STOI if `pystoi` installed |
| SI-SDR | dB | Requires `--reference` |
| Log-Spectral Distance | dB | Requires `--reference` |

---

### prosody

Per-frame F0 trajectory, perturbation measures, and speech rate. All metrics computed via normalized autocorrelation — no external dependencies.

**Normalized autocorrelation F0 detection:**

$$r_{xx}[\tau] = \frac{\sum_n x[n]\,x[n+\tau]}{\sum_n x[n]^2}, \quad \hat{f}_0 = \frac{f_s}{\hat{\tau}},\quad \hat{\tau} = \arg\max_{\tau \in [\tau_{\min},\tau_{\max}]} r_{xx}[\tau]$$

where $\tau$ is the lag in samples, $\hat{\tau}$ is the estimated fundamental period in samples, $\tau_{\min} = \lfloor f_s / f_{0,\max} \rfloor$ and $\tau_{\max} = \lfloor f_s / f_{0,\min} \rfloor$ with $f_{0,\min} = 60$ Hz and $f_{0,\max} = 500$ Hz. A frame is classified voiced when $r_{xx}[\hat{\tau}] > 0.40$.

**Jitter** (local period perturbation):

$$J = \frac{\frac{1}{N-1}\sum_{i=1}^{N-1}|T_i - T_{i-1}|}{\frac{1}{N}\sum_{i=1}^{N} T_i} \times 100\%$$

where $T_i = 1 / f_{0,i}$ is the fundamental period (in seconds) of the $i$-th consecutive voiced frame, and $N$ is the number of voiced frames in the sequence.

**Shimmer** (local amplitude perturbation):

$$S = \frac{\frac{1}{N-1}\sum_{i=1}^{N-1}|A_i - A_{i-1}|}{\frac{1}{N}\sum_{i=1}^{N} A_i} \times 100\%$$

where $A_i$ is the RMS amplitude of voiced frame $i$, and $N$ is the number of consecutive voiced frames.

| Metric | Unit | Description |
|--------|------|-------------|
| Voiced Frame Ratio | % | Fraction of frames with $r_{xx}[\hat\tau] > 0.40$ |
| F0 Mean | Hz | $\bar{f}_0 = \langle f_0(t) \rangle$ over voiced frames |
| F0 Std | Hz | Standard deviation of voiced F0 |
| F0 Min / Max | Hz | Extremes of voiced F0 |
| F0 Range | Hz | $f_{0,\max} - f_{0,\min}$ |
| Pitch Variability (CV) | % | $\sigma_{f_0} / \bar{f}_0 \times 100$ — higher = more expressive |
| F0 Slope | Hz/s | Linear regression of $f_0(t)$ over voiced frames |
| Jitter (Local) | % | Cycle-to-cycle F0 period perturbation. Normal: < 1% |
| Shimmer (Local) | % | Cycle-to-cycle amplitude perturbation. Normal: < 3% |
| Tremor Rate | Hz | Dominant spectral peak of F0 modulation in 2–15 Hz band |
| Tremor Depth | Hz² | Power of dominant F0 modulation component |
| Estimated Speech Rate | syll/s | Energy-envelope peak count / duration. Typical: 3–7 syll/s |

---

### psychoacoustic

Perceptual features based on auditory models. Uses Bark-scale critical-band analysis.

**Roughness** (Vassilakis 2001) — amplitude modulation between partial pairs:

$$R = \sum_{i<j} \left(\frac{A_i A_j}{A_i^2+A_j^2}\right)^{3.11} \cdot (A_i A_j)^{0.1} \cdot x^2 e^{-(x/0.25)^2}, \quad x = \frac{|f_j - f_i|}{\text{CBW}(f_i)}$$

where $A_i, A_j$ are the normalized amplitudes of the $i$-th and $j$-th spectral partials, $f_i < f_j$ are their frequencies in Hz, $\text{CBW}(f_i) = 25 + 75(1 + 1.4(f_i/1000)^2)^{0.69}$ is the critical bandwidth at $f_i$ (Zwicker 1961), and $x$ is the frequency difference normalized by the critical bandwidth. The roughness curve $x^2 e^{-(x/0.25)^2}$ peaks at $x = 0.25$ (Plomp & Levelt 1965).

**Sethares (1993) Sensory Dissonance:**

$$D = \sum_{i<j} A_i A_j \left(e^{-b_1 s |f_j - f_i|} - e^{-b_2 s |f_j - f_i|}\right), \quad s = \frac{0.24}{0.0207 f_i + 18.96}$$

where $b_1 = 3.5$ and $b_2 = 5.75$ are empirical constants fitted to the Plomp & Levelt (1965) consonance data, $f_i$ is the frequency of the lower partial in Hz, and $s$ is a frequency-dependent scaling factor that maps the physical frequency difference onto a perceptual dissonance curve.

**Sharpness** (Zwicker & Fastl 1990, Von Bismarck 1974):

$$S = 0.11 \frac{\sum_z N'(z)\,g(z)\,z}{\sum_z N'(z)}, \quad g(z) = \begin{cases}1 & z \leq 15\\ 0.066\,e^{0.171z} & z > 15\end{cases}$$

where $z$ is the Bark band index (0–24 Bark), $N'(z)$ is the specific loudness in Bark band $z$ (proportional to the square root of mean power in the band), and $g(z)$ is Von Bismarck's (1974) weighting function that increases the contribution of high-frequency bands above 15 Bark.

| Metric | Unit | Description |
|--------|------|-------------|
| Roughness | asper (rel.) | Vassilakis AM roughness model. High = grating/harsh |
| Sensory Dissonance | (rel.) | Sethares beating/clashing model. Low = consonant spectrum |
| Sharpness | acum | Zwicker high-frequency weighting. 1 acum = 1 kHz narrow-band noise |
| Spectral Flatness (SFM) | 0–1 | Geometric/arithmetic power mean ratio. 0 = sine, 1 = white noise |
| Tonality | 0–1 | $1 - \text{SFM}$. 1 = pure tone, 0 = noise-like |
| F0 (cepstral estimate) | Hz | Mean-spectrum cepstrum peak — cross-check of per-frame F0 |
| Harmonicity | 0–1 | Energy fraction at harmonic multiples of F0. 1 = purely harmonic |

---

### speaker

Voice characteristics and speaker demographics estimated from acoustic features. All estimates are heuristic, not biometric classifiers.

**LPC Formant Tracking** — order-12 LPC via Yule-Walker equations:

$$\mathbf{R}\,\mathbf{a} = -\mathbf{r}_{1:p}, \quad \mathbf{R}_{ij} = r[|i-j|]$$

where $\mathbf{R}$ is the $p \times p$ symmetric Toeplitz autocorrelation matrix with $\mathbf{R}_{ij} = r[|i-j|]$, $\mathbf{a} = [a_1, \ldots, a_p]^\top$ is the LPC coefficient vector, $\mathbf{r}_{1:p} = [r[1], \ldots, r[p]]^\top$ is the autocorrelation vector at lags 1 to $p$, $r[k] = \frac{1}{N}\sum_n x[n]\,x[n+k]$ is the biased autocorrelation at lag $k$, and $p = 12$ is the predictor order.

Formant frequencies are angular frequencies of LPC polynomial roots with positive imaginary part and bandwidth < 600 Hz:

$$F_k = \frac{\angle z_k \cdot f_s}{2\pi}, \quad \text{BW}_k = \frac{-\ln|z_k| \cdot f_s}{\pi}$$

where $z_k$ is the $k$-th complex root of the LPC polynomial $A(z) = 1 + a_1 z^{-1} + \cdots + a_p z^{-p}$, $\angle z_k$ is the argument of $z_k$ in radians, $|z_k|$ is its modulus, and $\text{BW}_k$ is the formant bandwidth. Only roots with $50 < F_k < 5500$ Hz and $\text{BW}_k < 600$ Hz are retained.

**Cepstral Peak Prominence** (Hillenbrand et al. 1994):

$$\text{CPP} = \max_q \bigl[c[q] - \hat{c}[q]\bigr]$$

where $c[q]$ is the real cepstrum at quefrency $q$ (samples), computed as $c = |\mathcal{F}^{-1}\{\log |X|^2\}|$, $\hat{c}[q]$ is a linear regression baseline fit to $c[q]$ over the quefrency range $[f_s/500,\, f_s/50]$ (corresponding to F0 range 50–500 Hz), and the peak is taken within that quefrency range.

**Gender estimation** (Traunmüller & Eriksson 1995 empirical distributions):

| F0 Mean | Estimate |
|---------|----------|
| < 145 Hz | male (adult) |
| 145–180 Hz | ambiguous overlap zone |
| 180–260 Hz | female (adult) |
| > 260 Hz | child / high soprano |

**Age estimation** — heuristic scoring from jitter, shimmer, HNR, spectral tilt, and pitch variability (Linville 2001; Xue & Deliyski 2001). Accuracy ± ~15 years.

| Metric | Unit | Description |
|--------|------|-------------|
| F1–F4 Formant Frequency | Hz | Median formant over voiced frames (LPC order 12) |
| Spectral Tilt | dB/oct | Power spectrum slope 100 Hz–Nyquist. Typical speech: −6 to −12 dB/oct |
| Cepstral Peak Prominence (CPP) | dB | Hillenbrand (1994). > 5 dB = modal voice. < 3 dB = breathy |
| Breathiness Index | 0–1 | CPP-derived. 0 = modal/clear, 1 = highly breathy |
| Creakiness (Vocal Fry) Ratio | % | Frames with autocorrelation peak in 20–80 Hz (vocal fry register) |
| Estimated Gender | — | Heuristic from F0 mean. 145–180 Hz = ambiguous overlap zone |
| Gender Confidence | 0–1 | Distance from overlap zone as proxy for certainty |
| Estimated Age Range | — | Heuristic from jitter, shimmer, HNR, tilt, pitch variability (± ~15 years) |

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
- ITU-T P.862 (2001). *Perceptual evaluation of speech quality (PESQ): An objective method for end-to-end speech quality assessment of narrow-band telephone networks and speech codecs.* International Telecommunication Union.
- ITU-T P.563 (2004). *Single-ended method for objective speech quality assessment in narrow-band telephony applications.* International Telecommunication Union.
- ITU-T P.56 (2011). *Objective measurement of active speech level.* International Telecommunication Union.

### Loudness & Level

- Williams, M. (1999). *Loudness normalisation: The EBU approach.* Proceedings of the AES 106th Convention.
- Lund, T. (2006). *Control of loudness in digital TV and radio.* EBU Technical Review, 2006(4).

### Perceptual Quality

- Rix, A. W., Beerends, J. G., Hollier, M. P., & Hekstra, A. P. (2001). Perceptual evaluation of speech quality (PESQ) — a new method for speech quality assessment of telephone networks and codecs. *ICASSP 2001*, 749–752.
- Taal, C. H., Hendriks, R. C., Heusdens, R., & Jensen, J. (2011). An algorithm for intelligibility prediction of time–frequency weighted noisy speech. *IEEE Transactions on Audio, Speech, and Language Processing*, 19(7), 2125–2136.
- Le Roux, J., Wisdom, S., Erdogan, H., & Hershey, J. R. (2019). SDR — half-baked or well done? *ICASSP 2019*, 626–630.
- Hu, G., & Loizou, P. C. (2008). Evaluation of objective quality measures for speech enhancement. *IEEE Transactions on Audio, Speech, and Language Processing*, 16(1), 229–238.

### Spectral Analysis

- Grey, J. M., & Gordon, J. W. (1978). Perceptual effects of spectral modifications on musical timbres. *Journal of the Acoustical Society of America*, 63(5), 1493–1500.
- Peeters, G. (2004). *A large set of audio features for sound description (similarity and classification) in the CUIDADO project.* IRCAM Technical Report.

### Psychoacoustics

- Plomp, R., & Levelt, W. J. M. (1965). Tonal consonance and critical bandwidth. *Journal of the Acoustical Society of America*, 38(4), 548–560.
- Sethares, W. A. (1993). Local consonance and the relationship between timbre and scale. *Journal of the Acoustical Society of America*, 94(3), 1218–1228.
- Vassilakis, P. N. (2001). *Perceptual and physical properties of amplitude fluctuation and their musical significance.* PhD dissertation, UCLA.
- Vassilakis, P. N. (2005). Auditory roughness as a means of musical expression. *Selected Reports in Ethnomusicology*, 12, 119–144.
- Zwicker, E., & Fastl, H. (1990). *Psychoacoustics: Facts and models.* Springer-Verlag.
- Von Bismarck, G. (1974). Sharpness as an attribute of the timbre of steady sounds. *Acustica*, 30(3), 159–172.
- Zwicker, E. (1961). Subdivision of the audible frequency range into critical bands (Frequenzgruppen). *Journal of the Acoustical Society of America*, 33(2), 248.
- Traunmüller, H. (1990). Analytical expressions for the tonotopic sensory scale. *Journal of the Acoustical Society of America*, 88(1), 97–100.

### Prosody & F0 Analysis

- Boersma, P. (1993). Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound. *Proceedings of the Institute of Phonetic Sciences*, 17, 97–110.
- De Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency estimator for speech and music. *Journal of the Acoustical Society of America*, 111(4), 1917–1930.
- Mermelstein, P. (1975). Automatic segmentation of speech into syllabic units. *Journal of the Acoustical Society of America*, 58(4), 880–883.
- Baken, R. J., & Orlikoff, R. F. (2000). *Clinical measurement of speech and voice* (2nd ed.). Singular Publishing.

### Voice Quality & Speaker Characteristics

- Hillenbrand, J., Cleveland, R. A., & Erickson, R. L. (1994). Acoustic correlates of breathy vocal quality. *Journal of Speech and Hearing Research*, 37(4), 769–778.
- Kreiman, J., & Sidtis, D. (2011). *Foundations of voice studies: An interdisciplinary approach to voice production and perception.* Wiley-Blackwell.
- Linville, S. E. (2001). *Vocal aging.* Singular Publishing Group.
- Xue, S. A., & Deliyski, D. D. (2001). Effects of aging on selected acoustic voice parameters: Preliminary normative data and educational implications. *Educational Gerontology*, 27(2), 159–168.
- Traunmüller, H., & Eriksson, A. (1995). *The frequency range of the voice fundamental in the speech of male and female adults.* Manuscript, Stockholm University.
- Hammarberg, B., Fritzell, B., Gauffin, J., Sundberg, J., & Wedin, L. (1980). Perceptual and acoustic correlates of abnormal voice qualities. *Acta Oto-Laryngologica*, 90(1–6), 441–451.

### LPC & Formant Tracking

- Markel, J. D., & Gray, A. H. (1976). *Linear prediction of speech.* Springer-Verlag.
- Proakis, J. G., & Manolakis, D. G. (2006). *Digital signal processing: Principles, algorithms, and applications* (4th ed.). Prentice Hall.
- Levinson, N. (1947). The Wiener (root mean square) error criterion in filter design and prediction. *Journal of Mathematics and Physics*, 25(1–4), 261–278.
- Durbin, J. (1960). The fitting of time-series models. *Revue de l'Institut International de Statistique*, 28(3), 233–244.

### Cepstral Analysis

- Noll, A. M. (1967). Cepstrum pitch determination. *Journal of the Acoustical Society of America*, 41(2), 293–309.
- Bogert, B. P., Healy, M. J. R., & Tukey, J. W. (1963). The quefrency alanysis of time series for echoes: Cepstrum, pseudo-autocovariance, cross-cepstrum and saphe cracking. *Proceedings of the Symposium on Time Series Analysis*, 209–243.

### MFCCs & Speech Features

- Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 28(4), 357–366.
- Logan, B. (2000). Mel frequency cepstral coefficients for music modeling. *ISMIR 2000*.

### Noise & SNR

- Boll, S. F. (1979). Suppression of acoustic noise in speech using spectral subtraction. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 27(2), 113–120.
- Loizou, P. C. (2007). *Speech enhancement: Theory and practice.* CRC Press.

### Software & Toolkits

- McFee, B., et al. (2015). librosa: Audio and music signal analysis in Python. *Proceedings of the 14th Python in Science Conference*, 18–25.
- Boersma, P., & Weenink, D. (2024). *Praat: Doing phonetics by computer* (version 6.4). Retrieved from https://www.praat.org/

---

## License

MIT
