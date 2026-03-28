"""
Quick smoke test using synthetic audio (no file I/O needed).
Run: python test_synthetic.py
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from qualiax.metrics import METRIC_GROUPS

# Synthetic speech-like audio: 3s, 16 kHz, voiced + unvoiced mixture
sr = 16000
t = np.linspace(0, 3, 3 * sr)
# Voiced segment: harmonic stack at 120 Hz
voiced = sum(np.sin(2 * np.pi * 120 * k * t) / k for k in range(1, 8))
# Add white noise floor
noise = np.random.randn(len(t)) * 0.05
audio = (voiced * 0.3 + noise).astype(np.float32)
audio /= np.max(np.abs(audio)) * 1.1  # normalize to ~-1 dBFS

# Reference: cleaner version
ref = (voiced * 0.3).astype(np.float32)
ref /= np.max(np.abs(ref)) * 1.1

print("=" * 70)
print("qualiax — Synthetic Audio Smoke Test")
print("=" * 70)

total_metrics = 0
errors = 0

for group_name, fn in METRIC_GROUPS.items():
    print(f"\n[{group_name.upper()}]")
    try:
        results = fn(audio, sr, ref_audio=ref, ref_sr=sr)
        for m in results:
            val = m.formatted_value()
            warn = f"  ⚠ {m.warning}" if m.warning else ""
            print(f"  {m.name:<50} {val:>12} {m.unit}{warn}")
            total_metrics += 1
    except Exception as e:
        print(f"  ERROR: {e}")
        errors += 1

print("\n" + "=" * 70)
print(f"✅ {total_metrics} metrics computed, {errors} group errors")
