import json
import wave
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from qualiax.cli import main
from qualiax.version import OUTPUT_SCHEMA_VERSION, __version__


def _write_wav(path: Path, sr: int = 16_000, duration_s: float = 0.5) -> None:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    pcm = np.clip(audio * 32767, -32768, 32767).astype("<i2")

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def test_cli_end_to_end_json_output_with_real_wav(tmp_path):
    wav_path = tmp_path / "tone.wav"
    out_path = tmp_path / "report.json"
    _write_wav(wav_path)

    runner = CliRunner()
    result = runner.invoke(main, [str(wav_path), "--output", str(out_path), "--silent"])

    assert result.exit_code == 0

    payload = json.loads(out_path.read_text())
    assert payload[0]["schema_version"] == OUTPUT_SCHEMA_VERSION
    assert payload[0]["tool_version"] == __version__
    assert payload[0]["file"].endswith("tone.wav")
    assert payload[0]["content_type"] in {"speech", "music", "noise", "silence", "mixed", "unknown"}
    assert isinstance(payload[0]["metrics"], list)


def test_cli_accepts_stdin_wav_and_labels_it_stdin(tmp_path):
    wav_path = tmp_path / "tone.wav"
    out_path = tmp_path / "stdin-report.json"
    _write_wav(wav_path)
    wav_bytes = wav_path.read_bytes()

    runner = CliRunner()
    result = runner.invoke(main, ["-", "--output", str(out_path), "--silent"], input=wav_bytes)

    assert result.exit_code == 0
    payload = json.loads(out_path.read_text())
    assert payload[0]["file"] == "stdin"


def test_cli_segment_mode_outputs_segment_metadata(tmp_path):
    wav_path = tmp_path / "tone.wav"
    out_path = tmp_path / "segments.json"
    _write_wav(wav_path, duration_s=1.2)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [str(wav_path), "--segment-seconds", "0.5", "--output", str(out_path), "--silent"],
    )

    assert result.exit_code == 0
    payload = json.loads(out_path.read_text())
    assert len(payload) == 3
    assert payload[0]["source_file"].endswith("tone.wav")
    assert payload[0]["segment_index"] == 1
    assert payload[0]["total_segments"] == 3
    assert payload[1]["segment_start_s"] == 0.5
