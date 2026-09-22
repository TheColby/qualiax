import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from qualiax.migrations import QualiaxDeprecationWarning
from qualiax.models import FileResult, MetricResult
from qualiax.operations import (
    AlertPolicy,
    DriftHistory,
    RollingMetricWindow,
    deliver_alert,
    prometheus_metrics,
    send_webhook,
)


@pytest.fixture
def webhook_server():
    """Local-only HTTP endpoint on 127.0.0.1 that records JSON posts."""
    received = []
    state = {"status": 200}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            received.append(
                {
                    "path": self.path,
                    "content_type": self.headers.get("Content-Type"),
                    "body": json.loads(self.rfile.read(length).decode("utf-8")),
                }
            )
            self.send_response(state["status"])
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/hook", received, state
    finally:
        server.shutdown()
        server.server_close()


def _result(path, **values):
    metrics = [MetricResult(name=name, value=value, group="g") for name, value in values.items()]
    return FileResult(path=path, metrics=metrics)


def test_rolling_window_is_bounded_and_summarizes():
    with pytest.warns(QualiaxDeprecationWarning, match="DriftHistory.window") as caught:
        window = RollingMetricWindow(size=3)
    assert caught[0].filename == __file__
    for value in (1.0, 2.0, 3.0, 10.0):
        summary = window.add("snr", value)

    assert summary["count"] == 3
    assert summary["mean"] == pytest.approx(5.0)
    assert (summary["min"], summary["max"], summary["latest"]) == (2.0, 10.0, 10.0)
    assert window.summary("missing")["count"] == 0


def test_drift_history_persists_bounds_and_windows(tmp_path):
    path = tmp_path / "state" / "history.json"
    history = DriftHistory(path, max_entries=3)
    for index, value in enumerate([10.0, 11.0, 12.0, 13.0]):
        history.append("snr", value, timestamp=float(index))

    reloaded = DriftHistory(path, max_entries=3)

    assert [item["value"] for item in reloaded.entries("snr")] == [11.0, 12.0, 13.0]
    assert reloaded.window("snr", 2)["mean"] == pytest.approx(12.5)
    assert reloaded.metrics() == ["snr"]
    smaller = DriftHistory(path, max_entries=2)
    assert [item["value"] for item in smaller.entries("snr")] == [12.0, 13.0]


def test_record_results_skips_errors_and_non_numeric_values(tmp_path):
    history = DriftHistory(tmp_path / "history.json")
    ok = _result("a.wav", snr=20.0, peak=-3.0, label="speech", flag=True, broken=float("nan"))
    failed = FileResult(path="b.wav", error="decode failed", metrics=[MetricResult(name="snr", value=1.0)])

    added = history.record_results([ok, failed], timestamp=5.0)
    only_snr = history.record_results([ok], metrics=["snr"], timestamp=6.0)

    assert added == 2
    assert only_snr == 1
    assert [item["value"] for item in history.entries("snr")] == [20.0, 20.0]
    assert history.entries("label") == []
    assert history.entries("broken") == []
    on_disk = json.loads((tmp_path / "history.json").read_text())
    assert [item["timestamp"] for item in on_disk["snr"]] == [5.0, 6.0]


def test_check_drift_uses_persistent_window(tmp_path):
    history = DriftHistory(tmp_path / "history.json")
    assert history.check_drift("snr", 20.0)["status"] == "insufficient_history"
    for value in (20.0, 21.0, 19.0, 20.0, 20.5, 19.5):
        history.append("snr", value)

    steady = history.check_drift("snr", 20.2)
    dropped = history.check_drift("snr", 8.0)

    assert steady["drifted"] is False
    assert steady["status"] == "ok"
    assert dropped["drifted"] is True
    assert dropped["zscore"] < -3
    assert len(history.entries("snr")) == 6  # check_drift does not record


def test_corrupt_history_is_quarantined_not_overwritten(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="could not be read"):
        history = DriftHistory(path)
    history.append("snr", 1.0)

    assert (tmp_path / "history.json.corrupt").read_text(encoding="utf-8") == "{not json"
    assert json.loads(path.read_text())["snr"][0]["value"] == 1.0


def test_alert_policy_cooldown_persists_across_instances(tmp_path):
    state = tmp_path / "alerts.json"
    first = AlertPolicy(cooldown_seconds=60, state_path=state)
    assert first.should_emit("noise", now=100.0) is True
    assert first.should_emit("noise", now=130.0) is False

    second = AlertPolicy(cooldown_seconds=60, state_path=state)
    assert second.should_emit("noise", now=150.0) is False
    assert second.should_emit("noise", now=161.0) is True
    assert second.should_emit("clipping", now=161.0) is True
    assert AlertPolicy(cooldown_seconds=60).should_emit("noise", now=150.0) is True


def test_send_webhook_posts_json_to_local_server(webhook_server):
    url, received, _state = webhook_server

    status = send_webhook(url, {"metric": "snr", "value": float("nan"), "files": ("a.wav",)})

    assert status == 200
    assert received == [
        {
            "path": "/hook",
            "content_type": "application/json",
            "body": {"metric": "snr", "value": None, "files": ["a.wav"]},
        }
    ]


def test_send_webhook_rejects_non_http_urls(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("x")
    with pytest.raises(ValueError, match="http or https"):
        send_webhook(secret.as_uri(), {"a": 1})
    with pytest.raises(ValueError):
        send_webhook("ftp://127.0.0.1/hook", {"a": 1})


def test_send_webhook_raises_on_http_errors(webhook_server):
    url, _received, state = webhook_server
    state["status"] = 500
    with pytest.raises(urllib.error.HTTPError):
        send_webhook(url, {"a": 1})


def test_deliver_alert_suppresses_and_retries_after_failure(webhook_server):
    url, received, state = webhook_server
    policy = AlertPolicy(cooldown_seconds=300)

    state["status"] = 503
    failed = deliver_alert(url, {"alert": "drift"}, policy=policy, key="snr", now=1000.0)
    state["status"] = 200
    sent = deliver_alert(url, {"alert": "drift"}, policy=policy, key="snr", now=1001.0)
    suppressed = deliver_alert(url, {"alert": "drift"}, policy=policy, key="snr", now=1100.0)

    assert failed["sent"] is False and failed["suppressed"] is False and "503" in failed["error"]
    assert sent == {"key": "snr", "sent": True, "suppressed": False, "status": 200, "error": None}
    assert suppressed["suppressed"] is True
    assert len(received) == 2  # the failed attempt and the retry; the suppressed call never posted


def test_deliver_alert_reports_unreachable_endpoint(monkeypatch):
    def refuse(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("qualiax.operations.urllib.request.urlopen", refuse)
    policy = AlertPolicy(cooldown_seconds=300)

    result = deliver_alert("http://alerts.invalid/hook", {"a": 1}, policy=policy, key="k")

    assert result["sent"] is False
    assert "connection refused" in result["error"]
    assert policy.is_suppressed("k") is False
    rejected = deliver_alert("file:///etc/hosts", {"a": 1}, policy=policy, key="k")
    assert "http or https" in rejected["error"]


def test_prometheus_exposition_format():
    ok = FileResult(
        path='calls/"vip"\\a.wav',
        metrics=[
            MetricResult(name="Estimated SNR", value=21.5, group="noise"),
            MetricResult(name="Loudness Range (LRA)", value=float("nan"), group="loudness"),
            MetricResult(name="Peak", value=float("-inf"), group="basic"),
            MetricResult(name="Near-Clipped Samples", value=np.int64(3), group="noise"),
            MetricResult(name="Clipping Detected", value=False, group="basic"),
            MetricResult(name="Estimated Gender", value="male", group="speaker"),
            MetricResult(name="Max Short-Term Loudness", value=None, group="loudness"),
        ],
    )
    failed = FileResult(path="bad.wav", error="decode failed")

    text = prometheus_metrics([ok, ok, failed])
    lines = text.splitlines()

    file_label = 'calls/\\"vip\\"\\\\a.wav'
    assert lines[:2] == ["# HELP qualiax_metric Numeric audio quality metric.", "# TYPE qualiax_metric gauge"]
    assert f'qualiax_metric{{file="{file_label}",group="noise",metric="Estimated SNR"}} 21.5' in lines
    assert f'qualiax_metric{{file="{file_label}",group="loudness",metric="Loudness Range (LRA)"}} NaN' in lines
    assert f'qualiax_metric{{file="{file_label}",group="basic",metric="Peak"}} -Inf' in lines
    assert f'qualiax_metric{{file="{file_label}",group="noise",metric="Near-Clipped Samples"}} 3.0' in lines
    assert not any("Clipping Detected" in line or "Estimated Gender" in line or "Max Short-Term" in line for line in lines)
    assert sum(line.startswith("qualiax_metric{") for line in lines) == 4  # duplicate file is not repeated
    assert f'qualiax_analysis_error{{file="{file_label}"}} 0' in lines
    assert 'qualiax_analysis_error{file="bad.wav"} 1' in lines
    assert "# TYPE qualiax_analysis_error gauge" in lines
    assert text.endswith("\n")


def test_prometheus_escapes_newlines_in_labels():
    text = prometheus_metrics([_result("a\nb.wav", snr=1.0)])
    assert 'file="a\\nb.wav"' in text
