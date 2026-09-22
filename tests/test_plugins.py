import types

import pytest

from qualiax.models import FileResult, MetricResult
from qualiax.plugins import PLUGIN_API_VERSION, PluginManager, is_compatible_api
from qualiax.validation import validate_report_payload


class FakeEntryPoint:
    def __init__(self, name, target=None, error=None):
        self.name = name
        self.value = f"fake_plugins:{name}"
        self.group = "qualiax.plugins"
        self._target = target
        self._error = error

    def load(self):
        if self._error is not None:
            raise self._error
        return self._target


class FakeEntryPoints:
    def __init__(self, groups):
        self._groups = groups

    def select(self, group):
        return list(self._groups.get(group, []))


def _module(api=None, register=None):
    module = types.ModuleType("fake_plugin")
    if api is not None:
        module.QUALIAX_PLUGIN_API = api
    if register is not None:
        module.register = register
    return module


def _install(monkeypatch, entry_points, *, legacy_dict=False):
    groups = {"qualiax.plugins": entry_points}
    fake = groups if legacy_dict else FakeEntryPoints(groups)
    monkeypatch.setattr("qualiax.plugins.metadata.entry_points", lambda: fake)


def _register_snr_label(manager):
    manager.register_label_provider(
        "low-snr",
        lambda result: [{"id": "custom_low_snr"}] if any(m.name == "Estimated SNR" and m.value < 15 for m in result.metrics) else [],
    )
    manager.register_reporter("tsv", lambda results: "\n".join(result.path for result in results))


def test_discover_registers_plugins_from_entry_points(monkeypatch):
    def register_partial_then_fail(manager):
        manager.register_label_provider("half", lambda result: [])
        raise RuntimeError("boom during register")

    _install(
        monkeypatch,
        [
            FakeEntryPoint("snr", _module(api="1.0", register=_register_snr_label)),
            FakeEntryPoint("callable", lambda manager: manager.register_reporter("plain", lambda results: "plain")),
            FakeEntryPoint("missing-dep", error=ImportError("No module named 'torch'")),
            FakeEntryPoint("future", _module(api="2.0", register=_register_snr_label)),
            FakeEntryPoint("newer-minor", _module(api="1.9", register=_register_snr_label)),
            FakeEntryPoint("half", _module(register=register_partial_then_fail)),
            FakeEntryPoint("not-callable", _module(api="1.0")),
        ],
    )
    manager = PluginManager()

    discovered = manager.discover()

    assert discovered == ["snr", "callable"]
    assert sorted(manager.label_providers) == ["low-snr"]  # "half" was rolled back
    assert sorted(manager.reporters) == ["plain", "tsv"]
    failures = {item["plugin"]: item for item in manager.discovery_failures}
    assert set(failures) == {"missing-dep", "future", "newer-minor", "half", "not-callable"}
    assert failures["missing-dep"]["stage"] == "load"
    assert failures["future"]["stage"] == "version"
    assert "2.0" in failures["future"]["error"]
    assert failures["half"]["stage"] == "register"
    assert manager.plugins["snr"]["api_version"] == "1.0"
    assert manager.plugins["callable"]["api_version"] is None


def test_discover_supports_python39_entry_point_dicts(monkeypatch):
    _install(monkeypatch, [FakeEntryPoint("snr", _module(register=_register_snr_label))], legacy_dict=True)
    manager = PluginManager()
    assert manager.discover() == ["snr"]
    assert manager.discover("other.group") == []


def test_is_compatible_api():
    assert PLUGIN_API_VERSION == "1.0"
    assert is_compatible_api("1.0") is True
    assert is_compatible_api("1") is True
    assert is_compatible_api("1.1") is False
    assert is_compatible_api("0.9") is False
    assert is_compatible_api("2.0") is False
    assert is_compatible_api("banana") is False


def test_run_label_providers_isolates_failures_and_bad_shapes():
    manager = PluginManager()
    manager.register_label_provider("good", lambda result: [{"id": "custom"}])
    manager.register_label_provider("raises", lambda result: 1 / 0)
    manager.register_label_provider("bad-shape", lambda result: [{"severity": "warn"}])

    labels, failures = manager.run_label_providers(FileResult(path="a.wav"))

    assert labels == [{"id": "custom"}]
    assert {item["plugin"] for item in failures} == {"raises", "bad-shape"}


def test_apply_label_providers_merges_labels_and_records_diagnostics():
    manager = PluginManager()
    _register_snr_label(manager)
    manager.register_label_provider("crashy", lambda result: {}["missing"])
    noisy = FileResult(path="noisy.wav", metrics=[MetricResult(name="Estimated SNR", value=9.0, group="noise")])
    noisy.insights = {"defect_labels": [{"id": "noisy_floor", "severity": "fail"}]}
    clean = FileResult(path="clean.wav", metrics=[MetricResult(name="Estimated SNR", value=30.0, group="noise")])

    summary = manager.apply_label_providers([noisy, clean])

    assert summary == {"labels": 1, "failures": 2}
    assert noisy.insights["defect_labels"][1] == {
        "id": "custom_low_snr",
        "severity": "warn",
        "confidence": None,
        "evidence": "",
        "evidence_metrics": [],
        "source": "plugin:low-snr",
    }
    assert "defect_labels" not in clean.insights
    diagnostic = noisy.diagnostics[0]
    assert (diagnostic.code, diagnostic.source, diagnostic.context) == (
        "plugin_label_provider_failed",
        "plugin",
        {"plugin": "crashy"},
    )
    assert validate_report_payload([noisy.to_dict(), clean.to_dict()]) == []


def test_conformance_report_checks_label_shape_and_discovery(monkeypatch):
    _install(monkeypatch, [FakeEntryPoint("broken", error=ImportError("nope"))])
    manager = PluginManager()
    manager.discover()
    manager.register_label_provider("good", lambda result: [{"id": "x"}])
    manager.register_label_provider("ints", lambda result: [1, 2])
    manager.register_label_provider("raises", lambda result: 1 / 0)
    manager.register_reporter("text", lambda results: "ok")
    manager.register_reporter("bytes", lambda results: b"ok")

    report = manager.conformance_report()

    assert report["api_version"] == "1.0"
    assert report["label_providers"]["good"] == "ok"
    assert report["label_providers"]["ints"].startswith("invalid")
    assert report["label_providers"]["raises"].startswith("error")
    assert report["reporters"] == {"text": "ok", "bytes": "invalid: reporter must return str"}
    assert report["discovery_failures"][0]["plugin"] == "broken"
    assert report["passed"] is False

    clean = PluginManager()
    clean.register_label_provider("good", lambda result: [])
    assert clean.conformance_report()["passed"] is True


def test_register_and_render_validation():
    manager = PluginManager()
    with pytest.raises(TypeError):
        manager.register_label_provider("x", "not callable")
    with pytest.raises(ValueError):
        manager.register_reporter("  ", lambda results: "")
    manager.register_reporter("Upper", lambda results: str(len(results)))
    assert manager.render("upper", [FileResult(path="a")]) == "1"
    with pytest.raises(KeyError):
        manager.render("missing", [])
    manager.register_reporter("bad", lambda results: 1)
    with pytest.raises(TypeError):
        manager.render("bad", [])
