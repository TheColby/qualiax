from qualiax.dataset_intelligence import audit_dataset
from qualiax.models import FileResult, MetricResult


def _result(path, *, signature=None, labels=(), content="speech", source_file=None, error=None, speaker_metric=None):
    insights = {}
    if signature is not None:
        insights["quality_fingerprint"] = {"signature": signature}
    insights["defect_labels"] = [{"id": label} for label in labels]
    metrics = [MetricResult(name="Estimated SNR", value=20.0, group="noise")]
    if speaker_metric is not None:
        metrics.append(MetricResult(name="Speaker Cluster", value=speaker_metric, group="speaker"))
    return FileResult(
        path=path,
        source_file=source_file,
        content_type=content,
        metrics=metrics,
        insights=insights if error is None else {},
        error=error,
    )


def test_speaker_mapping_drives_distribution_imbalance_and_speaker_leakage():
    results = [
        _result("corpus/train/a.wav", signature="s1"),
        _result("corpus/train/b.wav", signature="s2"),
        _result("corpus/train/c.wav", signature="s3"),
        _result("corpus/test/d.wav", signature="s4"),
    ]
    speakers = {
        "corpus/train/a.wav": "alice",
        "corpus/train/b.wav": "alice",
        "corpus/train/c.wav": "alice",
        "corpus/test/d.wav": "alice",
    }

    audit = audit_dataset(results, speakers=speakers)

    assert audit["speaker_distribution"] == {"alice": 4}
    speaker_leaks = [item for item in audit["split_leakage"] if item["type"] == "speaker"]
    assert speaker_leaks == [
        {
            "type": "speaker",
            "speaker": "alice",
            "files": list(speakers),
            "splits": ["test", "train"],
        }
    ]
    actions = {item["action"]: item for item in audit["remediation_plan"]}
    assert actions["rebalance_speakers"]["dominant_speaker"] == "alice"
    assert "No speaker ids available" not in " ".join(audit["warnings"])


def test_speaker_callable_and_metric_fallback():
    results = [
        _result("a.wav", signature="1", speaker_metric="spk-9"),
        _result("b.wav", signature="2"),
    ]

    by_metric = audit_dataset(results)
    by_callable = audit_dataset(results, speakers=lambda result: result.path[0].upper())

    assert by_metric["speaker_distribution"] == {"spk-9": 1}
    assert by_callable["speaker_distribution"] == {"A": 1, "B": 1}


def test_split_is_inferred_from_nearest_directory_or_explicit_mapping():
    results = [
        _result("/home/validation/project/test/a.wav", signature="dup"),
        _result("/home/validation/project/train/b.wav", signature="dup"),
        _result("/data/unsplit/c.wav", signature="solo"),
    ]

    audit = audit_dataset(results)

    assert audit["split_distribution"] == {"test": 1, "train": 1, "unassigned": 1}
    assert audit["split_leakage"] == [
        {
            "type": "duplicate",
            "files": ["/home/validation/project/test/a.wav", "/home/validation/project/train/b.wav"],
            "splits": ["test", "train"],
        }
    ]

    explicit = audit_dataset(
        results,
        splits={"/home/validation/project/test/a.wav": "val", "/home/validation/project/train/b.wav": "validation"},
    )
    assert explicit["split_leakage"] == []
    assert explicit["split_distribution"]["validation"] == 2


def test_segments_of_one_file_are_not_near_duplicates_of_each_other():
    results = [
        _result("long.wav [segment 1/2]", signature="quiet", source_file="long.wav"),
        _result("long.wav [segment 2/2]", signature="quiet", source_file="long.wav"),
        _result("x.wav", signature="loud"),
        _result("y.wav", signature="loud"),
    ]

    audit = audit_dataset(results)

    assert audit["near_duplicate_groups"] == [["x.wav", "y.wav"]]
    dedupe = next(item for item in audit["remediation_plan"] if item["action"] == "deduplicate")
    assert dedupe["files"] == ["y.wav"]


def test_declared_label_mismatches_and_conflicting_duplicates():
    results = [
        _result("a.wav", signature="dup", content="music"),
        _result("b.wav", signature="dup", content="speech"),
        _result("c.wav", signature="other", content="speech"),
    ]

    audit = audit_dataset(
        results,
        expected_content_type="speech",
        declared_labels={"a.wav": "speech", "b.wav": "music", "c.wav": "speech"},
    )

    reasons = sorted((item["reason"], item.get("file")) for item in audit["label_mismatch_risks"])
    assert reasons == [
        ("conflicting_duplicate_labels", None),
        ("declared_label_mismatch", "a.wav"),
        ("declared_label_mismatch", "b.wav"),
        ("unexpected_content_type", "a.wav"),
    ]
    review = next(item for item in audit["remediation_plan"] if item["action"] == "review_content_labels")
    assert review["files"] == ["a.wav", "b.wav"]
    assert review["priority"] == 2


def test_remediation_plan_is_prioritized_and_lists_files():
    results = [
        _result("train/a.wav", signature="dup", labels=["noisy_floor"]),
        _result("test/a-copy.wav", signature="dup"),
        _result("train/broken.wav", error="decode failed"),
    ]

    audit = audit_dataset(results)
    plan = audit["remediation_plan"]

    assert [item["priority"] for item in plan] == sorted(item["priority"] for item in plan)
    actions = {item["action"]: item for item in plan}
    assert actions["fix_or_remove_unreadable"]["files"] == ["train/broken.wav"]
    assert actions["remove_split_leakage"]["files"] == ["test/a-copy.wav", "train/a.wav"]
    assert actions["repair_or_remove_quality_outliers"]["files"] == ["train/a.wav"]
    assert audit["failed_files"] == ["train/broken.wav"]
    assert audit["defect_distribution"] == {"noisy_floor": 1}


def test_audit_warns_when_insights_are_missing():
    results = [FileResult(path="a.wav", content_type="speech"), FileResult(path="b.wav", content_type="speech")]

    audit = audit_dataset(results)

    assert audit["near_duplicate_groups"] == []
    assert any("insights=True" in warning for warning in audit["warnings"])
    assert audit["coverage"]["fingerprints"] == 0
