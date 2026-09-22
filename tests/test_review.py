import json

import pytest

from qualiax.review import REVIEW_DECISIONS, ReviewStore


def test_annotations_and_decisions_persist_with_attribution(tmp_path):
    path = tmp_path / "reviews" / "state.json"
    store = ReviewStore(path)
    store.annotate("call.wav", "Hum at 60 Hz", reviewer="alex")
    store.annotate("call.wav", "Dropout here", reviewer="sam", segment_index=3)
    store.decide("call.wav", "needs-review", reviewer="alex")
    store.decide("call.wav", "reject", reviewer="sam", note="dropout confirmed")

    entry = ReviewStore(path).for_file("call.wav")

    assert entry["decision"] == "reject"
    assert entry["decision_reviewer"] == "sam"
    assert [(item["decision"], item["reviewer"]) for item in entry["decision_history"]] == [
        ("needs-review", "alex"),
        ("reject", "sam"),
    ]
    assert entry["decision_history"][1]["note"] == "dropout confirmed"
    assert [item["reviewer"] for item in entry["annotations"]] == ["alex", "sam"]
    assert ReviewStore(path).segment_annotations("call.wav", 3)[0]["text"] == "Dropout here"


def test_two_stores_on_one_file_do_not_clobber_each_other(tmp_path):
    path = tmp_path / "reviews.json"
    first = ReviewStore(path)
    second = ReviewStore(path)

    first.annotate("a.wav", "from first", reviewer="alex")
    second.annotate("b.wav", "from second", reviewer="sam")
    first.decide("b.wav", "accept", reviewer="alex")

    on_disk = json.loads(path.read_text())
    assert on_disk["a.wav"]["annotations"][0]["text"] == "from first"
    assert on_disk["b.wav"]["annotations"][0]["text"] == "from second"
    assert on_disk["b.wav"]["decision"] == "accept"


def test_corrupt_store_is_never_overwritten(tmp_path):
    path = tmp_path / "reviews.json"
    path.write_text('{"a.wav": {"annotations": [', encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to overwrite"):
        ReviewStore(path)
    assert path.read_text(encoding="utf-8") == '{"a.wav": {"annotations": ['

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        ReviewStore(path)


def test_input_validation(tmp_path):
    store = ReviewStore(tmp_path / "reviews.json")
    with pytest.raises(ValueError):
        store.decide("a.wav", "approve", reviewer="alex")
    with pytest.raises(ValueError):
        store.decide("a.wav", "accept", reviewer="  ")
    with pytest.raises(ValueError):
        store.annotate("a.wav", "", reviewer="alex")
    with pytest.raises(ValueError):
        store.annotate("a.wav", "note", reviewer="alex", segment_index=-1)
    with pytest.raises(ValueError):
        store.annotate("a.wav", "note", reviewer="alex", segment_index=True)
    assert not (tmp_path / "reviews.json").exists()


def test_queries_and_summary(tmp_path):
    store = ReviewStore(tmp_path / "reviews.json")
    store.decide("a.wav", "accept", reviewer="alex")
    store.decide("b.wav", "reject", reviewer="alex")
    store.decide("c.wav", "needs-review", reviewer="alex")
    store.annotate("d.wav", "not decided yet", reviewer="sam")

    assert store.files() == ["a.wav", "b.wav", "c.wav", "d.wav"]
    assert store.files("needs-review") == ["c.wav"]
    assert store.files("undecided") == ["d.wav"]
    assert store.summary() == {
        "files": 4,
        "decisions": {"accept": 1, "reject": 1, "needs-review": 1, "undecided": 1},
    }
    assert REVIEW_DECISIONS == ("accept", "reject", "needs-review")
    with pytest.raises(ValueError):
        store.files("maybe")


def test_for_file_returns_a_copy(tmp_path):
    store = ReviewStore(tmp_path / "reviews.json")
    store.annotate("a.wav", "note", reviewer="alex")

    snapshot = store.for_file("a.wav")
    snapshot["annotations"].append({"text": "tampered"})

    assert len(store.for_file("a.wav")["annotations"]) == 1
    assert store.for_file("never-seen.wav") == {"annotations": [], "decision": None}


def test_portable_json_export(tmp_path):
    store_path = tmp_path / "reviews.json"
    store = ReviewStore(store_path)
    store.annotate("a.wav", "clip at 0:03", reviewer="alex", segment_index=1)
    store.decide("a.wav", "reject", reviewer="alex")

    exported = store.export_json(tmp_path / "export" / "review-export.json")
    document = json.loads(exported.read_text())

    assert document["format"] == "qualiax-review"
    assert document["version"] == 1
    assert document["summary"]["decisions"]["reject"] == 1
    assert document["files"] == store.export()
    assert document["files"]["a.wav"]["annotations"][0]["segment_index"] == 1
    with pytest.raises(ValueError):
        store.export_json(store_path)
    assert json.loads(store_path.read_text())["a.wav"]["decision"] == "reject"
