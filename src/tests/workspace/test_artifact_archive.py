"""Direct tests for `workspace.artifact_archive`.

Each test pins one property the shared artifact-signature convention promises
to be machine-checkable: stable across processes, insensitive to key order,
and different whenever the signature differs. `resolve_candidate_dir`'s own
tests pin the shape it composes on top of `compute_artifact_signature`.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path
from threading import Barrier, Thread

import polars as pl

from workspace.artifact_archive import (
    DEFAULT_SIGNATURE_DIGEST_SIZE,
    MANIFEST_FILENAME,
    advisory_artifact_lock,
    artifact_is_complete,
    begin_artifact_write,
    commit_artifact_write,
    compute_artifact_signature,
    read_artifact_batches,
    resolve_candidate,
    resolve_candidate_dir,
    write_artifact_batch,
)


def test_signature_is_stable_across_calls():
    # Not literally a second process, but pinning a known digest is what
    # actually proves `hashlib.blake2b` (unsalted) drives this rather than
    # Python's per-process-salted `hash()`: a salted implementation could
    # never reproduce a value pinned ahead of time.
    settings = {"vocab_size": 25000, "min_frequency": 1, "trainer": "wordpiece"}
    assert compute_artifact_signature(settings) == "2556f9127f418b1107dcbc5a85d81382"


def test_signature_is_insensitive_to_key_order():
    first = {"a": 1, "b": 2, "c": 3}
    second = {"c": 3, "a": 1, "b": 2}
    assert compute_artifact_signature(first) == compute_artifact_signature(second)


def test_signature_differs_whenever_settings_differ():
    base = {"vocab_size": 25000, "min_frequency": 1}
    changed = {"vocab_size": 25001, "min_frequency": 1}
    assert compute_artifact_signature(base) != compute_artifact_signature(changed)


def test_signature_is_a_real_digest_not_a_display_shortened_one():
    signature = compute_artifact_signature({"a": 1})
    assert len(signature) == DEFAULT_SIGNATURE_DIGEST_SIZE * 2
    assert all(char in "0123456789abcdef" for char in signature)


def test_signature_folds_input_content_hash_into_the_same_payload():
    # Decision: a signature covers the inputs as well as the settings, so
    # identical settings over a different corpus is a different artifact.
    same_settings = {"vocab_size": 25000}
    over_one_corpus = {**same_settings, "corpus_content_hash": "aaaa"}
    over_another_corpus = {**same_settings, "corpus_content_hash": "bbbb"}
    assert compute_artifact_signature(over_one_corpus) != compute_artifact_signature(
        over_another_corpus
    )


def test_signature_drops_a_setting_that_matches_its_declared_default():
    settings_before_new_field = {"vocab_size": 25000}
    settings_after_new_field_with_matching_default = {
        "vocab_size": 25000,
        "sp_byte_fallback": False,
    }
    assert compute_artifact_signature(settings_before_new_field) == (
        compute_artifact_signature(
            settings_after_new_field_with_matching_default,
            defaults={"sp_byte_fallback": False},
        )
    )


def test_signature_keeps_a_setting_that_deviates_from_its_declared_default():
    settings_matching_default = {"vocab_size": 25000, "sp_byte_fallback": False}
    settings_deviating_from_default = {"vocab_size": 25000, "sp_byte_fallback": True}
    assert compute_artifact_signature(
        settings_matching_default, defaults={"sp_byte_fallback": False}
    ) != compute_artifact_signature(
        settings_deviating_from_default, defaults={"sp_byte_fallback": False}
    )


def test_signature_ignores_a_default_for_a_key_settings_does_not_carry():
    with_key = {"vocab_size": 25000, "sp_byte_fallback": False}
    without_key = {"vocab_size": 25000}
    assert compute_artifact_signature(
        with_key, defaults={"sp_byte_fallback": False}
    ) == compute_artifact_signature(without_key, defaults={"sp_byte_fallback": False})


def test_resolve_candidate_dir_appends_the_signature_beneath_the_facets(
    tmp_path: Path,
):
    settings = {"vocab_size": 25000, "min_frequency": 1}
    candidate_dir = resolve_candidate_dir(
        tmp_path, "aabbccdd", "wordpiece", settings=settings
    )
    expected_signature = compute_artifact_signature(settings)
    assert candidate_dir == (tmp_path / "aabbccdd" / "wordpiece" / expected_signature)


def test_resolve_candidate_hands_back_the_digest_that_named_the_directory(
    tmp_path: Path,
):
    """A caller recording the digest records the one that resolved the
    directory, rather than hashing the same settings a second time and
    trusting the two to agree."""
    settings = {"vocab_size": 25000, "min_frequency": 1}

    resolved = resolve_candidate(tmp_path, "wordpiece", settings=settings)

    assert resolved.signature == compute_artifact_signature(settings)
    assert resolved.directory.name == resolved.signature
    assert resolved.directory == resolve_candidate_dir(
        tmp_path, "wordpiece", settings=settings
    )


def test_resolve_candidate_dir_with_no_facets_keys_directly_beneath_root(
    tmp_path: Path,
):
    settings = {"vocab_size": 25000}
    assert resolve_candidate_dir(tmp_path, settings=settings) == (
        tmp_path / compute_artifact_signature(settings)
    )


def test_resolve_candidate_dir_differs_when_settings_differ(tmp_path: Path):
    first = resolve_candidate_dir(tmp_path, "family", settings={"seed": 1})
    second = resolve_candidate_dir(tmp_path, "family", settings={"seed": 2})
    assert first != second
    assert first.parent == second.parent


def test_incomplete_candidate_is_not_a_hit(tmp_path: Path):
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "payload.txt").write_text("partial", encoding="utf-8")

    assert not artifact_is_complete(candidate_dir)


def test_concurrent_commits_converge_on_one_manifest(tmp_path: Path):
    candidate_dir = tmp_path / "candidate"
    barrier = Barrier(2)
    results: list[Path] = []

    def publish(value: str):
        temporary_dir = begin_artifact_write(candidate_dir)
        (temporary_dir / "value.txt").write_text(value, encoding="utf-8")
        barrier.wait()
        results.append(commit_artifact_write(candidate_dir, temporary_dir))

    threads = [Thread(target=publish, args=(value,)) for value in ("one", "two")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [candidate_dir, candidate_dir]
    assert artifact_is_complete(candidate_dir)
    assert (candidate_dir / MANIFEST_FILENAME).is_file()
    assert len(list(tmp_path.glob(".candidate.*"))) == 0


def test_repeated_batch_push_and_row_keys_merge_once(tmp_path: Path):
    candidate_dir = tmp_path / "candidate"
    first = pl.DataFrame({"row_id": [1, 2], "value": ["old", "two"]})
    second = pl.DataFrame({"row_id": [2, 3], "value": ["new", "three"]})

    write_artifact_batch(candidate_dir, first, batch_key="first")
    write_artifact_batch(candidate_dir, first, batch_key="first")
    write_artifact_batch(candidate_dir, second, batch_key="second")

    result = read_artifact_batches(candidate_dir, row_key="row_id").sort("row_id")
    assert result.to_dicts() == [
        {"row_id": 1, "value": "old"},
        {"row_id": 2, "value": "new"},
        {"row_id": 3, "value": "three"},
    ]


def test_dead_lock_does_not_delay_builder(tmp_path: Path):
    lock_path = tmp_path / "candidate.lock"
    lock_path.write_text(
        '{"host": "'
        + socket.gethostname()
        + '", "pid": 999999999, "started_at": '
        + str(time.time())
        + "}",
        encoding="utf-8",
    )

    started = time.monotonic()
    with advisory_artifact_lock(lock_path, wait_seconds=1.0) as acquired:
        assert acquired
    assert time.monotonic() - started < 1.0
