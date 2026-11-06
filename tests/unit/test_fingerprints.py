
import pytest

from backend.cache.fingerprints import (FingerprintError,build_permission_fingerprint,build_evidence_fingerprint,)



# Permission fingerprint tests


def test_permission_fingerprint_is_deterministic():
    first = build_permission_fingerprint(["hr", "finance"])
    second = build_permission_fingerprint(["hr", "finance"])

    assert first == second


def test_permission_fingerprint_ignores_tag_order():
    first = build_permission_fingerprint(["hr", "finance"])
    second = build_permission_fingerprint(["finance", "hr"])

    assert first == second


def test_permission_fingerprint_removes_duplicate_tags():
    first = build_permission_fingerprint(["hr", "finance"])
    second = build_permission_fingerprint(
        ["hr", "finance", "hr", "finance"]
    )

    assert first == second


def test_permission_fingerprint_normalizes_case():
    first = build_permission_fingerprint(["HR", "Finance"])
    second = build_permission_fingerprint(["hr", "finance"])

    assert first == second


def test_permission_fingerprint_normalizes_whitespace():
    first = build_permission_fingerprint(["  hr  ", "finance"])
    second = build_permission_fingerprint(["hr", "finance"])

    assert first == second


def test_permission_fingerprint_normalizes_internal_whitespace():
    first = build_permission_fingerprint(["human   resources"])
    second = build_permission_fingerprint(["human resources"])

    assert first == second


def test_different_permissions_produce_different_fingerprints():
    first = build_permission_fingerprint(["hr"])
    second = build_permission_fingerprint(["finance"])

    assert first != second


@pytest.mark.parametrize(
    "permission_tags",
    [
        None,
        "",
        "hr",
        b"hr",
        [],
        ["hr", None],
        ["hr", 123],
        ["hr", ""],
        ["hr", "   "],
        [None],
        [123],
    ],
)
def test_rejects_invalid_permission_tags(permission_tags):
    with pytest.raises(FingerprintError):
        build_permission_fingerprint(permission_tags)



# Evidence fingerprint test data


@pytest.fixture
def sample_evidence():
    return [
        {
            "document_id": "hr-policy",
            "document_version": "a" * 64,
            "chunk_id": "chunk-001",
        },
        {
            "document_id": "leave-policy",
            "document_version": "b" * 64,
            "chunk_id": "chunk-002",
        },
    ]



# Evidence fingerprint tests


def test_evidence_fingerprint_is_deterministic(sample_evidence):
    first = build_evidence_fingerprint(sample_evidence)
    second = build_evidence_fingerprint(sample_evidence)

    assert first == second


def test_evidence_fingerprint_ignores_record_order(sample_evidence):
    first = build_evidence_fingerprint(sample_evidence)
    second = build_evidence_fingerprint(
        list(reversed(sample_evidence))
    )

    assert first == second


def test_evidence_fingerprint_removes_duplicate_records(sample_evidence):
    duplicated = sample_evidence + [sample_evidence[0]]

    first = build_evidence_fingerprint(sample_evidence)
    second = build_evidence_fingerprint(duplicated)

    assert first == second


def test_different_document_ids_produce_different_fingerprints(sample_evidence):
    modified = [record.copy() for record in sample_evidence]
    modified[0]["document_id"] = "another-document"

    first = build_evidence_fingerprint(sample_evidence)
    second = build_evidence_fingerprint(modified)

    assert first != second


def test_document_version_change_produces_different_fingerprint(sample_evidence):
    modified = [record.copy() for record in sample_evidence]
    modified[0]["document_version"] = "c" * 64

    first = build_evidence_fingerprint(sample_evidence)
    second = build_evidence_fingerprint(modified)

    assert first != second


def test_chunk_id_change_produces_different_fingerprint(sample_evidence):
    modified = [record.copy() for record in sample_evidence]
    modified[0]["chunk_id"] = "chunk-999"

    first = build_evidence_fingerprint(sample_evidence)
    second = build_evidence_fingerprint(modified)

    assert first != second


def test_empty_evidence_is_rejected():
    with pytest.raises(FingerprintError):
        build_evidence_fingerprint([])


@pytest.mark.parametrize(
    "evidence",
    [
        None,
        "",
        "invalid-evidence",
        [None],
        [123],
        [{}],
        [
            {
                "document_id": "hr-policy",
                "chunk_id": "chunk-001",
            }
        ],
        [
            {
                "document_version": "a" * 64,
                "chunk_id": "chunk-001",
            }
        ],
        [
            {
                "document_id": "hr-policy",
                "document_version": "a" * 64,
            }
        ],
    ],
)
def test_rejects_invalid_evidence_records(evidence):
    with pytest.raises(FingerprintError):
        build_evidence_fingerprint(evidence)


@pytest.mark.parametrize(
    "version",
    [
        "",
        "invalid",
        "a" * 63,
        "a" * 65,
        "g" * 64,
        "z" * 64,
    ],
)
def test_rejects_invalid_document_versions(version):
    evidence = [
        {
            "document_id": "hr-policy",
            "document_version": version,
            "chunk_id": "chunk-001",
        }
    ]

    with pytest.raises(FingerprintError):
        build_evidence_fingerprint(evidence)


def test_evidence_fingerprint_is_a_sha256_digest(sample_evidence):
    fingerprint = build_evidence_fingerprint(sample_evidence)

    assert len(fingerprint) == 64
    assert all(character in "0123456789abcdef" for character in fingerprint)


def test_permission_fingerprint_is_a_sha256_digest():
    fingerprint = build_permission_fingerprint(["hr", "finance"])

    assert len(fingerprint) == 64
    assert all(character in "0123456789abcdef" for character in fingerprint)
