

import hashlib
import json
from collections.abc import Iterable, Mapping


class FingerprintError(ValueError):
    """Raised when permission or evidence data cannot be fingerprinted safely."""


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_permission_fingerprint(permission_tags: Iterable[str]) -> str:
    """Build a deterministic fingerprint for a permission-tag set."""

    if isinstance(permission_tags, (str, bytes)):
        raise FingerprintError(
            "Permission tags must be an iterable of tag strings, not a string."
        )

    try:
        tags = list(permission_tags)
    except TypeError as exc:
        raise FingerprintError(
            "Permission tags must be iterable."
        ) from exc

    if not tags:
        raise FingerprintError(
            "At least one permission tag is required."
        )

    normalized = set()

    for tag in tags:
        if not isinstance(tag, str) or not tag.strip():
            raise FingerprintError(
                "Every permission tag must be a non-empty string."
            )

        normalized.add(" ".join(tag.split()).casefold())

    return _sha256_json(sorted(normalized))


def build_evidence_fingerprint(evidence_records: Iterable[Mapping[str, str]]) -> str:
    """Fingerprint the exact document versions and chunks used as evidence.

    Each record must contain non-empty document_id, document_version (64-char
    SHA-256 hex), and chunk_id values. Duplicate records are removed and order
    is canonicalized so retrieval ordering does not change the fingerprint.
    """
    if isinstance(evidence_records, (str, bytes)):
        raise FingerprintError("Evidence records must be an iterable of mappings.")
    canonical: set[tuple[str, str, str]] = set()
    try:
        for record in evidence_records:
            if not isinstance(record, Mapping):
                raise FingerprintError("Every evidence record must be a mapping.")
            document_id = record.get("document_id")
            document_version = record.get("document_version")
            chunk_id = record.get("chunk_id")
            if not all(isinstance(value, str) and value.strip() for value in (
                document_id, document_version, chunk_id
            )):
                raise FingerprintError(
                    "Each evidence record requires document_id, document_version and chunk_id."
                )
            version = document_version.strip().lower()
            if len(version) != 64:
                raise FingerprintError("document_version must be a SHA-256 hex digest.")
            try:
                int(version, 16)
            except ValueError as exc:
                raise FingerprintError("document_version must contain hexadecimal characters only.") from exc
            canonical.add((document_id.strip(), version, chunk_id.strip()))
    except TypeError as exc:
        raise FingerprintError("Evidence records must be iterable.") from exc
    if not canonical:
        raise FingerprintError("At least one evidence record is required.")
    records = [
        {"document_id": doc_id, "document_version": version, "chunk_id": chunk_id}
        for doc_id, version, chunk_id in sorted(canonical)
    ]
    return _sha256_json(records)
