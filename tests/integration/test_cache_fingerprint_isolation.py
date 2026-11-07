
import hashlib
from uuid import uuid4

import numpy as np

from backend.cache.fingerprints import (build_permission_fingerprint,build_evidence_fingerprint)
from backend.cache.redis_cache import RedisCache


def make_version(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def make_evidence(version: str, chunk_id: str) -> list[dict]:
    return [
        {
            "document_id": "task5-isolation-test-document",
            "document_version": version,
            "chunk_id": chunk_id,
        }
    ]


def cleanup_namespace(cache: RedisCache, namespace: str) -> None:
    """Delete only semantic keys belonging to this test namespace."""
    entry_pattern = (
        "semantic-entry:"
        + namespace.removeprefix("semantic-index:")
        + ":*"
    )

    entry_keys = list(cache.client.scan_iter(match=entry_pattern))

    if entry_keys:
        cache.client.delete(*entry_keys)

    cache.client.delete(namespace)


def main() -> None:
    cache = RedisCache(
        semantic_threshold=0.92,
        semantic_max_entries=100,
    )

    permission_tags = ["hr", "finance"]
    reordered_permission_tags = ["finance", "hr"]
    different_permission_tags = ["hr"]

    permission_fp = build_permission_fingerprint(permission_tags)
    reordered_permission_fp = build_permission_fingerprint(
        reordered_permission_tags
    )
    different_permission_fp = build_permission_fingerprint(
        different_permission_tags
    )

    version_one = make_version("document version one")
    version_two = make_version("document version two")

    evidence_one = build_evidence_fingerprint(
        make_evidence(version_one, "chunk-001")
    )
    evidence_reordered = build_evidence_fingerprint(
        make_evidence(version_one, "chunk-001")
    )
    evidence_new_version = build_evidence_fingerprint(
        make_evidence(version_two, "chunk-001")
    )
    evidence_new_chunk = build_evidence_fingerprint(
        make_evidence(version_one, "chunk-002")
    )

    # A unique question prevents collisions with unrelated cache entries.
    question = f"Task 5 Redis isolation test {uuid4().hex}"

    answer = {
        "answer": "Task 5 isolation test answer",
        "citations": ["C1"],
        "not_found": False,
    }

    # A deterministic, normalized 1024-dimensional test vector.
    embedding = np.zeros(1024, dtype=np.float32)
    embedding[0] = 1.0

    exact_keys = []
    semantic_namespaces = []

    try:
        assert cache.connect(), "Redis connection failed."
        print("[PASS] Redis connection")

       
        # Canonical fingerprint checks
        

        assert permission_fp == reordered_permission_fp
        print("[PASS] Permission order produces the same fingerprint")

        assert evidence_one == evidence_reordered
        print("[PASS] Identical evidence produces the same fingerprint")

        assert permission_fp != different_permission_fp
        print("[PASS] Different permissions produce different fingerprints")

        assert evidence_one != evidence_new_version
        print("[PASS] Document version change produces a different fingerprint")

        assert evidence_one != evidence_new_chunk
        print("[PASS] Evidence chunk change produces a different fingerprint")

       
        # Exact cache isolation
        

        exact_keys.extend(
            [
                cache.build_exact_cache_key(
                    question, permission_fp, evidence_one
                ),
                cache.build_exact_cache_key(
                    question, different_permission_fp, evidence_one
                ),
                cache.build_exact_cache_key(
                    question, permission_fp, evidence_new_version
                ),
                cache.build_exact_cache_key(
                    question, permission_fp, evidence_new_chunk
                ),
            ]
        )

        assert cache.set_exact(
            question,
            permission_fp,
            evidence_one,
            answer,
        )
        print("[PASS] Exact cache storage")

        # Same canonical permissions and evidence must retrieve the answer.
        assert cache.get_exact(
            question,
            reordered_permission_fp,
            evidence_reordered,
        ) == answer
        print("[PASS] Exact cache hit with equivalent fingerprints")

        # A different permission set must not retrieve it.
        assert cache.get_exact(
            question,
            different_permission_fp,
            evidence_one,
        ) is None
        print("[PASS] Exact cache permission isolation")

        # A changed document version must not retrieve it.
        assert cache.get_exact(
            question,
            permission_fp,
            evidence_new_version,
        ) is None
        print("[PASS] Exact cache document-version isolation")

        # Changed evidence chunk must not retrieve it.
        assert cache.get_exact(
            question,
            permission_fp,
            evidence_new_chunk,
        ) is None
        print("[PASS] Exact cache evidence isolation")

        
        # Semantic cache isolation
        

        semantic_namespaces.extend(
            [
                cache._semantic_namespace(permission_fp, evidence_one),
                cache._semantic_namespace(
                    different_permission_fp, evidence_one
                ),
                cache._semantic_namespace(
                    permission_fp, evidence_new_version
                ),
                cache._semantic_namespace(
                    permission_fp, evidence_new_chunk
                ),
            ]
        )

        assert cache.set_semantic(
            question,
            embedding,
            permission_fp,
            evidence_one,
            answer,
        )
        print("[PASS] Semantic cache storage")

        # Same permission/evidence identity should produce a semantic hit.
        assert cache.get_semantic(
            embedding,
            reordered_permission_fp,
            evidence_reordered,
        ) == answer
        print("[PASS] Semantic cache hit with equivalent fingerprints")

        # Different permissions must be isolated.
        assert cache.get_semantic(
            embedding,
            different_permission_fp,
            evidence_one,
        ) is None
        print("[PASS] Semantic cache permission isolation")

        # Changed document version must be isolated.
        assert cache.get_semantic(
            embedding,
            permission_fp,
            evidence_new_version,
        ) is None
        print("[PASS] Semantic cache document-version isolation")

        # Changed evidence chunk must be isolated.
        assert cache.get_semantic(
            embedding,
            permission_fp,
            evidence_new_chunk,
        ) is None
        print("[PASS] Semantic cache evidence isolation")

        print("\nAll Task 5 real Redis isolation checks passed.")

    finally:
        # Remove only keys generated by this test.
        for key in set(exact_keys):
            cache.client.delete(key)

        for namespace in set(semantic_namespaces):
            cleanup_namespace(cache, namespace)

        cache.close()


if __name__ == "__main__":
    main()
