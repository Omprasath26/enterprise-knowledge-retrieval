


from uuid import uuid4

import numpy as np

from backend.cache.redis_cache import RedisCache


def run_validation():
    cache = RedisCache()
    cache.connect()

    token = uuid4().hex

    permission = f"task4-test-permission-{token}"
    evidence = f"task4-test-evidence-{token}"

    limit_permission = f"task4-limit-permission-{token}"
    limit_evidence = f"task4-limit-evidence-{token}"

    namespace = cache._semantic_namespace(permission, evidence)
    limit_namespace = cache._semantic_namespace(
        limit_permission,
        limit_evidence,
    )

    try:
        # 1. Create synthetic 1024-dimensional embeddings.
        original_embedding = np.zeros(1024, dtype=np.float32)
        original_embedding[0] = 1.0

        similar_embedding = np.zeros(1024, dtype=np.float32)
        similar_embedding[0] = 0.95
        similar_embedding[1] = np.sqrt(1 - 0.95**2)

        unrelated_embedding = np.zeros(1024, dtype=np.float32)
        unrelated_embedding[1] = 1.0

        answer = {
            "answer": "Redis semantic integration test passed.",
            "citations": [],
            "not_found": False,
            "model": "integration-test",
        }

        # 2. Store an answer.
        assert cache.set_semantic(
            "What is the integration test?",
            original_embedding,
            permission,
            evidence,
            answer,
        )

        print("[PASS] Semantic cache storage")

        # 3. Similar vector should retrieve the cached answer.
        result = cache.get_semantic(
            similar_embedding,
            permission,
            evidence,
        )

        assert result == answer

        print("[PASS] Similarity hit at threshold 0.92")

        # 4. Unrelated vector should not retrieve the answer.
        result = cache.get_semantic(
            unrelated_embedding,
            permission,
            evidence,
        )

        assert result is None

        print("[PASS] Similarity miss")

        # 5. Different permission must not retrieve the answer.
        result = cache.get_semantic(
            original_embedding,
            f"different-permission-{token}",
            evidence,
        )

        assert result is None

        print("[PASS] Permission isolation")

        # 6. Different evidence fingerprint must not retrieve the answer.
        result = cache.get_semantic(
            original_embedding,
            permission,
            f"different-evidence-{token}",
        )

        assert result is None

        print("[PASS] Evidence isolation")

        # 7. Verify maximum-entry enforcement.
        limited_cache = RedisCache(
            client=cache.client,
            semantic_max_entries=2,
        )

        for index in range(3):
            limited_cache.set_semantic(
                f"Maximum entry test question {index}",
                original_embedding,
                limit_permission,
                limit_evidence,
                {
                    "answer": f"Test answer {index}",
                    "not_found": False,
                    "citations": [],
                },
            )

        entry_count = cache.client.zcard(limit_namespace)

        assert entry_count == 2, (
            f"Expected 2 entries, found {entry_count}"
        )

        print("[PASS] Maximum 2 entries enforced")

        print("\nAll real Redis semantic-cache checks passed.")

    finally:
        # Delete only the temporary entries created by this test.
        for current_namespace in (namespace, limit_namespace):
            entry_ids = cache.client.zrange(
                current_namespace,
                0,
                -1,
            )

            entry_keys = [
                cache._semantic_entry_key(current_namespace, entry_id)
                for entry_id in entry_ids
            ]

            if entry_keys:
                cache.client.delete(*entry_keys)

            cache.client.delete(current_namespace)

        cache.close()


if __name__ == "__main__":
    run_validation()
