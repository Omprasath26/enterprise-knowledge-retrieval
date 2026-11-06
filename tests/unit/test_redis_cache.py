


import json

import pytest
import redis

from backend.cache.redis_cache import (RedisCache,RedisCacheConfigurationError,RedisCacheConnectionError,RedisCacheOperationError)


# FAKE REDIS CLIENT


class FakeRedisClient:
    """In-memory Redis substitute for unit testing."""

    def __init__(
        self,
        ping_result=True,
        ping_error=None):
        self.ping_result = ping_result
        self.ping_error = ping_error

        self.ping_count = 0
        self.closed = False

        self.store = {}
        self.expirations = {}
        self.sorted_sets = {}

    def ping(self):
        self.ping_count += 1

        if self.ping_error:
            raise self.ping_error

        return self.ping_result

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value
        self.expirations[key] = ex
        return True

    def delete(self, *keys):
        deleted = 0

        for key in keys:
            if key in self.store:
                del self.store[key]
                deleted += 1

            self.expirations.pop(key, None)

        return deleted

    def expire(self, key, seconds):
        self.expirations[key] = seconds
        return True

    def zadd(self, key, mapping):
        entries = self.sorted_sets.setdefault(key, {})
        added = 0

        for member, score in mapping.items():
            if member not in entries:
                added += 1

            entries[member] = score

        return added

    def zcard(self, key):
        return len(self.sorted_sets.get(key, {}))

    def zrange(self, key, start, end):
        entries = self.sorted_sets.get(key, {})

        ordered = sorted(
            entries,
            key=lambda member: entries[member],
        )

        if end == -1:
            return ordered[start:]

        return ordered[start:end + 1]

    def zrevrange(self, key, start, end):
        entries = self.sorted_sets.get(key, {})

        ordered = sorted(
            entries,
            key=lambda member: entries[member],
            reverse=True,
        )

        if end == -1:
            return ordered[start:]

        return ordered[start:end + 1]

    def zrem(self, key, *members):
        entries = self.sorted_sets.get(key, {})
        removed = 0

        for member in members:
            if member in entries:
                del entries[member]
                removed += 1

        return removed

    def close(self):
        self.closed = True



# TEST HELPERS



def make_embedding(value=1.0):
    """Return a valid 1024-dimensional test embedding."""

    return [value] + [0.0] * 1023


def make_answer(answer="Test answer",not_found=False):
    """Return a sample answer payload."""

    return {
        "answer": answer,
        "citations": [
            {
                "reference": "C1",
                "source_file": "HR_Policy.pdf",
                "headings": ["Leave Policy"],
                "page_numbers": [2],
                "relevance_score": 0.70,
            }
        ],
        "not_found": not_found,
        "model": "gpt-4o-mini",
    }



# REDIS CONFIGURATION TESTS



def test_default_configuration():
    cache = RedisCache(client=FakeRedisClient())

    assert cache.host == "localhost"
    assert cache.port == 6379
    assert cache.db == 0
    assert cache.default_ttl == 3600
    assert cache.semantic_threshold == 0.92
    assert cache.semantic_max_entries == 100


def test_custom_configuration():
    cache = RedisCache(
        host="127.0.0.1",
        port=6380,
        db=1,
        password="test-password",
        default_ttl=1800,
        semantic_threshold=0.95,
        semantic_max_entries=50,
        client=FakeRedisClient(),
    )

    assert cache.host == "127.0.0.1"
    assert cache.port == 6380
    assert cache.db == 1
    assert cache.password == "test-password"
    assert cache.default_ttl == 1800
    assert cache.semantic_threshold == 0.95
    assert cache.semantic_max_entries == 50


def test_invalid_port_raises_configuration_error():
    with pytest.raises(RedisCacheConfigurationError):
        RedisCache(port=70000, client=FakeRedisClient())


def test_zero_port_raises_configuration_error():
    with pytest.raises(RedisCacheConfigurationError):
        RedisCache(port=0, client=FakeRedisClient())


def test_invalid_database_raises_configuration_error():
    with pytest.raises(RedisCacheConfigurationError):
        RedisCache(db=-1, client=FakeRedisClient())


def test_invalid_ttl_raises_configuration_error():
    with pytest.raises(RedisCacheConfigurationError):
        RedisCache(default_ttl=0, client=FakeRedisClient())


def test_invalid_semantic_threshold_raises_configuration_error():
    with pytest.raises(RedisCacheConfigurationError):
        RedisCache(
            semantic_threshold=1.5,
            client=FakeRedisClient(),
        )


def test_invalid_semantic_max_entries_raises_configuration_error():
    with pytest.raises(RedisCacheConfigurationError):
        RedisCache(
            semantic_max_entries=0,
            client=FakeRedisClient(),
        )


def test_environment_configuration(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "redis-test")
    monkeypatch.setenv("REDIS_PORT", "6381")
    monkeypatch.setenv("REDIS_DB", "2")
    monkeypatch.setenv("REDIS_CACHE_TTL_SECONDS", "2400")
    monkeypatch.setenv("REDIS_SEMANTIC_THRESHOLD", "0.94")
    monkeypatch.setenv("REDIS_SEMANTIC_MAX_ENTRIES", "75")

    cache = RedisCache(client=FakeRedisClient())

    assert cache.host == "redis-test"
    assert cache.port == 6381
    assert cache.db == 2
    assert cache.default_ttl == 2400
    assert cache.semantic_threshold == 0.94
    assert cache.semantic_max_entries == 75


def test_invalid_numeric_environment_configuration(monkeypatch):
    monkeypatch.setenv("REDIS_PORT", "invalid")

    with pytest.raises(RedisCacheConfigurationError):
        RedisCache(client=FakeRedisClient())



# REDIS CONNECTION TESTS



def test_redis_connection_success():
    client = FakeRedisClient()
    cache = RedisCache(client=client)

    assert cache.connect() is True
    assert client.ping_count == 1


def test_redis_connection_failure():
    client = FakeRedisClient(
        ping_error=redis.exceptions.ConnectionError(
            "Connection failed"
        )
    )

    cache = RedisCache(client=client)

    with pytest.raises(RedisCacheConnectionError):
        cache.connect()


def test_redis_ping_false_raises_connection_error():
    client = FakeRedisClient(ping_result=False)
    cache = RedisCache(client=client)

    with pytest.raises(RedisCacheConnectionError):
        cache.connect()


def test_redis_connection_close():
    client = FakeRedisClient()
    cache = RedisCache(client=client)

    cache.close()

    assert client.closed is True



# EXACT CACHE KEY TESTS



def test_exact_cache_key_is_deterministic():
    key1 = RedisCache.build_exact_cache_key(
        "What is the leave policy?",
        "hr",
        "evidence-001",
    )

    key2 = RedisCache.build_exact_cache_key(
        "What is the leave policy?",
        "hr",
        "evidence-001",
    )

    assert key1 == key2


def test_exact_cache_key_normalizes_case():
    key1 = RedisCache.build_exact_cache_key(
        "What is the leave policy?",
        "hr",
        "evidence-001",
    )

    key2 = RedisCache.build_exact_cache_key(
        "WHAT IS THE LEAVE POLICY?",
        "hr",
        "evidence-001",
    )

    assert key1 == key2


def test_exact_cache_key_normalizes_whitespace():
    key1 = RedisCache.build_exact_cache_key(
        "What is the leave policy?",
        "hr",
        "evidence-001",
    )

    key2 = RedisCache.build_exact_cache_key(
        "  What   is the leave policy?  ",
        "hr",
        "evidence-001",
    )

    assert key1 == key2


def test_exact_cache_key_changes_with_permission():
    key1 = RedisCache.build_exact_cache_key(
        "Question?",
        "hr",
        "evidence-001",
    )

    key2 = RedisCache.build_exact_cache_key(
        "Question?",
        "finance",
        "evidence-001",
    )

    assert key1 != key2


def test_exact_cache_key_changes_with_evidence():
    key1 = RedisCache.build_exact_cache_key(
        "Question?",
        "hr",
        "evidence-001",
    )

    key2 = RedisCache.build_exact_cache_key(
        "Question?",
        "hr",
        "evidence-002",
    )

    assert key1 != key2


def test_exact_cache_rejects_empty_question():
    with pytest.raises(ValueError):
        RedisCache.build_exact_cache_key(
            "",
            "hr",
            "evidence-001",
        )


def test_exact_cache_rejects_missing_permission_fingerprint():
    with pytest.raises(ValueError):
        RedisCache.build_exact_cache_key(
            "Question?",
            "",
            "evidence-001",
        )


def test_exact_cache_rejects_missing_evidence_fingerprint():
    with pytest.raises(ValueError):
        RedisCache.build_exact_cache_key(
            "Question?",
            "hr",
            "",
        )



# EXACT CACHE STORAGE AND RETRIEVAL TESTS



def test_exact_cache_store_and_retrieve():
    client = FakeRedisClient()
    cache = RedisCache(client=client)

    answer_data = make_answer()

    stored = cache.set_exact(
        "What is the leave policy?",
        "hr",
        "evidence-001",
        answer_data,
    )

    retrieved = cache.get_exact(
        "What is the leave policy?",
        "hr",
        "evidence-001",
    )

    assert stored is True
    assert retrieved == answer_data


def test_exact_cache_miss_returns_none():
    cache = RedisCache(client=FakeRedisClient())

    result = cache.get_exact(
        "Unknown question?",
        "hr",
        "evidence-001",
    )

    assert result is None


def test_exact_cache_isolated_by_permission():
    client = FakeRedisClient()
    cache = RedisCache(client=client)

    cache.set_exact(
        "Question?",
        "hr",
        "evidence-001",
        make_answer(),
    )

    result = cache.get_exact(
        "Question?",
        "finance",
        "evidence-001",
    )

    assert result is None


def test_exact_cache_isolated_by_evidence():
    client = FakeRedisClient()
    cache = RedisCache(client=client)

    cache.set_exact(
        "Question?",
        "hr",
        "evidence-001",
        make_answer(),
    )

    result = cache.get_exact(
        "Question?",
        "hr",
        "evidence-002",
    )

    assert result is None


def test_exact_cache_uses_default_ttl():
    client = FakeRedisClient()
    cache = RedisCache(
        client=client,
        default_ttl=1800,
    )

    cache.set_exact(
        "Question?",
        "hr",
        "evidence-001",
        make_answer(),
    )

    key = RedisCache.build_exact_cache_key(
        "Question?",
        "hr",
        "evidence-001",
    )

    assert client.expirations[key] == 1800


def test_exact_cache_uses_custom_ttl():
    client = FakeRedisClient()
    cache = RedisCache(client=client)

    cache.set_exact(
        "Question?",
        "hr",
        "evidence-001",
        make_answer(),
        ttl=120,
    )

    key = RedisCache.build_exact_cache_key(
        "Question?",
        "hr",
        "evidence-001",
    )

    assert client.expirations[key] == 120


def test_exact_cache_rejects_invalid_ttl():
    cache = RedisCache(client=FakeRedisClient())

    with pytest.raises(ValueError):
        cache.set_exact(
            "Question?",
            "hr",
            "evidence-001",
            make_answer(),
            ttl=0,
        )


def test_exact_cache_rejects_non_dictionary_answer():
    cache = RedisCache(client=FakeRedisClient())

    with pytest.raises(ValueError):
        cache.set_exact(
            "Question?",
            "hr",
            "evidence-001",
            "Not a dictionary",
        )


def test_exact_cache_rejects_non_serializable_answer():
    cache = RedisCache(client=FakeRedisClient())

    with pytest.raises(ValueError):
        cache.set_exact(
            "Question?",
            "hr",
            "evidence-001",
            {"invalid": object()},
        )


def test_exact_cache_rejects_corrupted_json():
    client = FakeRedisClient()
    cache = RedisCache(client=client)

    key = RedisCache.build_exact_cache_key(
        "Question?",
        "hr",
        "evidence-001",
    )

    client.set(key, "not-valid-json", ex=100)

    with pytest.raises(RedisCacheOperationError):
        cache.get_exact(
            "Question?",
            "hr",
            "evidence-001",
        )


def test_exact_cache_rejects_unsupported_schema():
    client = FakeRedisClient()
    cache = RedisCache(client=client)

    key = RedisCache.build_exact_cache_key(
        "Question?",
        "hr",
        "evidence-001",
    )

    client.set(
        key,
        json.dumps({
            "cache_schema_version": 999,
            "answer_data": make_answer(),
        }),
        ex=100,
    )

    with pytest.raises(RedisCacheOperationError):
        cache.get_exact(
            "Question?",
            "hr",
            "evidence-001",
        )



# SEMANTIC CACHE CONFIGURATION TESTS



def test_semantic_cache_environment_settings(monkeypatch):
    monkeypatch.setenv("REDIS_SEMANTIC_THRESHOLD", "0.95")
    monkeypatch.setenv("REDIS_SEMANTIC_MAX_ENTRIES", "50")

    cache = RedisCache(client=FakeRedisClient())

    assert cache.semantic_threshold == 0.95
    assert cache.semantic_max_entries == 50


def test_semantic_cache_namespace_is_deterministic():
    namespace1 = RedisCache._semantic_namespace(
        "hr",
        "evidence-001",
    )

    namespace2 = RedisCache._semantic_namespace(
        "hr",
        "evidence-001",
    )

    assert namespace1 == namespace2


def test_semantic_cache_namespace_isolated_by_permission():
    namespace1 = RedisCache._semantic_namespace(
        "hr",
        "evidence-001",
    )

    namespace2 = RedisCache._semantic_namespace(
        "finance",
        "evidence-001",
    )

    assert namespace1 != namespace2


def test_semantic_cache_namespace_isolated_by_evidence():
    namespace1 = RedisCache._semantic_namespace(
        "hr",
        "evidence-001",
    )

    namespace2 = RedisCache._semantic_namespace(
        "hr",
        "evidence-002",
    )

    assert namespace1 != namespace2



# SEMANTIC CACHE STORAGE AND RETRIEVAL TESTS


def test_semantic_cache_store_and_retrieve():
    client = FakeRedisClient()
    cache = RedisCache(client=client)

    answer_data = make_answer(
        "Employees receive 26 weeks of maternity leave."
    )

    stored = cache.set_semantic(
        "How many weeks of maternity leave?",
        make_embedding(),
        "hr",
        "evidence-001",
        answer_data,
    )

    retrieved = cache.get_semantic(
        make_embedding(),
        "hr",
        "evidence-001",
    )

    assert stored is True
    assert retrieved == answer_data


def test_semantic_cache_miss_returns_none():
    cache = RedisCache(client=FakeRedisClient())

    result = cache.get_semantic(
        make_embedding(),
        "hr",
        "evidence-001",
    )

    assert result is None


def test_semantic_cache_rejects_dissimilar_embedding():
    client = FakeRedisClient()
    cache = RedisCache(
        client=client,
        semantic_threshold=0.92,
    )

    cache.set_semantic(
        "Maternity leave duration?",
        make_embedding(1.0),
        "hr",
        "evidence-001",
        make_answer(),
    )

    opposite_embedding = [-1.0] + [0.0] * 1023

    result = cache.get_semantic(
        opposite_embedding,
        "hr",
        "evidence-001",
    )

    assert result is None


def test_semantic_cache_accepts_identical_embedding():
    client = FakeRedisClient()
    cache = RedisCache(client=client)

    answer_data = make_answer("Test answer")

    cache.set_semantic(
        "Original question?",
        make_embedding(),
        "hr",
        "evidence-001",
        answer_data,
    )

    result = cache.get_semantic(
        make_embedding(),
        "hr",
        "evidence-001",
    )

    assert result == answer_data


def test_semantic_cache_isolated_by_permission():
    client = FakeRedisClient()
    cache = RedisCache(client=client)

    cache.set_semantic(
        "Maternity leave duration?",
        make_embedding(),
        "hr",
        "evidence-001",
        make_answer(),
    )

    result = cache.get_semantic(
        make_embedding(),
        "finance",
        "evidence-001",
    )

    assert result is None


def test_semantic_cache_isolated_by_evidence():
    client = FakeRedisClient()
    cache = RedisCache(client=client)

    cache.set_semantic(
        "Maternity leave duration?",
        make_embedding(),
        "hr",
        "evidence-001",
        make_answer(),
    )

    result = cache.get_semantic(
        make_embedding(),
        "hr",
        "evidence-002",
    )

    assert result is None


def test_semantic_cache_does_not_store_not_found_answers():
    cache = RedisCache(client=FakeRedisClient())

    with pytest.raises(ValueError, match="Not-found answers"):
        cache.set_semantic(
            "Unknown question?",
            make_embedding(),
            "hr",
            "evidence-001",
            make_answer(
                "No supporting information found.",
                not_found=True,
            ),
        )


def test_semantic_cache_rejects_non_dictionary_answer():
    cache = RedisCache(client=FakeRedisClient())

    with pytest.raises(ValueError):
        cache.set_semantic(
            "Question?",
            make_embedding(),
            "hr",
            "evidence-001",
            "Invalid answer",
        )


def test_semantic_cache_rejects_invalid_embedding_dimension():
    cache = RedisCache(client=FakeRedisClient())

    with pytest.raises(ValueError, match="1024 dimensions"):
        cache.set_semantic(
            "Question?",
            [0.1, 0.2],
            "hr",
            "evidence-001",
            make_answer(),
        )


def test_semantic_cache_rejects_zero_embedding():
    cache = RedisCache(client=FakeRedisClient())

    with pytest.raises(ValueError, match="zero vector"):
        cache.set_semantic(
            "Question?",
            [0.0] * 1024,
            "hr",
            "evidence-001",
            make_answer(),
        )


def test_semantic_cache_rejects_non_finite_embedding():
    cache = RedisCache(client=FakeRedisClient())

    embedding = make_embedding()
    embedding[5] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        cache.set_semantic(
            "Question?",
            embedding,
            "hr",
            "evidence-001",
            make_answer(),
        )


def test_semantic_cache_rejects_two_dimensional_embedding():
    cache = RedisCache(client=FakeRedisClient())

    with pytest.raises(ValueError, match="one-dimensional"):
        cache.set_semantic(
            "Question?",
            [make_embedding()],
            "hr",
            "evidence-001",
            make_answer(),
        )


def test_semantic_cache_rejects_invalid_threshold():
    cache = RedisCache(client=FakeRedisClient())

    with pytest.raises(ValueError, match="between 0 and 1"):
        cache.get_semantic(
            make_embedding(),
            "hr",
            "evidence-001",
            threshold=1.5,
        )


def test_semantic_cache_accepts_custom_threshold():
    client = FakeRedisClient()
    cache = RedisCache(client=client)

    answer_data = make_answer()

    cache.set_semantic(
        "Question?",
        make_embedding(),
        "hr",
        "evidence-001",
        answer_data,
    )

    result = cache.get_semantic(
        make_embedding(),
        "hr",
        "evidence-001",
        threshold=0.99,
    )

    assert result == answer_data


def test_semantic_cache_uses_default_ttl():
    client = FakeRedisClient()
    cache = RedisCache(
        client=client,
        default_ttl=1800,
    )

    cache.set_semantic(
        "Question?",
        make_embedding(),
        "hr",
        "evidence-001",
        make_answer(),
    )

    entry_expirations = [
        value
        for key, value in client.expirations.items()
        if key.startswith("semantic-entry:")
    ]

    assert entry_expirations == [1800]


def test_semantic_cache_uses_custom_ttl():
    client = FakeRedisClient()
    cache = RedisCache(client=client)

    cache.set_semantic(
        "Question?",
        make_embedding(),
        "hr",
        "evidence-001",
        make_answer(),
        ttl=120,
    )

    entry_expirations = [
        value
        for key, value in client.expirations.items()
        if key.startswith("semantic-entry:")
    ]

    assert entry_expirations == [120]


def test_semantic_cache_rejects_invalid_ttl():
    cache = RedisCache(client=FakeRedisClient())

    with pytest.raises(ValueError):
        cache.set_semantic(
            "Question?",
            make_embedding(),
            "hr",
            "evidence-001",
            make_answer(),
            ttl=0,
        )



# SEMANTIC CACHE ENTRY LIMIT TESTS


def test_semantic_cache_enforces_maximum_entries():
    client = FakeRedisClient()

    cache = RedisCache(
        client=client,
        semantic_max_entries=2,
    )

    for index in range(3):
        cache.set_semantic(
            f"Question {index}?",
            make_embedding(),
            "hr",
            "evidence-001",
            make_answer(f"Answer {index}"),
        )

    namespace = cache._semantic_namespace(
        "hr",
        "evidence-001",
    )

    assert client.zcard(namespace) == 2


def test_semantic_cache_keeps_newest_entries():
    client = FakeRedisClient()

    cache = RedisCache(
        client=client,
        semantic_max_entries=2,
    )

    cache.set_semantic(
        "Question one?",
        make_embedding(),
        "hr",
        "evidence-001",
        make_answer("Answer one"),
    )

    cache.set_semantic(
        "Question two?",
        make_embedding(),
        "hr",
        "evidence-001",
        make_answer("Answer two"),
    )

    cache.set_semantic(
        "Question three?",
        make_embedding(),
        "hr",
        "evidence-001",
        make_answer("Answer three"),
    )

    namespace = cache._semantic_namespace(
        "hr",
        "evidence-001",
    )

    entry_ids = client.zrange(namespace, 0, -1)

    assert len(entry_ids) == 2


def test_semantic_cache_refreshes_namespace_ttl():
    client = FakeRedisClient()
    cache = RedisCache(client=client, default_ttl=1800)

    cache.set_semantic(
        "Question?",
        make_embedding(),
        "hr",
        "evidence-001",
        make_answer(),
    )

    namespace = cache._semantic_namespace(
        "hr",
        "evidence-001",
    )

    assert client.expirations[namespace] == 1800


def test_semantic_cache_removes_stale_index_members():
    client = FakeRedisClient()
    cache = RedisCache(client=client)

    namespace = cache._semantic_namespace(
        "hr",
        "evidence-001",
    )

    client.zadd(namespace, {"expired-entry": 1})

    result = cache.get_semantic(
        make_embedding(),
        "hr",
        "evidence-001",
    )

    assert result is None
    assert client.zcard(namespace) == 0



# SEMANTIC CACHE CORRUPTION TESTS



def test_semantic_cache_ignores_corrupted_entry():
    client = FakeRedisClient()
    cache = RedisCache(client=client)

    namespace = cache._semantic_namespace(
        "hr",
        "evidence-001",
    )

    entry_id = "corrupted-entry"

    entry_key = cache._semantic_entry_key(
        namespace,
        entry_id,
    )

    client.set(
        entry_key,
        "invalid-json",
        ex=100,
    )

    client.zadd(
        namespace,
        {entry_id: 1},
    )

    result = cache.get_semantic(
        make_embedding(),
        "hr",
        "evidence-001",
    )

    assert result is None


def test_semantic_cache_ignores_unsupported_schema():
    client = FakeRedisClient()
    cache = RedisCache(client=client)

    namespace = cache._semantic_namespace(
        "hr",
        "evidence-001",
    )

    entry_id = "unsupported-schema"
    entry_key = cache._semantic_entry_key(
        namespace,
        entry_id,
    )

    client.set(
        entry_key,
        json.dumps({
            "cache_schema_version": 999,
            "embedding": make_embedding(),
            "answer_data": make_answer(),
        }),
        ex=100,
    )

    client.zadd(
        namespace,
        {entry_id: 1},
    )

    result = cache.get_semantic(
        make_embedding(),
        "hr",
        "evidence-001",
    )

    assert result is None


def test_semantic_cache_ignores_corrupted_embedding():
    client = FakeRedisClient()
    cache = RedisCache(client=client)

    namespace = cache._semantic_namespace(
        "hr",
        "evidence-001",
    )

    entry_id = "bad-embedding"
    entry_key = cache._semantic_entry_key(
        namespace,
        entry_id,
    )

    client.set(
        entry_key,
        json.dumps({
            "cache_schema_version": 1,
            "question": "Question?",
            "embedding": [0.1, 0.2],
            "answer_data": make_answer(),
        }),
        ex=100,
    )

    client.zadd(
        namespace,
        {entry_id: 1},
    )

    result = cache.get_semantic(
        make_embedding(),
        "hr",
        "evidence-001",
    )

    assert result is None
