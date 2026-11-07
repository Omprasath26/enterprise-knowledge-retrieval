


import hashlib
import json
import os
import re
import time
from typing import Any
from uuid import uuid4

import numpy as np
import redis
from dotenv import load_dotenv


DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_DB = 0
DEFAULT_CONNECT_TIMEOUT = 5
DEFAULT_SOCKET_TIMEOUT = 5

DEFAULT_CACHE_TTL = 3600
DEFAULT_SEMANTIC_THRESHOLD = 0.92
DEFAULT_SEMANTIC_MAX_ENTRIES = 100
DEFAULT_EMBEDDING_DIMENSION = 1024


class RedisCacheConfigurationError(Exception):
    """Raised when Redis configuration is invalid."""


class RedisCacheConnectionError(Exception):
    """Raised when a Redis connection operation fails."""


class RedisCacheOperationError(Exception):
    """Raised when a Redis cache operation fails."""


class RedisCache:
    """Manage Redis connections, exact caching, and semantic caching."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        db: int | None = None,
        password: str | None = None,
        client: Any | None = None,
        default_ttl: int | None = None,
        semantic_threshold: float | None = None,
        semantic_max_entries: int | None = None) -> None:
        load_dotenv()

        try:
            self.host = host or os.getenv(
                "REDIS_HOST", DEFAULT_REDIS_HOST
            )

            self.port = (
                port
                if port is not None
                else int(os.getenv("REDIS_PORT", DEFAULT_REDIS_PORT))
            )

            self.db = (
                db
                if db is not None
                else int(os.getenv("REDIS_DB", DEFAULT_REDIS_DB))
            )

            self.password = (
                password
                if password is not None
                else os.getenv("REDIS_PASSWORD") or None
            )

            self.default_ttl = (
                default_ttl
                if default_ttl is not None
                else int(
                    os.getenv(
                        "REDIS_CACHE_TTL_SECONDS",
                        DEFAULT_CACHE_TTL,
                    )
                )
            )

            self.semantic_threshold = (
                semantic_threshold
                if semantic_threshold is not None
                else float(
                    os.getenv(
                        "REDIS_SEMANTIC_THRESHOLD",
                        DEFAULT_SEMANTIC_THRESHOLD,
                    )
                )
            )

            self.semantic_max_entries = (
                semantic_max_entries
                if semantic_max_entries is not None
                else int(
                    os.getenv(
                        "REDIS_SEMANTIC_MAX_ENTRIES",
                        DEFAULT_SEMANTIC_MAX_ENTRIES,
                    )
                )
            )

        except (TypeError, ValueError) as exc:
            raise RedisCacheConfigurationError(
                "Redis configuration contains invalid numeric values."
            ) from exc

        if not isinstance(self.host, str) or not self.host.strip():
            raise RedisCacheConfigurationError(
                "Redis host must be a non-empty string."
            )

        if (
            not isinstance(self.port, int)
            or isinstance(self.port, bool)
            or not 1 <= self.port <= 65535
        ):
            raise RedisCacheConfigurationError(
                "Redis port must be between 1 and 65535."
            )

        if (
            not isinstance(self.db, int)
            or isinstance(self.db, bool)
            or self.db < 0
        ):
            raise RedisCacheConfigurationError(
                "Redis database number cannot be negative."
            )

        if (
            not isinstance(self.default_ttl, int)
            or isinstance(self.default_ttl, bool)
            or self.default_ttl <= 0
        ):
            raise RedisCacheConfigurationError(
                "Cache TTL must be a positive integer."
            )

        if (
            not isinstance(self.semantic_threshold, (int, float))
            or isinstance(self.semantic_threshold, bool)
            or not 0.0 <= self.semantic_threshold <= 1.0
        ):
            raise RedisCacheConfigurationError(
                "Semantic threshold must be between 0 and 1."
            )

        if (
            not isinstance(self.semantic_max_entries, int)
            or isinstance(self.semantic_max_entries, bool)
            or self.semantic_max_entries <= 0):
            raise RedisCacheConfigurationError(
                "Semantic maximum entries must be a positive integer."
            )

        self.client = client if client is not None else redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            socket_connect_timeout=DEFAULT_CONNECT_TIMEOUT,
            socket_timeout=DEFAULT_SOCKET_TIMEOUT,
            decode_responses=True,
        )

    
    # CONNECTION MANAGEMENT
    

    def connect(self) -> bool:
        """Check whether Redis is reachable."""

        try:
            response = self.client.ping()

            if response is not True:
                raise RedisCacheConnectionError(
                    "Redis did not return a successful PING response."
                )

            return True

        except RedisCacheConnectionError:
            raise

        except (redis.exceptions.RedisError, OSError, TimeoutError) as exc:
            raise RedisCacheConnectionError(
                "Unable to connect to Redis. Check the Redis server "
                "and connection settings."
            ) from exc

    def close(self) -> None:
        """Close the Redis client."""

        try:
            self.client.close()

        except (redis.exceptions.RedisError, OSError) as exc:
            raise RedisCacheConnectionError(
                "Unable to close the Redis connection."
            ) from exc

    
    # COMMON HELPERS
    

    @staticmethod
    def _normalize_question(question: str) -> str:
        """Normalize question whitespace and capitalization."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        return re.sub(r"\s+", " ", question.strip()).casefold()

    @staticmethod
    def _sha256(value: str) -> str:
        """Return a SHA-256 hexadecimal digest."""

        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def _fingerprint_hashes(
        cls,
        permission_fingerprint: str,
        evidence_fingerprint: str) -> tuple[str, str]:
        """Validate and hash permission and evidence fingerprints."""

        if (
            not isinstance(permission_fingerprint, str)
            or not permission_fingerprint.strip()
        ):
            raise ValueError(
                "Permission fingerprint must be a non-empty string."
            )

        if (
            not isinstance(evidence_fingerprint, str)
            or not evidence_fingerprint.strip()
        ):
            raise ValueError(
                "Evidence fingerprint must be a non-empty string."
            )

        return (
            cls._sha256(permission_fingerprint),
            cls._sha256(evidence_fingerprint),
        )

    @staticmethod
    def _validate_ttl(ttl: int) -> None:
        if (
            not isinstance(ttl, int)
            or isinstance(ttl, bool)
            or ttl <= 0
        ):
            raise ValueError("Cache TTL must be a positive integer.")

    @staticmethod
    def _validate_embedding(embedding: Any) -> np.ndarray:
        """Validate and normalize a BGE-M3 embedding vector."""

        try:
            vector = np.asarray(embedding, dtype=np.float32)

        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Embedding must contain numeric values."
            ) from exc

        if vector.ndim != 1:
            raise ValueError(
                "Embedding must be a one-dimensional vector."
            )

        if vector.size != DEFAULT_EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding must contain exactly "
                f"{DEFAULT_EMBEDDING_DIMENSION} dimensions."
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                "Embedding contains non-finite values."
            )

        norm = float(np.linalg.norm(vector))

        if norm == 0.0:
            raise ValueError("Embedding cannot be a zero vector.")

        return vector / norm

    @staticmethod
    def _cosine_similarity(
        first: np.ndarray,
        second: np.ndarray) -> float:
        """Calculate cosine similarity between normalized vectors."""

        return float(np.dot(first, second))

    
    # EXACT CACHE
   
    @classmethod
    def build_exact_cache_key(
        cls,
        question: str,
        permission_fingerprint: str,
        evidence_fingerprint: str) -> str:
        """Build a deterministic exact-cache key."""

        permission_hash, evidence_hash = cls._fingerprint_hashes(
            permission_fingerprint,
            evidence_fingerprint,
        )

        normalized_question = cls._normalize_question(question)
        question_hash = cls._sha256(normalized_question)

        return (
            f"exact:{permission_hash}:"
            f"{evidence_hash}:{question_hash}"
        )

    def get_exact(
        self,
        question: str,
        permission_fingerprint: str,
        evidence_fingerprint: str) -> dict[str, Any] | None:
        """Retrieve an exact cached answer."""

        key = self.build_exact_cache_key(
            question,
            permission_fingerprint,
            evidence_fingerprint,
        )

        try:
            cached_value = self.client.get(key)

            if cached_value is None:
                return None

            payload = json.loads(cached_value)

            if not isinstance(payload, dict):
                raise ValueError("Cached value must be a JSON object.")

            if payload.get("cache_schema_version") != 1:
                raise ValueError("Unsupported cached-value schema.")

            answer_data = payload.get("answer_data")

            if not isinstance(answer_data, dict):
                raise ValueError("Cached answer data is invalid.")

            return answer_data

        except (redis.exceptions.RedisError, OSError) as exc:
            raise RedisCacheOperationError(
                "Unable to retrieve the exact cached answer."
            ) from exc

        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RedisCacheOperationError(
                "The exact cached entry is invalid or corrupted."
            ) from exc

    def set_exact(
        self,
        question: str,
        permission_fingerprint: str,
        evidence_fingerprint: str,
        answer_data: dict[str, Any],
        ttl: int | None = None) -> bool:
        """Store an exact cached answer."""

        if not isinstance(answer_data, dict):
            raise ValueError("Answer data must be a dictionary.")

        effective_ttl = (
            self.default_ttl if ttl is None else ttl
        )
        self._validate_ttl(effective_ttl)

        key = self.build_exact_cache_key(
            question,
            permission_fingerprint,
            evidence_fingerprint,
        )

        payload = {
            "cache_schema_version": 1,
            "answer_data": answer_data,
        }

        try:
            serialized = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
            )

            result = self.client.set(
                key,
                serialized,
                ex=effective_ttl,
            )

            return bool(result)

        except (redis.exceptions.RedisError, OSError) as exc:
            raise RedisCacheOperationError(
                "Unable to store the exact cached answer."
            ) from exc

        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Answer data must contain JSON-serializable values."
            ) from exc

    
    # SEMANTIC CACHE KEY HELPERS
    

    @classmethod
    def _semantic_namespace(
        cls,
        permission_fingerprint: str,
        evidence_fingerprint: str) -> str:
        """Build a permission- and evidence-isolated namespace."""

        permission_hash, evidence_hash = cls._fingerprint_hashes(
            permission_fingerprint,
            evidence_fingerprint,
        )

        return (
            f"semantic-index:{permission_hash}:"
            f"{evidence_hash}"
        )

    @staticmethod
    def _semantic_entry_key(
        namespace: str,
        entry_id: str) -> str:
        """Build the Redis key for one semantic entry."""

        namespace_hashes = namespace.removeprefix(
            "semantic-index:"
        )

        return f"semantic-entry:{namespace_hashes}:{entry_id}"

    def _remove_stale_semantic_entries(
        self,
        namespace: str) -> None:
        """Remove index members whose entry data has expired."""

        entry_ids = self.client.zrange(namespace, 0, -1)
        stale_ids = []

        for entry_id in entry_ids:
            entry_key = self._semantic_entry_key(
                namespace,
                entry_id,
            )

            if self.client.get(entry_key) is None:
                stale_ids.append(entry_id)

        if stale_ids:
            self.client.zrem(namespace, *stale_ids)

    
    # SEMANTIC CACHE WRITE
    

    def set_semantic(
        self,
        question: str,
        embedding: Any,
        permission_fingerprint: str,
        evidence_fingerprint: str,
        answer_data: dict[str, Any],
        ttl: int | None = None) -> bool:
        """Store an answer and its embedding in semantic cache."""

        normalized_question = self._normalize_question(question)
        vector = self._validate_embedding(embedding)

        if not isinstance(answer_data, dict):
            raise ValueError("Answer data must be a dictionary.")

        if answer_data.get("not_found") is True:
            raise ValueError(
                "Not-found answers cannot be stored in semantic cache."
            )

        effective_ttl = (
            self.default_ttl if ttl is None else ttl
        )
        self._validate_ttl(effective_ttl)

        namespace = self._semantic_namespace(
            permission_fingerprint,
            evidence_fingerprint,
        )

        entry_id = uuid4().hex
        entry_key = self._semantic_entry_key(
            namespace,
            entry_id,
        )

        payload = {
            "cache_schema_version": 1,
            "question": normalized_question,
            "embedding": vector.tolist(),
            "answer_data": answer_data,
            "created_at": time.time(),
        }

        try:
            serialized = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
            )

            # Remove expired entry references before enforcing the limit.
            self._remove_stale_semantic_entries(namespace)

            # Store the entry data with its own TTL.
            self.client.set(
                entry_key,
                serialized,
                ex=effective_ttl,
            )

            # Add the entry ID to the namespace index.
            self.client.zadd(
                namespace,
                {entry_id: time.time_ns()},
            )

            # Refresh the namespace index TTL.
            self.client.expire(namespace, effective_ttl)

            # Enforce the configured namespace entry limit.
            current_count = self.client.zcard(namespace)

            if current_count > self.semantic_max_entries:
                excess = current_count - self.semantic_max_entries

                oldest_ids = self.client.zrange(
                    namespace,
                    0,
                    excess - 1,
                )

                if oldest_ids:
                    self.client.zrem(namespace, *oldest_ids)

                    old_keys = [
                        self._semantic_entry_key(namespace, old_id)
                        for old_id in oldest_ids
                    ]

                    self.client.delete(*old_keys)

            return True

        except (redis.exceptions.RedisError, OSError) as exc:
            raise RedisCacheOperationError(
                "Unable to store the semantic cached answer."
            ) from exc

        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Semantic cache data must be JSON-serializable."
            ) from exc

    
    # SEMANTIC CACHE READ
    
    def get_semantic(
        self,
        embedding: Any,
        permission_fingerprint: str,
        evidence_fingerprint: str,
        threshold: float | None = None) -> dict[str, Any] | None:
        """Retrieve the most similar answer from semantic cache.

        Returns:
            The cached answer dictionary if similarity meets the threshold.
            None if there is no suitable semantic cache entry.
        """

        query_vector = self._validate_embedding(embedding)

        effective_threshold = (
            self.semantic_threshold
            if threshold is None
            else threshold
        )

        if (
            not isinstance(effective_threshold, (int, float))
            or isinstance(effective_threshold, bool)
            or not 0.0 <= effective_threshold <= 1.0
        ):
            raise ValueError(
                "Similarity threshold must be between 0 and 1."
            )

        namespace = self._semantic_namespace(
            permission_fingerprint,
            evidence_fingerprint,
        )

        try:
            # Remove stale references before searching.
            self._remove_stale_semantic_entries(namespace)

            # Search newest entries first.
            entry_ids = self.client.zrevrange(
                namespace,
                0,
                self.semantic_max_entries - 1,
            )

            best_score = -1.0
            best_answer = None

            for entry_id in entry_ids:
                entry_key = self._semantic_entry_key(
                    namespace,
                    entry_id,
                )

                cached_value = self.client.get(entry_key)

                if cached_value is None:
                    continue

                try:
                    payload = json.loads(cached_value)

                    if (
                        not isinstance(payload, dict)
                        or payload.get("cache_schema_version") != 1
                    ):
                        continue

                    stored_vector = self._validate_embedding(
                        payload.get("embedding")
                    )

                    answer_data = payload.get("answer_data")

                    if not isinstance(answer_data, dict):
                        continue

                    # Never reuse not-found answers semantically.
                    if answer_data.get("not_found") is True:
                        continue

                    score = self._cosine_similarity(
                        query_vector,
                        stored_vector,
                    )

                    if (
                        score >= effective_threshold
                        and score > best_score
                    ):
                        best_score = score
                        best_answer = answer_data

                except (
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ):
                    # Ignore malformed individual entries.
                    continue

            return best_answer

        except (redis.exceptions.RedisError, OSError) as exc:
            raise RedisCacheOperationError(
                "Unable to search the semantic cache."
            ) from exc


if __name__ == "__main__":
    print("Redis cache module loaded successfully.")
