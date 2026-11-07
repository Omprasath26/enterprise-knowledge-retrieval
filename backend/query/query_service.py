
from dataclasses import asdict
from typing import Any

from backend.cache.fingerprints import (build_evidence_fingerprint,build_permission_fingerprint)
from backend.cache.redis_cache import RedisCache
from backend.query.answer_generator import (AnswerCitation,AnswerGenerator,GeneratedAnswer)
from backend.query.retrieval import DocumentRetriever, RetrievedChunk


class QueryServiceError(Exception):
    """Raised when query orchestration fails."""


class QueryService:
    """Coordinate authorized retrieval, caching, and grounded generation."""

    def __init__(
        self,
        retriever: DocumentRetriever | None = None,
        answer_generator: AnswerGenerator | None = None,
        cache: RedisCache | None = None) -> None:
        self.retriever = (
            retriever if retriever is not None else DocumentRetriever()
        )
        self.answer_generator = (
            answer_generator
            if answer_generator is not None
            else AnswerGenerator()
        )
        self.cache = cache if cache is not None else RedisCache()

    @staticmethod
    def _answer_to_dict(answer: GeneratedAnswer) -> dict[str, Any]:
        return {
            "answer": answer.answer,
            "citations": [
                asdict(citation) for citation in answer.citations
            ],
            "not_found": answer.not_found,
            "model": answer.model,
        }

    @staticmethod
    def _answer_from_dict(data: dict[str, Any]) -> GeneratedAnswer:
        """Validate cached answer data before returning it."""
        if not isinstance(data, dict):
            raise QueryServiceError("Cached answer must be an object.")

        answer_text = data.get("answer")
        citations_data = data.get("citations")
        not_found = data.get("not_found")
        model = data.get("model")

        if not isinstance(answer_text, str) or not answer_text.strip():
            raise QueryServiceError("Cached answer text is invalid.")

        if not isinstance(citations_data, list):
            raise QueryServiceError("Cached citations are invalid.")

        if not isinstance(not_found, bool):
            raise QueryServiceError("Cached not_found value is invalid.")

        if not isinstance(model, str) or not model.strip():
            raise QueryServiceError("Cached model name is invalid.")

        citations: list[AnswerCitation] = []

        for item in citations_data:
            if not isinstance(item, dict):
                raise QueryServiceError("A cached citation is invalid.")

            reference = item.get("reference")
            source_file = item.get("source_file")
            headings = item.get("headings")
            page_numbers = item.get("page_numbers")
            relevance_score = item.get("relevance_score")

            if not isinstance(reference, str) or not reference.strip():
                raise QueryServiceError("Cached citation reference is invalid.")

            if not isinstance(source_file, str) or not source_file.strip():
                raise QueryServiceError("Cached citation source is invalid.")

            if not isinstance(headings, list) or not all(
                isinstance(value, str) for value in headings
            ):
                raise QueryServiceError("Cached citation headings are invalid.")

            if not isinstance(page_numbers, list) or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in page_numbers
            ):
                raise QueryServiceError("Cached citation pages are invalid.")

            if (
                not isinstance(relevance_score, (int, float))
                or isinstance(relevance_score, bool)
            ):
                raise QueryServiceError(
                    "Cached citation relevance score is invalid."
                )

            citations.append(
                AnswerCitation(
                    reference=reference,
                    source_file=source_file,
                    headings=headings,
                    page_numbers=page_numbers,
                    relevance_score=float(relevance_score),
                )
            )

        if not_found and citations:
            raise QueryServiceError(
                "A cached not-found answer cannot contain citations."
            )

        if not not_found and not citations:
            raise QueryServiceError(
                "A cached grounded answer must contain citations."
            )

        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            not_found=not_found,
            model=model,
        )

    @staticmethod
    def _evidence_records(
        chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
        return [
            {
                "document_id": chunk.document_id,
                "document_version": chunk.document_version,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        ]

    @staticmethod
    def _not_found_answer() -> GeneratedAnswer:
        return GeneratedAnswer(
            answer=(
                "I couldn't find sufficient information in the "
                "available documents to answer this question."
            ),
            citations=[],
            not_found=True,
            model="not_applicable",
        )

    def answer_question(
        self,
        question: str,
        trusted_access_tags: list[str] | tuple[str, ...]) -> GeneratedAnswer:
        """Answer a question using trusted permissions and current evidence."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        if isinstance(trusted_access_tags, (str, bytes)):
            raise ValueError("Trusted access tags must be a sequence.")

        if not isinstance(trusted_access_tags, (list, tuple)):
            raise ValueError("Trusted access tags must be a list or tuple.")

        if any(
            not isinstance(tag, str) or not tag.strip()
            for tag in trusted_access_tags
        ):
            raise ValueError("Trusted access tags must be non-empty strings.")

        normalized_tags = list(
            dict.fromkeys(tag.strip() for tag in trusted_access_tags)
        )

        # No permissions means no retrieval and no cache access.
        if not normalized_tags:
            return self._not_found_answer()

        try:
            # Generate one vector and reuse it for Qdrant and semantic cache.
            query_embedding = self.retriever.generate_query_embedding(question)

            # Current authorization and current evidence are checked first.
            chunks = self.retriever.retrieve(
                question=question,
                allowed_access_tags=normalized_tags,
                query_embedding=query_embedding,
            )

            # Do not cache an empty-evidence response.
            if not chunks:
                return self.answer_generator.generate_answer(question, [])

            permission_fingerprint = build_permission_fingerprint(
                normalized_tags
            )

            evidence_fingerprint = build_evidence_fingerprint(
                self._evidence_records(chunks)
            )

            # Exact cache lookup.
            exact_data = self.cache.get_exact(
                question=question,
                permission_fingerprint=permission_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
            )

            if exact_data is not None:
                return self._answer_from_dict(exact_data)

            # Semantic cache lookup, isolated by the same permissions/evidence.
            semantic_data = self.cache.get_semantic(
                embedding=query_embedding,
                permission_fingerprint=permission_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
            )

            if semantic_data is not None:
                answer = self._answer_from_dict(semantic_data)

                # A semantic cache hit is never allowed to return not-found.
                if not answer.not_found:
                    return answer

            # Cache miss: generate from the currently retrieved evidence.
            generated = self.answer_generator.generate_answer(
                question,
                chunks,
            )

            answer_data = self._answer_to_dict(generated)

            # Exact cache may store a not-found answer for this exact question
            # and exact evidence set.
            self.cache.set_exact(
                question=question,
                permission_fingerprint=permission_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                answer_data=answer_data,
            )

            # Semantic cache must never store not-found answers.
            if not generated.not_found:
                self.cache.set_semantic(
                    question=question,
                    embedding=query_embedding,
                    permission_fingerprint=permission_fingerprint,
                    evidence_fingerprint=evidence_fingerprint,
                    answer_data=answer_data,
                )

            return generated

        except Exception as exc:
            if isinstance(exc, (ValueError, QueryServiceError)):
                raise

            raise QueryServiceError(
                "The query could not be completed."
            ) from exc

    def close(self) -> None:
        """Close service clients."""
        errors = []

        for component in (
            self.retriever,
            self.answer_generator,
            self.cache,
        ):
            close_method = getattr(component, "close", None)

            if callable(close_method):
                try:
                    close_method()
                except Exception as exc:
                    errors.append(exc)

        if errors:
            raise QueryServiceError(
                "One or more query-service connections could not be closed."
            ) from errors[0]
