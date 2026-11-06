
import json
from types import SimpleNamespace
import pytest
from backend.query.answer_generator import (AnswerGenerationError,AnswerGenerator,NOT_FOUND_MESSAGE)
from backend.query.retrieval import RetrievedChunk


class FakeOpenAIClient:
    def __init__(self, response_content=None, error=None):
        self.response_content = response_content
        self.error = error
        self.call_count = 0
        self.closed = False

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.call_count += 1

        if self.error:
            raise self.error

        message = SimpleNamespace(content=self.response_content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    def close(self):
        self.closed = True


@pytest.fixture
def retrieved_chunk():
    return RetrievedChunk(
        chunk_index=5,
        text="Employees are eligible for 26 weeks of maternity leave.",
        source_file="IT_Industry_HR_Policies.pdf",
        headings=["5. Leave Policy"],
        page_numbers=[2],
        relevance_score=0.67,
    )


def make_response(found, answer, citations):
    return json.dumps({
        "found": found,
        "answer": answer,
        "citations": citations,
    })


def test_grounded_answer_with_verified_citation(retrieved_chunk):
    fake = FakeOpenAIClient(
        response_content=make_response(
            True,
            "Employees are eligible for 26 weeks of maternity leave [C1].",
            ["C1"],
        )
    )

    generator = AnswerGenerator(client=fake)
    result = generator.generate_answer(
        "How many weeks of maternity leave are available?",
        [retrieved_chunk],
    )

    assert result.not_found is False
    assert "[C1]" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].source_file == "IT_Industry_HR_Policies.pdf"
    assert result.citations[0].headings == ["5. Leave Policy"]
    assert result.citations[0].page_numbers == [2]
    assert fake.call_count == 1


def test_empty_retrieval_skips_openai_call():
    fake = FakeOpenAIClient()
    generator = AnswerGenerator(client=fake)

    result = generator.generate_answer("What is the leave policy?", [])

    assert result.not_found is True
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.citations == []
    assert fake.call_count == 0


def test_insufficient_evidence_returns_not_found(retrieved_chunk):
    fake = FakeOpenAIClient(
        response_content=make_response(False, "Insufficient evidence.", [])
    )

    generator = AnswerGenerator(client=fake)
    result = generator.generate_answer("What is the office address?", [retrieved_chunk])

    assert result.not_found is True
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.citations == []


def test_fabricated_citation_is_rejected(retrieved_chunk):
    fake = FakeOpenAIClient(
        response_content=make_response(
            True,
            "Employees receive 30 weeks of leave [C9].",
            ["C9"],
        )
    )

    generator = AnswerGenerator(client=fake)

    with pytest.raises(AnswerGenerationError, match="invalid or unverifiable"):
        generator.generate_answer("What is the leave policy?", [retrieved_chunk])


def test_missing_inline_citation_is_rejected(retrieved_chunk):
    fake = FakeOpenAIClient(
        response_content=make_response(
            True,
            "Employees are eligible for 26 weeks of maternity leave.",
            ["C1"],
        )
    )

    generator = AnswerGenerator(client=fake)

    with pytest.raises(AnswerGenerationError, match="invalid or unverifiable"):
        generator.generate_answer("What is the leave policy?", [retrieved_chunk])


def test_api_error_is_handled(retrieved_chunk):
    fake = FakeOpenAIClient(error=RuntimeError("Simulated API failure"))
    generator = AnswerGenerator(client=fake)

    with pytest.raises(AnswerGenerationError, match="request failed"):
        generator.generate_answer("What is the leave policy?", [retrieved_chunk])


def test_invalid_question_is_rejected():
    fake = FakeOpenAIClient()
    generator = AnswerGenerator(client=fake)

    with pytest.raises(ValueError):
        generator.generate_answer(" ", [])

    assert fake.call_count == 0


def test_client_close():
    fake = FakeOpenAIClient()
    generator = AnswerGenerator(client=fake)

    generator.close()

    assert fake.closed is True
