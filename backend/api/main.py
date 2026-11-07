
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend.api.dependencies import (AuthenticatedUser,get_current_user)
from backend.query.query_service import (QueryService,QueryServiceError)


app = FastAPI(
    title="Enterprise Knowledge Retrieval & RAG Assistant",
    description=(
        "Secure enterprise document retrieval and question-answering API."
    ),
    version="1.0.0",
)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=10000,
        description="Question to answer using authorized enterprise documents.",
    )


@lru_cache(maxsize=1)
def get_query_service() -> QueryService:
    """Create the query service lazily when the endpoint is first used."""
    return QueryService()


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "enterprise-knowledge-retrieval",
    }


@app.get("/api/v1/auth/me", tags=["Authentication"])
def get_authenticated_user(
    current_user: AuthenticatedUser = Depends(get_current_user)):
    return {
        "subject": current_user.subject,
        "groups": list(current_user.groups),
        "access_tags": list(current_user.access_tags),
    }


@app.post("/api/v1/query", tags=["Query"])
def query_documents(
    request: QueryRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    query_service: QueryService = Depends(get_query_service)):
    """Answer a question using the authenticated user's document permissions."""

    try:
        result = query_service.answer_question(
            question=request.question,
            trusted_access_tags=list(current_user.access_tags),
        )

        return {
            "answer": result.answer,
            "not_found": result.not_found,
            "model": result.model,
            "citations": [
                {
                    "reference": citation.reference,
                    "source_file": citation.source_file,
                    "headings": citation.headings,
                    "page_numbers": citation.page_numbers,
                    "relevance_score": citation.relevance_score,
                }
                for citation in result.citations
            ],
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except QueryServiceError as exc:
        raise HTTPException(
            status_code=500,
            detail="The query service could not complete the request.",
        ) from exc
