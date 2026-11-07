
from fastapi.testclient import TestClient

from backend.api.dependencies import (AuthenticatedUser,get_current_user)
from backend.api.main import app


client = TestClient(app)


def test_health_endpoint_is_public():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "enterprise-knowledge-retrieval",
    }


def test_authenticated_endpoint_without_token_returns_401():
    response = client.get("/api/v1/auth/me")

    assert response.status_code == 401


def test_authenticated_endpoint_returns_user_context():
    expected_user = AuthenticatedUser(
        subject="test-user-123",
        groups=("HR",),
        access_tags=("hr",),
    )

    app.dependency_overrides[get_current_user] = lambda: expected_user

    try:
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer test-token"},
        )

        assert response.status_code == 200
        assert response.json() == {
            "subject": "test-user-123",
            "groups": ["HR"],
            "access_tags": ["hr"],
        }
    finally:
        app.dependency_overrides.clear()



def test_invalid_token_returns_401(monkeypatch):
    from backend.api.dependencies import get_current_user
    from backend.api.dependencies import get_jwt_validator
    from backend.auth.jwt_validator import CognitoTokenValidationError

    class FakeJWTValidator:
        def validate_access_token(self, token):
            raise CognitoTokenValidationError("Invalid token")

    monkeypatch.setattr(
        "backend.api.dependencies.get_jwt_validator",
        lambda: FakeJWTValidator(),
    )

    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer invalid-token"},
    )

    assert response.status_code == 401

