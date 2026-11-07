
import pytest

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from backend.api.dependencies import (AuthenticatedUser,_authenticate_user)
from backend.auth.jwt_validator import CognitoTokenValidationError
from backend.auth.permissions import (PermissionConfigurationError,PermissionMappingError)


class FakeJWTValidator:
    def __init__(self, claims=None, error=None):
        self.claims = claims
        self.error = error

    def validate_access_token(self, token):
        if self.error:
            raise self.error

        return self.claims


def bearer(token="valid-token"):
    return HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials=token,
    )


def test_valid_user_returns_authenticated_context(monkeypatch):
    claims = {
        "sub": "user-123",
        "cognito:groups": ["HR"],
    }

    monkeypatch.setattr(
        "backend.api.dependencies.get_user_access_tags",
        lambda claims: ("hr",),
    )

    result = _authenticate_user(
        bearer(),
        FakeJWTValidator(claims=claims),
    )

    assert result == AuthenticatedUser(
        subject="user-123",
        groups=("HR",),
        access_tags=("hr",),
    )


def test_missing_credentials_returns_401():
    with pytest.raises(HTTPException) as exc:
        _authenticate_user(None, FakeJWTValidator())

    assert exc.value.status_code == 401
    assert exc.value.headers["WWW-Authenticate"] == "Bearer"


def test_invalid_token_returns_401():
    validator = FakeJWTValidator(
        error=CognitoTokenValidationError("Invalid token")
    )

    with pytest.raises(HTTPException) as exc:
        _authenticate_user(bearer(), validator)

    assert exc.value.status_code == 401


def test_empty_token_returns_401():
    with pytest.raises(HTTPException) as exc:
        _authenticate_user(
            bearer(""),
            FakeJWTValidator(),
        )

    assert exc.value.status_code == 401


def test_missing_subject_returns_401():
    claims = {
        "cognito:groups": ["HR"],
    }

    with pytest.raises(HTTPException) as exc:
        _authenticate_user(
            bearer(),
            FakeJWTValidator(claims),
        )

    assert exc.value.status_code == 401


def test_missing_permissions_returns_403(monkeypatch):
    claims = {
        "sub": "user-123",
        "cognito:groups": [],
    }

    def reject_permissions(claims):
        raise PermissionMappingError("No approved groups")

    monkeypatch.setattr(
        "backend.api.dependencies.get_user_access_tags",
        reject_permissions,
    )

    with pytest.raises(HTTPException) as exc:
        _authenticate_user(
            bearer(),
            FakeJWTValidator(claims),
        )

    assert exc.value.status_code == 403


def test_unknown_group_returns_403(monkeypatch):
    claims = {
        "sub": "user-123",
        "cognito:groups": ["Unknown"],
    }

    def reject_permissions(claims):
        raise PermissionMappingError("Unknown group")

    monkeypatch.setattr(
        "backend.api.dependencies.get_user_access_tags",
        reject_permissions,
    )

    with pytest.raises(HTTPException) as exc:
        _authenticate_user(
            bearer(),
            FakeJWTValidator(claims),
        )

    assert exc.value.status_code == 403


def test_invalid_permission_configuration_returns_500(monkeypatch):
    claims = {
        "sub": "user-123",
        "cognito:groups": ["HR"],
    }

    def invalid_configuration(claims):
        raise PermissionConfigurationError("Invalid mapping")

    monkeypatch.setattr(
        "backend.api.dependencies.get_user_access_tags",
        invalid_configuration,
    )

    with pytest.raises(HTTPException) as exc:
        _authenticate_user(
            bearer(),
            FakeJWTValidator(claims),
        )

    assert exc.value.status_code == 500


def test_client_supplied_permissions_are_not_used(monkeypatch):
    claims = {
        "sub": "user-123",
        "cognito:groups": ["HR"],
        "access_tags": ["ADMIN"],
    }

    monkeypatch.setattr(
        "backend.api.dependencies.get_user_access_tags",
        lambda claims: ("hr",),
    )

    result = _authenticate_user(
        bearer(),
        FakeJWTValidator(claims),
    )

    assert result.access_tags == ("hr",)
    assert "ADMIN" not in result.access_tags


def test_malformed_group_claim_returns_403(monkeypatch):
    claims = {
        "sub": "user-123",
        "cognito:groups": "HR",
    }

    def reject_permissions(claims):
        raise PermissionMappingError("Malformed group claim")

    monkeypatch.setattr(
        "backend.api.dependencies.get_user_access_tags",
        reject_permissions,
    )

    with pytest.raises(HTTPException) as exc:
        _authenticate_user(
            bearer(),
            FakeJWTValidator(claims),
        )

    assert exc.value.status_code == 403
