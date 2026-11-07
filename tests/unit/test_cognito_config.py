
import pytest

from backend.auth.cognito import (CognitoConfig,CognitoConfigurationError)


def test_cognito_configuration_success(monkeypatch):
    monkeypatch.setenv("COGNITO_REGION", "ap-south-1")
    monkeypatch.setenv(
        "COGNITO_USER_POOL_ID",
        "ap-south-1_example123"
    )
    monkeypatch.setenv(
        "COGNITO_APP_CLIENT_ID",
        "example-client-id"
    )

    config = CognitoConfig.from_environment()

    assert config.region == "ap-south-1"
    assert config.user_pool_id == "ap-south-1_example123"
    assert config.app_client_id == "example-client-id"

    assert config.issuer == (
        "https://cognito-idp.ap-south-1.amazonaws.com/"
        "ap-south-1_example123"
    )

    assert config.jwks_url == (
        config.issuer + "/.well-known/jwks.json"
    )


def test_missing_region_fails(monkeypatch):
    monkeypatch.delenv("COGNITO_REGION", raising=False)
    monkeypatch.setenv(
        "COGNITO_USER_POOL_ID",
        "ap-south-1_example123"
    )
    monkeypatch.setenv(
        "COGNITO_APP_CLIENT_ID",
        "example-client-id"
    )

    with pytest.raises(CognitoConfigurationError):
        CognitoConfig.from_environment()


def test_region_mismatch_fails(monkeypatch):
    monkeypatch.setenv("COGNITO_REGION", "ap-south-1")
    monkeypatch.setenv(
        "COGNITO_USER_POOL_ID",
        "us-east-1_example123"
    )
    monkeypatch.setenv(
        "COGNITO_APP_CLIENT_ID",
        "example-client-id"
    )

    with pytest.raises(CognitoConfigurationError):
        CognitoConfig.from_environment()


def test_missing_client_id_fails(monkeypatch):
    monkeypatch.setenv("COGNITO_REGION", "ap-south-1")
    monkeypatch.setenv(
        "COGNITO_USER_POOL_ID",
        "ap-south-1_example123"
    )
    monkeypatch.delenv(
        "COGNITO_APP_CLIENT_ID",
        raising=False
    )

    with pytest.raises(CognitoConfigurationError):
        CognitoConfig.from_environment()
