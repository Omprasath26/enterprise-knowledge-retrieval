
import time
from types import SimpleNamespace
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from backend.auth.cognito import CognitoConfig
from backend.auth.jwt_validator import (CognitoJWTValidator,CognitoTokenValidationError)


@pytest.fixture
def rsa_keys():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    return private_key, private_key.public_key()


@pytest.fixture
def config():
    return CognitoConfig(
        region="ap-south-1",
        user_pool_id="ap-south-1_example123",
        app_client_id="test-client-id",
        issuer=(
            "https://cognito-idp.ap-south-1.amazonaws.com/"
            "ap-south-1_example123"
        ),
        jwks_url=(
            "https://cognito-idp.ap-south-1.amazonaws.com/"
            "ap-south-1_example123/.well-known/jwks.json"
        ),
    )


@pytest.fixture
def validator(config):
    return CognitoJWTValidator(config)


def create_token(
    private_key,
    config,
    **overrides):
    now = int(time.time())

    claims = {
        "sub": "test-user-123",
        "iss": config.issuer,
        "token_use": "access",
        "client_id": config.app_client_id,
        "iat": now,
        "exp": now + 300,
    }

    claims.update(overrides)

    return jwt.encode(
        claims,
        private_key,
        algorithm="RS256",
        headers={"kid": "test-key-id"},
    )


def mock_signing_key(validator, public_key):
    validator.jwks_client.get_signing_key_from_jwt = (
        lambda token: SimpleNamespace(key=public_key)
    )


def test_valid_access_token(validator, config, rsa_keys):
    private_key, public_key = rsa_keys
    mock_signing_key(validator, public_key)

    token = create_token(private_key, config)

    claims = validator.validate_access_token(token)

    assert claims["sub"] == "test-user-123"
    assert claims["token_use"] == "access"
    assert claims["client_id"] == config.app_client_id


def test_invalid_signature_is_rejected(
    validator,
    config,
    rsa_keys):
    private_key, _ = rsa_keys

    another_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    mock_signing_key(validator, another_key.public_key())
    token = create_token(private_key, config)

    with pytest.raises(CognitoTokenValidationError):
        validator.validate_access_token(token)


def test_expired_token_is_rejected(
    validator,
    config,
    rsa_keys):
    private_key, public_key = rsa_keys
    mock_signing_key(validator, public_key)

    token = create_token(
        private_key,
        config,
        exp=int(time.time()) - 60,
    )

    with pytest.raises(CognitoTokenValidationError):
        validator.validate_access_token(token)


def test_wrong_issuer_is_rejected(
    validator,
    config,
    rsa_keys):
    private_key, public_key = rsa_keys
    mock_signing_key(validator, public_key)

    token = create_token(
        private_key,
        config,
        iss="https://wrong-issuer.example.com",
    )

    with pytest.raises(CognitoTokenValidationError):
        validator.validate_access_token(token)


def test_wrong_client_id_is_rejected(
    validator,
    config,
    rsa_keys):
    private_key, public_key = rsa_keys
    mock_signing_key(validator, public_key)

    token = create_token(
        private_key,
        config,
        client_id="different-client-id",
    )

    with pytest.raises(CognitoTokenValidationError):
        validator.validate_access_token(token)


def test_id_token_is_rejected(
    validator,
    config,
    rsa_keys):
    private_key, public_key = rsa_keys
    mock_signing_key(validator, public_key)

    token = create_token(
        private_key,
        config,
        token_use="id",
    )

    with pytest.raises(CognitoTokenValidationError):
        validator.validate_access_token(token)


def test_empty_token_is_rejected(validator):
    with pytest.raises(CognitoTokenValidationError):
        validator.validate_access_token("")


def test_invalid_token_format_is_rejected(validator):
    with pytest.raises(CognitoTokenValidationError):
        validator.validate_access_token("not-a-valid-jwt")
