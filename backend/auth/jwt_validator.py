
from typing import Any

import jwt
from jwt import PyJWKClient
from jwt.exceptions import PyJWKClientError, PyJWTError

from backend.auth.cognito import (CognitoConfig,get_cognito_config)


class CognitoTokenValidationError(Exception):
    """Raised when a Cognito access token is invalid."""


class CognitoJWTValidator:
    """Validate Cognito access tokens using Cognito JWKS public keys."""

    ALGORITHM = "RS256"

    def __init__(self, config: CognitoConfig):
        self.config = config
        self.jwks_client = PyJWKClient(config.jwks_url)

    def validate_access_token(self, token: str) -> dict[str, Any]:
        if not isinstance(token, str) or not token.strip():
            raise CognitoTokenValidationError(
                "Access token must be a non-empty string."
            )

        try:
            # Inspect the header only to select the key.
            # The token is not trusted until jwt.decode verifies it.
            header = jwt.get_unverified_header(token)

            if header.get("alg") != self.ALGORITHM:
                raise CognitoTokenValidationError(
                    "Unsupported JWT signing algorithm."
                )

            if not isinstance(header.get("kid"), str) or not header["kid"]:
                raise CognitoTokenValidationError(
                    "JWT signing key identifier is missing."
                )

            signing_key = (
                self.jwks_client.get_signing_key_from_jwt(token)
            )

            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=[self.ALGORITHM],
                issuer=self.config.issuer,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iss": True,
                    # Cognito access tokens use client_id,
                    # not the ID-token audience claim.
                    "verify_aud": False,
                    "require": [
                        "exp",
                        "iss",
                        "sub",
                        "token_use",
                        "client_id",
                    ],
                },
            )

            if claims.get("token_use") != "access":
                raise CognitoTokenValidationError(
                    "Only Cognito access tokens are accepted."
                )

            if claims.get("client_id") != self.config.app_client_id:
                raise CognitoTokenValidationError(
                    "Token was issued for an unexpected app client."
                )

            subject = claims.get("sub")

            if not isinstance(subject, str) or not subject.strip():
                raise CognitoTokenValidationError(
                    "Token subject is invalid."
                )

            return claims

        except CognitoTokenValidationError:
            raise

        except (PyJWTError, PyJWKClientError, ValueError, TypeError) as exc:
            raise CognitoTokenValidationError(
                "Cognito access token validation failed."
            ) from exc


def get_jwt_validator() -> CognitoJWTValidator:
    """Create a validator using the configured Cognito settings."""
    return CognitoJWTValidator(get_cognito_config())
