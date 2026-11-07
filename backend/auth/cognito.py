
import os
import re
from dataclasses import dataclass
from functools import lru_cache


class CognitoConfigurationError(Exception):
    """Raised when Cognito configuration is missing or invalid."""


@dataclass(frozen=True)
class CognitoConfig:
    region: str
    user_pool_id: str
    app_client_id: str
    issuer: str
    jwks_url: str

    @classmethod
    def from_environment(cls) -> "CognitoConfig":
        region = os.getenv("COGNITO_REGION", "").strip()
        user_pool_id = os.getenv("COGNITO_USER_POOL_ID", "").strip()
        app_client_id = os.getenv("COGNITO_APP_CLIENT_ID", "").strip()

        if not region:
            raise CognitoConfigurationError(
                "COGNITO_REGION is not configured."
            )

        if not user_pool_id:
            raise CognitoConfigurationError(
                "COGNITO_USER_POOL_ID is not configured."
            )

        if not app_client_id:
            raise CognitoConfigurationError(
                "COGNITO_APP_CLIENT_ID is not configured."
            )

        # Basic AWS region format validation.
        if not re.fullmatch(
            r"[a-z]{2}(?:-gov)?-[a-z]+-\d",
            region
        ):
            raise CognitoConfigurationError(
                "COGNITO_REGION has an invalid format."
            )

        # Cognito User Pool IDs normally begin with the region.
        if not user_pool_id.startswith(f"{region}_"):
            raise CognitoConfigurationError(
                "User Pool ID does not match the configured region."
            )

        issuer = (
            f"https://cognito-idp.{region}.amazonaws.com/"
            f"{user_pool_id}"
        )

        jwks_url = f"{issuer}/.well-known/jwks.json"

        return cls(
            region=region,
            user_pool_id=user_pool_id,
            app_client_id=app_client_id,
            issuer=issuer,
            jwks_url=jwks_url,
        )


@lru_cache(maxsize=1)
def get_cognito_config() -> CognitoConfig:
    """Return the cached Cognito configuration."""
    return CognitoConfig.from_environment()
