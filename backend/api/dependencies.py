
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.auth.cognito import CognitoConfigurationError
from backend.auth.jwt_validator import (CognitoJWTValidator,CognitoTokenValidationError,get_jwt_validator)
from backend.auth.permissions import (PermissionConfigurationError,PermissionMappingError,get_user_access_tags)


@dataclass(frozen=True)
class AuthenticatedUser:
    subject: str
    groups: tuple[str, ...]
    access_tags: tuple[str, ...]


security = HTTPBearer(auto_error=False)


def _authenticate_user(
    credentials: HTTPAuthorizationCredentials | None,
    jwt_validator: CognitoJWTValidator) -> AuthenticatedUser:
    """Validate credentials and construct the trusted user context."""

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        claims = jwt_validator.validate_access_token(credentials.credentials)
    except CognitoTokenValidationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    subject = claims.get("sub")

    if not isinstance(subject, str) or not subject.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user identity",
        )

    try:
        access_tags = get_user_access_tags(claims)
    except PermissionMappingError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have approved permissions",
        )
    except PermissionConfigurationError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Permission configuration error",
        )

    groups = claims.get("cognito:groups", [])

    return AuthenticatedUser(
        subject=subject,
        groups=tuple(groups),
        access_tags=tuple(access_tags),
    )


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> AuthenticatedUser:
    """FastAPI dependency with lazy Cognito validator initialization."""

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        jwt_validator = get_jwt_validator()
    except CognitoConfigurationError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication configuration error",
        )

    return _authenticate_user(credentials, jwt_validator)


CurrentUser = Depends(get_current_user)
