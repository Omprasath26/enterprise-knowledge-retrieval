
import json
import os
import re
from collections.abc import Mapping
from typing import Any


class PermissionConfigurationError(Exception):
    """Raised when the Cognito permission mapping is invalid."""


class PermissionMappingError(Exception):
    """Raised when validated Cognito claims cannot be mapped safely."""


def _normalize_tag(value: str) -> str:
    """Normalize an access tag consistently."""
    return " ".join(value.split()).casefold()


def load_group_access_mapping() -> dict[str, tuple[str, ...]]:
    """
    Load the backend-controlled Cognito group-to-access-tag mapping.

    Expected environment variable:
    COGNITO_GROUP_ACCESS_TAGS='{"HR":["HR"],"Finance":["FINANCE"]}'
    """
    raw_mapping = os.getenv("COGNITO_GROUP_ACCESS_TAGS")

    if not raw_mapping:
        raise PermissionConfigurationError(
            "COGNITO_GROUP_ACCESS_TAGS is missing or empty."
        )

    try:
        parsed = json.loads(raw_mapping)
    except json.JSONDecodeError as exc:
        raise PermissionConfigurationError(
            "COGNITO_GROUP_ACCESS_TAGS must contain valid JSON."
        ) from exc

    if not isinstance(parsed, dict) or not parsed:
        raise PermissionConfigurationError(
            "The permission mapping must be a non-empty JSON object."
        )

    mapping: dict[str, tuple[str, ...]] = {}

    for group, tags in parsed.items():
        if not isinstance(group, str) or not group.strip():
            raise PermissionConfigurationError(
                "Every Cognito group name must be a non-empty string."
            )

        if not isinstance(tags, list) or not tags:
            raise PermissionConfigurationError(
                f"Access tags for group '{group}' must be a non-empty list."
            )

        normalized_tags = []

        for tag in tags:
            if not isinstance(tag, str) or not tag.strip():
                raise PermissionConfigurationError(
                    f"Group '{group}' contains an invalid access tag."
                )

            normalized_tags.append(_normalize_tag(tag))

        if any(not tag for tag in normalized_tags):
            raise PermissionConfigurationError(
                f"Group '{group}' contains an empty access tag."
            )

        mapping[group] = tuple(sorted(set(normalized_tags)))

    return mapping


def get_user_access_tags(
    claims: Mapping[str, Any],
    group_access_mapping: Mapping[str, tuple[str, ...]] | None = None) -> tuple[str, ...]:
    """
    Convert validated Cognito JWT claims into trusted application access tags.

    The claims must come from the JWT validation layer, not directly
    from a client request.
    """
    if not isinstance(claims, Mapping):
        raise PermissionMappingError("Validated token claims are required.")

    groups = claims.get("cognito:groups")

    if not isinstance(groups, list) or not groups:
        raise PermissionMappingError(
            "The authenticated user has no approved Cognito groups."
        )

    if any(not isinstance(group, str) or not group.strip() for group in groups):
        raise PermissionMappingError("The Cognito group claim is malformed.")

    if group_access_mapping is None:
        group_access_mapping = load_group_access_mapping()

    if not isinstance(group_access_mapping, Mapping) or not group_access_mapping:
        raise PermissionConfigurationError(
            "The Cognito group permission mapping is empty or invalid."
        )

    access_tags: set[str] = set()

    for group in groups:
        if group not in group_access_mapping:
            raise PermissionMappingError(
                f"Cognito group '{group}' is not approved for application access."
            )

        tags = group_access_mapping[group]

        if not isinstance(tags, (list, tuple)) or not tags:
            raise PermissionConfigurationError(
                f"The permission mapping for group '{group}' is invalid."
            )

        for tag in tags:
            if not isinstance(tag, str) or not tag.strip():
                raise PermissionConfigurationError(
                    f"Group '{group}' contains an invalid access tag."
                )

            normalized_tag = _normalize_tag(tag)

            if not normalized_tag:
                raise PermissionConfigurationError(
                    f"Group '{group}' contains an empty access tag."
                )

            access_tags.add(normalized_tag)

    if not access_tags:
        raise PermissionMappingError(
            "No application access permissions were resolved."
        )

    return tuple(sorted(access_tags))
