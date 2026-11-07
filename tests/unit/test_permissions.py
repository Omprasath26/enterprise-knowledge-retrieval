
import json

import pytest

from backend.auth.permissions import (PermissionConfigurationError,PermissionMappingError,get_user_access_tags,load_group_access_mapping)


@pytest.fixture
def mapping():
    return {
        "HR": ("hr",),
        "Finance": ("finance",),
        "Administrators": ("hr", "finance", "it"),
    }


def test_valid_hr_group(mapping):
    claims = {"cognito:groups": ["HR"]}

    assert get_user_access_tags(claims, mapping) == ("hr",)


def test_valid_finance_group(mapping):
    claims = {"cognito:groups": ["Finance"]}

    assert get_user_access_tags(claims, mapping) == ("finance",)


def test_multiple_approved_groups(mapping):
    claims = {"cognito:groups": ["HR", "Finance"]}

    assert get_user_access_tags(claims, mapping) == (
        "finance",
        "hr",
    )


def test_administrator_group(mapping):
    claims = {"cognito:groups": ["Administrators"]}

    assert get_user_access_tags(claims, mapping) == (
        "finance",
        "hr",
        "it",
    )


def test_unknown_group_is_rejected(mapping):
    claims = {"cognito:groups": ["Unknown"]}

    with pytest.raises(PermissionMappingError):
        get_user_access_tags(claims, mapping)


def test_missing_group_claim_is_rejected(mapping):
    with pytest.raises(PermissionMappingError):
        get_user_access_tags({}, mapping)


def test_empty_group_list_is_rejected(mapping):
    claims = {"cognito:groups": []}

    with pytest.raises(PermissionMappingError):
        get_user_access_tags(claims, mapping)


def test_malformed_group_claim_is_rejected(mapping):
    claims = {"cognito:groups": "HR"}

    with pytest.raises(PermissionMappingError):
        get_user_access_tags(claims, mapping)


def test_invalid_group_entry_is_rejected(mapping):
    claims = {"cognito:groups": ["HR", ""]}

    with pytest.raises(PermissionMappingError):
        get_user_access_tags(claims, mapping)


def test_duplicate_groups_are_deduplicated(mapping):
    claims = {"cognito:groups": ["HR", "HR"]}

    assert get_user_access_tags(claims, mapping) == ("hr",)


def test_non_mapping_claims_are_rejected(mapping):
    with pytest.raises(PermissionMappingError):
        get_user_access_tags(None, mapping)


def test_empty_mapping_is_rejected():
    with pytest.raises(PermissionConfigurationError):
        get_user_access_tags(
            {"cognito:groups": ["HR"]},
            {},
        )


def test_invalid_mapping_tag_is_rejected():
    mapping = {"HR": (" ",)}

    with pytest.raises(PermissionConfigurationError):
        get_user_access_tags(
            {"cognito:groups": ["HR"]},
            mapping,
        )


def test_load_mapping_from_environment(monkeypatch):
    config = {
        "HR": ["HR"],
        "Finance": ["FINANCE"],
    }

    monkeypatch.setenv(
        "COGNITO_GROUP_ACCESS_TAGS",
        json.dumps(config),
    )

    result = load_group_access_mapping()

    assert result == {
        "HR": ("hr",),
        "Finance": ("finance",),
    }


def test_missing_environment_mapping(monkeypatch):
    monkeypatch.delenv("COGNITO_GROUP_ACCESS_TAGS", raising=False)

    with pytest.raises(PermissionConfigurationError):
        load_group_access_mapping()


def test_invalid_environment_json(monkeypatch):
    monkeypatch.setenv("COGNITO_GROUP_ACCESS_TAGS", "{invalid")

    with pytest.raises(PermissionConfigurationError):
        load_group_access_mapping()
