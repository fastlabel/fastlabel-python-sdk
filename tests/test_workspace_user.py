"""Tests for the workspace user API client methods.

These verify that get/create/update/delete_workspace_user build the correct
endpoint, query params and payload. The HTTP layer (client.api.*_request) is
stubbed so no real request is made.
"""

import pytest

import fastlabel

# --- get_workspace_users ---------------------------------------------------


def test_get_workspace_users_default(client, capture_request):
    calls = capture_request(client, "get_request", return_value=[])

    client.get_workspace_users()

    assert calls[0]["endpoint"] == "workspaces-users"
    # keyword/offset are omitted, limit defaults to 20
    assert calls[0]["kwargs"]["params"] == {"limit": 20}


def test_get_workspace_users_with_params(client, capture_request):
    calls = capture_request(client, "get_request", return_value=[])

    client.get_workspace_users(keyword="john", offset=10, limit=50)

    assert calls[0]["kwargs"]["params"] == {
        "keyword": "john",
        "offset": 10,
        "limit": 50,
    }


def test_get_workspace_users_offset_zero_included(client, capture_request):
    calls = capture_request(client, "get_request", return_value=[])

    client.get_workspace_users(offset=0)

    # offset=0 should still be sent (is not None), keyword empty is omitted
    assert calls[0]["kwargs"]["params"] == {"offset": 0, "limit": 20}


# --- create_workspace_user -------------------------------------------------


def test_create_workspace_user_without_modules(client, capture_request):
    calls = capture_request(client, "post_request", return_value={})

    client.create_workspace_user(
        name="John Doe",
        email="john@example.com",
        language="en",
        role="member",
    )

    assert calls[0]["endpoint"] == "workspaces-users/internal-users"
    assert calls[0]["kwargs"]["payload"] == {
        "name": "John Doe",
        "email": "john@example.com",
        "language": "en",
        "role": "member",
    }


# --- update_workspace_user -------------------------------------------------


def test_update_workspace_user_role(client, capture_request):
    calls = capture_request(client, "put_request", return_value={})

    client.update_workspace_user(email="john@example.com", role="owner")

    assert calls[0]["endpoint"] == "workspaces-users/internal-users"
    assert calls[0]["kwargs"]["payload"] == {
        "email": "john@example.com",
        "role": "owner",
    }


# --- delete_workspace_user -------------------------------------------------


def test_delete_workspace_user(client, capture_request):
    # deletion is performed via PUT with role='none' (no DELETE endpoint)
    calls = capture_request(client, "put_request", return_value=None)

    result = client.delete_workspace_user(email="john@example.com")

    assert calls[0]["endpoint"] == "workspaces-users/internal-users"
    assert calls[0]["kwargs"]["payload"] == {
        "email": "john@example.com",
        "role": "none",
    }
    assert result is None


# --- create_workspace_user_module_permissions ------------------------------


@pytest.mark.parametrize(
    "module, expected_path",
    [
        ("annotation", "function-resource-permissions/annotation/internal-users"),
        ("dataset", "function-resource-permissions/dataset/internal-users"),
        ("modelDev", "function-resource-permissions/model-dev/internal-users"),
    ],
)
def test_create_module_permissions_single(
    client, capture_request, module, expected_path
):
    calls = capture_request(client, "post_request", return_value=module)

    # a single module string is accepted (not only a list)
    result = client.create_workspace_user_module_permissions(
        email="john@example.com", modules=module
    )

    assert len(calls) == 1
    assert calls[0]["endpoint"] == expected_path
    assert calls[0]["kwargs"]["payload"] == {"email": "john@example.com"}
    assert result == [module]


def test_create_module_permissions_multiple(client, capture_request):
    calls = capture_request(client, "post_request", return_value="ok")

    result = client.create_workspace_user_module_permissions(
        email="john@example.com", modules=["annotation", "dataset"]
    )

    assert [c["endpoint"] for c in calls] == [
        "function-resource-permissions/annotation/internal-users",
        "function-resource-permissions/dataset/internal-users",
    ]
    assert all(c["kwargs"]["payload"] == {"email": "john@example.com"} for c in calls)
    assert result == ["ok", "ok"]


def test_create_module_permissions_invalid_module(client, capture_request):
    capture_request(client, "post_request", return_value=None)

    with pytest.raises(fastlabel.exceptions.FastLabelInvalidException):
        client.create_workspace_user_module_permissions(
            email="john@example.com", modules="unknown"
        )


# --- delete_workspace_user_module_permissions ------------------------------


def test_delete_module_permissions_single(client, capture_request):
    calls = capture_request(client, "delete_request", return_value=None)

    client.delete_workspace_user_module_permissions(
        email="john@example.com", modules="modelDev"
    )

    assert len(calls) == 1
    assert calls[0]["endpoint"] == "function-resource-permissions"
    assert calls[0]["kwargs"]["payload"] == {
        "email": "john@example.com",
        "resource": "modelDev",
    }


def test_delete_module_permissions_multiple(client, capture_request):
    calls = capture_request(client, "delete_request", return_value=None)

    client.delete_workspace_user_module_permissions(
        email="john@example.com", modules=["annotation", "modelDev"]
    )

    assert [c["kwargs"]["payload"]["resource"] for c in calls] == [
        "annotation",
        "modelDev",
    ]
    assert all(c["endpoint"] == "function-resource-permissions" for c in calls)


def test_delete_module_permissions_invalid_module(client, capture_request):
    capture_request(client, "delete_request", return_value=None)

    with pytest.raises(fastlabel.exceptions.FastLabelInvalidException):
        client.delete_workspace_user_module_permissions(
            email="john@example.com", modules="unknown"
        )
