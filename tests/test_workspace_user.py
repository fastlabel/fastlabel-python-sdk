"""Tests for the workspace user API client methods.

These verify that get/create/update/delete_workspace_user build the correct
endpoint, query params and payload. The HTTP layer (client.api.*_request) is
stubbed so no real request is made.
"""

import pytest

import fastlabel


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("FASTLABEL_ACCESS_TOKEN", "dummy-token")
    return fastlabel.Client()


def _capture(monkeypatch, client, method_name, return_value=None):
    """Replace an api.*_request method with a recorder and return the calls list."""
    calls = []

    def fake(endpoint, *args, **kwargs):
        calls.append({"endpoint": endpoint, "args": args, "kwargs": kwargs})
        return return_value

    monkeypatch.setattr(client.api, method_name, fake)
    return calls


# --- get_workspace_users ---------------------------------------------------


def test_get_workspace_users_default(monkeypatch, client):
    calls = _capture(monkeypatch, client, "get_request", return_value=[])

    client.get_workspace_users()

    assert calls[0]["endpoint"] == "workspaces-users"
    # keyword/offset are omitted, limit defaults to 20
    assert calls[0]["kwargs"]["params"] == {"limit": 20}


def test_get_workspace_users_with_params(monkeypatch, client):
    calls = _capture(monkeypatch, client, "get_request", return_value=[])

    client.get_workspace_users(keyword="john", offset=10, limit=50)

    assert calls[0]["kwargs"]["params"] == {
        "keyword": "john",
        "offset": 10,
        "limit": 50,
    }


def test_get_workspace_users_offset_zero_included(monkeypatch, client):
    calls = _capture(monkeypatch, client, "get_request", return_value=[])

    client.get_workspace_users(offset=0)

    # offset=0 should still be sent (is not None), keyword empty is omitted
    assert calls[0]["kwargs"]["params"] == {"offset": 0, "limit": 20}


# --- create_workspace_user -------------------------------------------------


def test_create_workspace_user_without_modules(monkeypatch, client):
    calls = _capture(monkeypatch, client, "post_request", return_value={})

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


def test_update_workspace_user_role(monkeypatch, client):
    calls = _capture(monkeypatch, client, "put_request", return_value={})

    client.update_workspace_user(email="john@example.com", role="owner")

    assert calls[0]["endpoint"] == "workspaces-users/internal-users"
    assert calls[0]["kwargs"]["payload"] == {
        "email": "john@example.com",
        "role": "owner",
    }


# --- delete_workspace_user -------------------------------------------------


def test_delete_workspace_user(monkeypatch, client):
    # deletion is performed via PUT with role='none' (no DELETE endpoint)
    calls = _capture(monkeypatch, client, "put_request", return_value=None)

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
def test_create_module_permissions_single(monkeypatch, client, module, expected_path):
    calls = _capture(monkeypatch, client, "post_request", return_value=module)

    # a single module string is accepted (not only a list)
    result = client.create_workspace_user_module_permissions(
        email="john@example.com", modules=module
    )

    assert len(calls) == 1
    assert calls[0]["endpoint"] == expected_path
    assert calls[0]["kwargs"]["payload"] == {"email": "john@example.com"}
    assert result == [module]


def test_create_module_permissions_multiple(monkeypatch, client):
    calls = _capture(monkeypatch, client, "post_request", return_value="ok")

    result = client.create_workspace_user_module_permissions(
        email="john@example.com", modules=["annotation", "dataset"]
    )

    assert [c["endpoint"] for c in calls] == [
        "function-resource-permissions/annotation/internal-users",
        "function-resource-permissions/dataset/internal-users",
    ]
    assert all(c["kwargs"]["payload"] == {"email": "john@example.com"} for c in calls)
    assert result == ["ok", "ok"]


def test_create_module_permissions_invalid_module(monkeypatch, client):
    _capture(monkeypatch, client, "post_request", return_value=None)

    with pytest.raises(fastlabel.exceptions.FastLabelInvalidException):
        client.create_workspace_user_module_permissions(
            email="john@example.com", modules="unknown"
        )


# --- delete_workspace_user_module_permissions ------------------------------


def test_delete_module_permissions_single(monkeypatch, client):
    calls = _capture(monkeypatch, client, "delete_request", return_value=None)

    client.delete_workspace_user_module_permissions(
        email="john@example.com", modules="modelDev"
    )

    assert len(calls) == 1
    assert calls[0]["endpoint"] == "function-resource-permissions"
    assert calls[0]["kwargs"]["payload"] == {
        "email": "john@example.com",
        "resource": "modelDev",
    }


def test_delete_module_permissions_multiple(monkeypatch, client):
    calls = _capture(monkeypatch, client, "delete_request", return_value=None)

    client.delete_workspace_user_module_permissions(
        email="john@example.com", modules=["annotation", "modelDev"]
    )

    assert [c["kwargs"]["payload"]["resource"] for c in calls] == [
        "annotation",
        "modelDev",
    ]
    assert all(c["endpoint"] == "function-resource-permissions" for c in calls)


def test_delete_module_permissions_invalid_module(monkeypatch, client):
    _capture(monkeypatch, client, "delete_request", return_value=None)

    with pytest.raises(fastlabel.exceptions.FastLabelInvalidException):
        client.delete_workspace_user_module_permissions(
            email="john@example.com", modules="unknown"
        )
