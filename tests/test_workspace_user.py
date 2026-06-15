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


def test_create_workspace_user_with_modules(monkeypatch, client):
    calls = _capture(monkeypatch, client, "post_request", return_value={})

    client.create_workspace_user(
        name="John Doe",
        email="john@example.com",
        language="ja",
        role="owner",
        modules=["annotation", "dataset"],
    )

    assert calls[0]["kwargs"]["payload"] == {
        "name": "John Doe",
        "email": "john@example.com",
        "language": "ja",
        "role": "owner",
        "modules": ["annotation", "dataset"],
    }


def test_create_workspace_user_empty_modules_sent(monkeypatch, client):
    calls = _capture(monkeypatch, client, "post_request", return_value={})

    client.create_workspace_user(
        name="John Doe",
        email="john@example.com",
        language="en",
        role="member",
        modules=[],
    )

    assert calls[0]["kwargs"]["payload"]["modules"] == []


# --- update_workspace_user -------------------------------------------------


def test_update_workspace_user_role_only(monkeypatch, client):
    calls = _capture(monkeypatch, client, "put_request", return_value={})

    client.update_workspace_user(id="wsu-1", role="owner")

    assert calls[0]["endpoint"] == "workspaces-users/internal-users/wsu-1"
    assert calls[0]["kwargs"]["payload"] == {"role": "owner"}


def test_update_workspace_user_modules_unchanged_when_none(monkeypatch, client):
    calls = _capture(monkeypatch, client, "put_request", return_value={})

    client.update_workspace_user(id="wsu-1", role="member")

    # modules omitted -> not present in payload (left unchanged server-side)
    assert "modules" not in calls[0]["kwargs"]["payload"]


def test_update_workspace_user_modules_sync(monkeypatch, client):
    calls = _capture(monkeypatch, client, "put_request", return_value={})

    client.update_workspace_user(id="wsu-1", modules=["annotation", "modelDev"])

    assert calls[0]["kwargs"]["payload"] == {"modules": ["annotation", "modelDev"]}


def test_update_workspace_user_empty_modules_revokes_all(monkeypatch, client):
    calls = _capture(monkeypatch, client, "put_request", return_value={})

    client.update_workspace_user(id="wsu-1", modules=[])

    # empty list is distinct from None: it is sent to revoke all permissions
    assert calls[0]["kwargs"]["payload"] == {"modules": []}


# --- delete_workspace_user -------------------------------------------------


def test_delete_workspace_user(monkeypatch, client):
    calls = _capture(monkeypatch, client, "delete_request", return_value=None)

    result = client.delete_workspace_user(id="wsu-1")

    assert calls[0]["endpoint"] == "workspaces-users/internal-users/wsu-1"
    assert result is None
