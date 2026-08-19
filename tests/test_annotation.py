"""Tests for the annotation class API client methods.

These verify that create_annotation and update_annotation build the correct
endpoint and payload, with a focus on max_area_count. The HTTP layer
(client.api.*_request) is stubbed so no real request is made.
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


# --- create_annotation -----------------------------------------------------


def test_create_annotation_defaults_max_area_count_to_one(monkeypatch, client):
    calls = _capture(monkeypatch, client, "post_request", return_value="anno-id")

    client.create_annotation(
        project="my-project", type="segmentation", value="cat", title="Cat"
    )

    assert calls[0]["endpoint"] == "annotations"
    assert calls[0]["kwargs"]["payload"] == {
        "project": "my-project",
        "type": "segmentation",
        "value": "cat",
        "title": "Cat",
        "maxAreaCount": 1,
    }


def test_create_annotation_with_max_area_count(monkeypatch, client):
    calls = _capture(monkeypatch, client, "post_request", return_value="anno-id")

    client.create_annotation(
        project="my-project",
        type="segmentation",
        value="cat",
        title="Cat",
        max_area_count=10,
    )

    assert calls[0]["kwargs"]["payload"]["maxAreaCount"] == 10


def test_create_annotation_without_max_area_count_limit(monkeypatch, client):
    calls = _capture(monkeypatch, client, "post_request", return_value="anno-id")

    client.create_annotation(
        project="my-project",
        type="segmentation",
        value="cat",
        title="Cat",
        max_area_count=None,
    )

    # None is sent as an explicit null, which means no limit on the server side
    assert calls[0]["kwargs"]["payload"]["maxAreaCount"] is None


# --- update_annotation -----------------------------------------------------


def test_update_annotation_omits_max_area_count_by_default(monkeypatch, client):
    calls = _capture(monkeypatch, client, "put_request", return_value="anno-id")

    client.update_annotation(annotation_id="anno-id", title="Cat")

    assert calls[0]["endpoint"] == "annotations/anno-id"
    assert calls[0]["kwargs"]["payload"] == {"title": "Cat"}


def test_update_annotation_with_max_area_count(monkeypatch, client):
    calls = _capture(monkeypatch, client, "put_request", return_value="anno-id")

    client.update_annotation(annotation_id="anno-id", max_area_count=10)

    assert calls[0]["kwargs"]["payload"] == {"maxAreaCount": 10}


def test_update_annotation_without_max_area_count_limit(monkeypatch, client):
    calls = _capture(monkeypatch, client, "put_request", return_value="anno-id")

    client.update_annotation(annotation_id="anno-id", max_area_count=None)

    # None is sent as an explicit null, which means no limit on the server side
    assert calls[0]["kwargs"]["payload"] == {"maxAreaCount": None}
