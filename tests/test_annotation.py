"""Tests for the annotation class API client methods.

These verify that create_annotation and update_annotation build the correct
endpoint and payload, with a focus on max_area_count. The HTTP layer
(client.api.*_request) is stubbed so no real request is made.
"""

import copy
import pickle

import pytest

import fastlabel

# --- create_annotation -----------------------------------------------------


def test_create_annotation_omits_max_area_count_by_default(client, capture_request):
    calls = capture_request(client, "post_request", return_value="anno-id")

    client.create_annotation(
        project="my-project", type="segmentation", value="cat", title="Cat"
    )

    # The field is left out entirely so the API applies its own default of 1
    assert calls[0]["endpoint"] == "annotations"
    assert calls[0]["kwargs"]["payload"] == {
        "project": "my-project",
        "type": "segmentation",
        "value": "cat",
        "title": "Cat",
    }


def test_create_annotation_with_max_area_count(client, capture_request):
    calls = capture_request(client, "post_request", return_value="anno-id")

    client.create_annotation(
        project="my-project",
        type="segmentation",
        value="cat",
        title="Cat",
        max_area_count=10,
    )

    assert calls[0]["kwargs"]["payload"]["maxAreaCount"] == 10


def test_create_annotation_without_max_area_count_limit(client, capture_request):
    calls = capture_request(client, "post_request", return_value="anno-id")

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


def test_update_annotation_omits_max_area_count_by_default(client, capture_request):
    calls = capture_request(client, "put_request", return_value="anno-id")

    client.update_annotation(annotation_id="anno-id", title="Cat")

    assert calls[0]["endpoint"] == "annotations/anno-id"
    assert calls[0]["kwargs"]["payload"] == {"title": "Cat"}


def test_update_annotation_with_max_area_count(client, capture_request):
    calls = capture_request(client, "put_request", return_value="anno-id")

    client.update_annotation(annotation_id="anno-id", max_area_count=10)

    assert calls[0]["kwargs"]["payload"] == {"maxAreaCount": 10}


def test_update_annotation_without_max_area_count_limit(client, capture_request):
    calls = capture_request(client, "put_request", return_value="anno-id")

    client.update_annotation(annotation_id="anno-id", max_area_count=None)

    # None is sent as an explicit null, which means no limit on the server side
    assert calls[0]["kwargs"]["payload"] == {"maxAreaCount": None}


# --- the "not passed" marker -----------------------------------------------


@pytest.mark.parametrize(
    "round_trip",
    [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))],
    ids=["deepcopy", "pickle"],
)
def test_create_annotation_keeps_marker_meaning_after_round_trip(
    client, capture_request, round_trip
):
    calls = capture_request(client, "post_request", return_value="anno-id")

    # Callers that collect keyword arguments and pass them around must still
    # get "omitted" out of the default, which relies on the marker's identity
    kwargs = round_trip(
        {
            "project": "my-project",
            "type": "segmentation",
            "value": "cat",
            "title": "Cat",
            "max_area_count": fastlabel._UNSET,
        }
    )
    client.create_annotation(**kwargs)

    assert "maxAreaCount" not in calls[0]["kwargs"]["payload"]
