"""Tests for create_multi_image_classification_task.

The HTTP layer (client.api.post_request) is stubbed so no real request is made.
"""

import pytest

from fastlabel.exceptions import FastLabelInvalidException


def test_create_multi_image_classification_task_accepts_gif(
    client, capture_request, tmp_path
):
    (tmp_path / "1_front.gif").write_bytes(b"GIF89a")
    (tmp_path / "2_last.jpg").write_bytes(b"jpg")
    calls = capture_request(client, "post_request", return_value="task-id")

    client.create_multi_image_classification_task(
        project="my-project", name="sample", folder_path=str(tmp_path)
    )

    assert calls[0]["endpoint"] == "tasks/multi-image/classification"
    names = {c["name"] for c in calls[0]["kwargs"]["payload"]["contents"]}
    assert names == {"1_front.gif", "2_last.jpg"}


def test_create_multi_image_classification_task_rejects_unsupported_ext(
    client, capture_request, tmp_path
):
    (tmp_path / "sample.bmp").write_bytes(b"bmp")
    calls = capture_request(client, "post_request", return_value="task-id")

    with pytest.raises(FastLabelInvalidException):
        client.create_multi_image_classification_task(
            project="my-project", name="sample", folder_path=str(tmp_path)
        )

    assert calls == []
