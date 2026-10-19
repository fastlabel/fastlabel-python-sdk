"""Tests for add_task_tags / remove_task_tags.

The HTTP layer (client.api.post_request) is stubbed so no real request is made.
"""

import pytest

from fastlabel import const
from fastlabel.exceptions import FastLabelInvalidException


@pytest.mark.parametrize(
    "method_name, endpoint",
    [("add_task_tags", "tasks/tags/add"), ("remove_task_tags", "tasks/tags/remove")],
)
def test_sends_task_ids_and_tags(client, capture_request, method_name, endpoint):
    calls = capture_request(
        client, "post_request", return_value={"updatedTaskCount": 2}
    )

    task_count = getattr(client, method_name)(
        project="my-project", task_ids=["t1", "t2"], tags=["a", "b"]
    )

    assert task_count == 2
    assert calls == [
        {
            "endpoint": endpoint,
            "args": (),
            "kwargs": {
                "payload": {
                    "project": "my-project",
                    "taskIds": ["t1", "t2"],
                    "tags": ["a", "b"],
                }
            },
        }
    ]


def test_splits_task_ids_over_the_limit_and_sums_updated_task_counts(
    client, monkeypatch
):
    monkeypatch.setattr(const, "TASK_TAGS_UPDATE_TASK_IDS_LIMIT", 2)
    # t2 already has the tag, so the first batch updates only t1.
    updated_task_counts = iter([1, 1])
    sent_task_ids = []

    def fake_post_request(endpoint, payload):
        sent_task_ids.append(payload["taskIds"])
        return {"updatedTaskCount": next(updated_task_counts)}

    monkeypatch.setattr(client.api, "post_request", fake_post_request)

    task_count = client.add_task_tags(
        project="my-project", task_ids=["t1", "t2", "t3", "t1"], tags=["a"]
    )

    assert sent_task_ids == [["t1", "t2"], ["t3"]]
    assert task_count == 2


@pytest.mark.parametrize(
    "task_ids, tags",
    [([], ["a"]), (["t1"], [])],
)
def test_rejects_empty_task_ids_or_tags(client, capture_request, task_ids, tags):
    calls = capture_request(
        client, "post_request", return_value={"updatedTaskCount": 0}
    )

    with pytest.raises(FastLabelInvalidException):
        client.remove_task_tags(project="my-project", task_ids=task_ids, tags=tags)

    assert calls == []
