import sys
from pathlib import Path
from uuid import uuid4

import pytest

from fastlabel import Client


@pytest.fixture
def client() -> Client:
    return Client()


@pytest.fixture
def testing_image_dataset(client: Client) -> dict:
    name = f"test-{uuid4()}"
    dataset: dict = client.create_dataset(name=name, slug=name, type="image")
    yield dataset
    client.delete_dataset(dataset_id=dataset["id"])


@pytest.fixture
def testing_video_dataset(client: Client) -> dict:
    name = f"test-{uuid4()}"
    dataset: dict = client.create_dataset(name=name, slug=name, type="video")
    yield dataset
    client.delete_dataset(dataset_id=dataset["id"])


@pytest.fixture
def testing_audio_dataset(client: Client) -> dict:
    name = f"test-{uuid4()}"
    dataset: dict = client.create_dataset(name=name, slug=name, type="audio")
    yield dataset
    client.delete_dataset(dataset_id=dataset["id"])


class TestImageDataset:
    def test_find_dataset(self, client: Client, testing_image_dataset: dict):
        dataset = client.find_dataset(dataset_id=testing_image_dataset["id"])
        assert dataset == testing_image_dataset

    def test_get_dataset(self, client: Client, testing_image_dataset: dict):
        datasets = client.get_datasets(keyword=testing_image_dataset["slug"])
        assert datasets is not None
        assert len(datasets) == 1
        assert datasets[0] == testing_image_dataset

    def test_update_dataset(self, client: Client, testing_image_dataset: dict):
        dataset = client.update_dataset(
            dataset_id=testing_image_dataset["id"], name="update name"
        )
        assert dataset["name"] == "update name"

    def test_create_dataset_object(self, client: Client, testing_image_dataset: dict):
        target_file = Path(sys.path[0]) / "files/test_image.jpg"
        dataset_object = client.create_image_dataset_object(
            dataset_id=testing_image_dataset["id"],
            name="test_image.jpg",
            file_path=str(target_file),
        )
        assert dataset_object is not None
        assert dataset_object["name"] == "test_image.jpg"
        assert dataset_object["size"] == 6717
        assert dataset_object["height"] == 225
        assert dataset_object["width"] == 225
        assert dataset_object["groupId"] is None


class TestVideoDataset:
    def test_find_dataset(self, client: Client, testing_video_dataset: dict):
        dataset = client.find_dataset(dataset_id=testing_video_dataset["id"])
        assert dataset == testing_video_dataset

    def test_get_dataset(self, client: Client, testing_video_dataset: dict):
        datasets = client.get_datasets(keyword=testing_video_dataset["slug"])
        assert datasets is not None
        assert len(datasets) == 1
        assert datasets[0] == testing_video_dataset

    def test_update_dataset(self, client: Client, testing_video_dataset: dict):
        dataset = client.update_dataset(
            dataset_id=testing_video_dataset["id"], name="update name"
        )
        assert dataset["name"] == "update name"

    def test_create_dataset_object(self, client: Client, testing_video_dataset: dict):
        target_file = Path(sys.path[0]) / "files/test_video.mp4"
        dataset_object = client.create_video_dataset_object(
            dataset_id=testing_video_dataset["id"],
            name="test_video.mp4",
            file_path=str(target_file),
        )
        assert dataset_object is not None
        assert dataset_object["name"] == "test_video.mp4"
        assert dataset_object["size"] == 534032
        assert dataset_object["height"] == 240
        assert dataset_object["width"] == 320
        assert dataset_object["groupId"] is None


class TestAudioDataset:
    def test_find_dataset(self, client: Client, testing_audio_dataset: dict):
        dataset = client.find_dataset(dataset_id=testing_audio_dataset["id"])
        assert dataset == testing_audio_dataset

    def test_get_dataset(self, client: Client, testing_audio_dataset: dict):
        datasets = client.get_datasets(keyword=testing_audio_dataset["slug"])
        assert datasets is not None
        assert len(datasets) == 1
        assert datasets[0] == testing_audio_dataset

    def test_update_dataset(self, client: Client, testing_audio_dataset: dict):
        dataset = client.update_dataset(
            dataset_id=testing_audio_dataset["id"], name="update name"
        )
        assert dataset["name"] == "update name"

    def test_create_dataset_object(self, client: Client, testing_audio_dataset: dict):
        target_file = Path(sys.path[0]) / "files/test_audio.mp3"
        dataset_object = client.create_audio_dataset_object(
            dataset_id=testing_audio_dataset["id"],
            name="test_audio.mp3",
            file_path=str(target_file),
        )
        assert dataset_object is not None
        assert dataset_object["name"] == "test_audio.mp3"
        assert dataset_object["size"] == 16182
        assert dataset_object["height"] == 0
        assert dataset_object["width"] == 0
        assert dataset_object["groupId"] is None
