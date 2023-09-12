import sys
from pathlib import Path
from uuid import uuid4

import pytest

from fastlabel import Client


@pytest.fixture
def client() -> Client:
    return Client()


@pytest.fixture
def testing_dataset(client: Client) -> dict:
    # Arrange
    name = f"test-{uuid4()}"
    dataset = client.create_dataset(name=name)
    yield dataset
    # Cleanup
    client.delete_dataset(dataset_id=dataset["id"])


class TestImageDataset:
    def test_find_dataset(self, client: Client, testing_dataset: dict):
        # Act
        dataset = client.find_dataset(dataset_id=testing_dataset["id"])
        # Assert
        assert dataset == testing_dataset

    def test_get_dataset(self, client: Client, testing_dataset: dict):
        # Act
        datasets = client.get_datasets(keyword=testing_dataset["name"])
        # Assert
        assert datasets is not None
        assert len(datasets) == 1
        assert datasets[0] == testing_dataset

    def test_update_dataset(self, client: Client, testing_dataset: dict):
        # Act
        dataset = client.update_dataset(
            dataset_id=testing_dataset["id"],
            name="update-name",
        )
        # Assert
        assert dataset["name"] == "update-name"

    def test_create_dataset_object(self, client: Client, testing_dataset: dict):
        # Arrange
        target_file = Path(sys.path[0]) / "files/test_image.jpg"
        # Act
        dataset_object = client.create_dataset_object(
            dataset=testing_dataset["name"],
            name="test_image.jpg",
            file_path=str(target_file),
        )
        # Assert
        assert dataset_object is not None
        assert dataset_object["name"] == "test_image.jpg"
        assert dataset_object["size"] == 6717
        assert dataset_object["height"] == 225
        assert dataset_object["width"] == 225

    def test_find_dataset_object(self, client: Client, testing_dataset: dict):
        # Arrange
        target_file = Path(sys.path[0]) / "files/test_image.jpg"
        dataset_object = client.create_dataset_object(
            dataset=testing_dataset["name"],
            name="test_image.jpg",
            file_path=str(target_file),
        )
        # Act
        result = client.find_dataset_object(dataset_object_id=dataset_object["id"])
        # Assert
        assert result["name"] == dataset_object["name"]

    def test_get_dataset_object(self, client: Client, testing_dataset: dict):
        # Arrange
        target_file = Path(sys.path[0]) / "files/test_image.jpg"
        client.create_dataset_object(
            dataset=testing_dataset["name"],
            name="test_image1.jpg",
            file_path=str(target_file),
            tags=["image1"],
        )
        client.create_dataset_object(
            dataset=testing_dataset["name"],
            name="test_image2.jpg",
            file_path=str(target_file),
            tags=["image1"],
        )
        # Act
        results = client.get_dataset_objects(
            dataset=testing_dataset["name"],
            tags=["image1"],
        )
        # Assert
        assert results is not None
        assert len(results) == 2


class TestVideoDataset:
    def test_find_dataset(self, client: Client, testing_dataset: dict):
        # Act
        dataset = client.find_dataset(dataset_id=testing_dataset["id"])
        # Assert
        assert dataset == testing_dataset

    def test_get_dataset(self, client: Client, testing_dataset: dict):
        # Act
        datasets = client.get_datasets(keyword=testing_dataset["name"])
        # Assert
        assert datasets is not None
        assert len(datasets) == 1
        assert datasets[0] == testing_dataset

    def test_update_dataset(self, client: Client, testing_dataset: dict):
        # Act
        dataset = client.update_dataset(
            dataset_id=testing_dataset["id"], name="update-name"
        )
        # Assert
        assert dataset["name"] == "update-name"

    def test_create_dataset_object(self, client: Client, testing_dataset: dict):
        # Arrange
        target_file = Path(sys.path[0]) / "files/test_video.mp4"
        # Act
        dataset_object = client.create_dataset_object(
            dataset=testing_dataset["name"],
            name="test_video.mp4",
            file_path=str(target_file),
        )
        # Assert
        assert dataset_object is not None
        assert dataset_object["name"] == "test_video.mp4"
        assert dataset_object["size"] == 534032
        assert dataset_object["height"] == 240
        assert dataset_object["width"] == 320

    def test_find_dataset_object(self, client: Client, testing_dataset: dict):
        # Arrange
        target_file = Path(sys.path[0]) / "files/test_video.mp4"
        dataset_object = client.create_dataset_object(
            dataset=testing_dataset["name"],
            name="test_video.mp4",
            file_path=str(target_file),
        )
        # Act
        result = client.find_dataset_object(dataset_object_id=dataset_object["id"])
        # Assert
        assert result["name"] == dataset_object["name"]

    def test_get_dataset_object(self, client: Client, testing_dataset: dict):
        # Arrange
        target_file = Path(sys.path[0]) / "files/test_video.mp4"
        client.create_dataset_object(
            dataset=testing_dataset["name"],
            name="test_video1.mp4",
            file_path=str(target_file),
            tags=["video1"],
        )
        client.create_dataset_object(
            dataset=testing_dataset["name"],
            name="test_video2.mp4",
            file_path=str(target_file),
            tags=["video1"],
        )
        # Act
        results = client.get_dataset_objects(
            dataset=testing_dataset["name"],
            tags=["video1"],
        )
        # Assert
        assert results is not None
        assert len(results) == 2


class TestAudioDataset:
    def test_find_dataset(self, client: Client, testing_dataset: dict):
        # Act
        dataset = client.find_dataset(dataset_id=testing_dataset["id"])
        # Assert
        assert dataset == testing_dataset

    def test_get_dataset(self, client: Client, testing_dataset: dict):
        # Act
        datasets = client.get_datasets(keyword=testing_dataset["name"])
        # Assert
        assert datasets is not None
        assert len(datasets) == 1
        assert datasets[0] == testing_dataset

    def test_update_dataset(self, client: Client, testing_dataset: dict):
        # Act
        dataset = client.update_dataset(
            dataset_id=testing_dataset["id"],
            name="update-name",
        )
        # Assert
        assert dataset["name"] == "update-name"

    def test_create_dataset_object(self, client: Client, testing_dataset: dict):
        # Arrange
        target_file = Path(sys.path[0]) / "files/test_audio.mp3"
        # Act
        dataset_object = client.create_dataset_object(
            dataset=testing_dataset["name"],
            name="test_audio.mp3",
            file_path=str(target_file),
        )
        # Assert
        assert dataset_object is not None
        assert dataset_object["name"] == "test_audio.mp3"
        assert dataset_object["size"] == 16182
        assert dataset_object["height"] == 0
        assert dataset_object["width"] == 0

    def test_find_dataset_object(self, client: Client, testing_dataset: dict):
        # Arrange
        target_file = Path(sys.path[0]) / "files/test_audio.mp3"
        dataset_object = client.create_dataset_object(
            dataset=testing_dataset["name"],
            name="test_audio.mp3",
            file_path=str(target_file),
        )
        # Act
        result = client.find_dataset_object(dataset_object_id=dataset_object["id"])
        # Assert
        assert result["name"] == dataset_object["name"]

    def test_get_dataset_object(self, client: Client, testing_dataset: dict):
        # Arrange
        target_file = Path(sys.path[0]) / "files/test_audio.mp3"
        client.create_dataset_object(
            dataset=testing_dataset["name"],
            name="test_audio1.mp3",
            file_path=str(target_file),
            tags=["audio1"],
        )
        client.create_dataset_object(
            dataset=testing_dataset["name"],
            name="test_audio2.mp3",
            file_path=str(target_file),
            tags=["audio1"],
        )
        # Act
        results = client.get_dataset_objects(
            dataset=testing_dataset["name"],
            tags=["audio1"],
        )
        # Assert
        assert results is not None
        assert len(results) == 2


class TestMixingDataset:
    def test_find_dataset(self, client: Client, testing_dataset: dict):
        dataset = client.find_dataset(dataset_id=testing_dataset["id"])
        assert dataset == testing_dataset

    def test_create_dataset_object(self, client: Client, testing_dataset: dict):
        # Arrange
        target_file = Path(sys.path[0]) / "files/test_other_file.txt"
        # Act
        dataset_object = client.create_dataset_object(
            dataset=testing_dataset["name"],
            name="test_other_file.txt",
            file_path=str(target_file),
        )
        # Assert
        assert dataset_object is not None
        assert dataset_object["name"] == "test_other_file.txt"
        assert dataset_object["size"] == 3090
        assert dataset_object["height"] == 0
        assert dataset_object["width"] == 0
