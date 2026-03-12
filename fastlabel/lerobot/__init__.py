from fastlabel.exceptions import FastLabelInvalidException
from fastlabel.lerobot import v3
from fastlabel.lerobot.common import (
    check_dependencies,
    detect_version,
    format_episode_name,
    get_camera_dirs,
)

__all__ = [
    "get_episode_indices",
    "create_episode_zip",
    "format_episode_name",
    "get_camera_dirs",
]


def get_episode_indices(lerobot_data_path):
    """Get all episode indices from a LeRobot v3 dataset."""
    check_dependencies()
    version = detect_version(lerobot_data_path)
    if version == "v2":
        raise FastLabelInvalidException(
            "LeRobot dataset v2 is not supported. Please convert to v3.",
            422,
        )
    return v3.get_episode_indices(lerobot_data_path)


def create_episode_zip(lerobot_data_path, episode_index):
    """Create a ZIP file for a single episode in the format expected by FastLabel.

    Supports LeRobot dataset v3 only.

    ZIP structure:
        {episode_name}/
            {content_name}.mp4  (one per camera)
            {episode_name}.json (frame data)

    Returns the path to the created ZIP file.
    The caller is responsible for cleaning up the returned ZIP file.
    """
    check_dependencies()
    version = detect_version(lerobot_data_path)
    if version == "v2":
        raise FastLabelInvalidException(
            "LeRobot dataset v2 is not supported. Please convert to v3.",
            422,
        )
    return v3.create_episode_zip(lerobot_data_path, episode_index)
