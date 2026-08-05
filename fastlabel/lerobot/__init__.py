from pathlib import Path
from typing import Any

from fastlabel.exceptions import FastLabelInvalidException
from fastlabel.lerobot import v3
from fastlabel.lerobot.common import (
    Camera,
    check_dependencies,
    detect_version,
    load_info,
)
from fastlabel.lerobot.converter import LeRobotConverter
from fastlabel.lerobot.v3 import EpisodeMap, Frame

__all__ = [
    "Camera",
    "LeRobotConverter",
    "build_episode_map",
    "get_episode_indices",
    "get_episode_raw_frames",
    "create_episode_zip",
    "load_info",
]


def _ensure_v3(lerobot_data_path: Path) -> None:
    check_dependencies()
    version = detect_version(lerobot_data_path)
    if version == "v2":
        raise FastLabelInvalidException(
            "LeRobot dataset v2 is not supported. Please convert to v3.",
            422,
        )


def get_episode_indices(lerobot_data_path: Path) -> list[int]:
    """Get all episode indices from a LeRobot v3 dataset."""
    _ensure_v3(lerobot_data_path)
    return v3.get_episode_indices(lerobot_data_path)


def build_episode_map(lerobot_data_path: Path) -> EpisodeMap:
    """Build episode map from dataset. Returns a dict keyed by episode index."""
    _ensure_v3(lerobot_data_path)
    return v3._build_episode_map(lerobot_data_path)


def get_episode_raw_frames(
    lerobot_data_path: Path,
    episode_index: int,
    episode_map: EpisodeMap,
) -> list[Frame]:
    """Return every frame of an episode as native-Python dicts (all columns).

    Supports LeRobot dataset v3 only. The v3 check is not repeated here because
    ``episode_map`` can only be obtained from ``build_episode_map`` (which runs
    it once).
    """
    ep_info = v3.resolve_episode(episode_index, episode_map)
    return v3.get_episode_raw_frames(
        lerobot_data_path,
        episode_index,
        ep_info["data_chunk"],
        ep_info["data_file_stem"],
    )


def create_episode_zip(
    lerobot_data_path: Path,
    episode_index: int,
    episode_name: str,
    converter: LeRobotConverter,
    output_dir: Path,
    episode_map: EpisodeMap,
    raw_frames: list[Frame],
) -> str:
    """Create a ZIP file for a single episode in the format expected by FastLabel.

    Supports LeRobot dataset v3 only.

    ZIP structure (files at root, ZIP name = episode name):
        {content_name}.mp4  (one per selected camera)
        {episode_name}.json (telemetry frame data)

    episode_name is the identifier for the task / JSON / ZIP; the caller passes
    it (from ``converter.build_episode_name``) so the task and artifact names
    stay consistent.
    converter is a LeRobotConverter instance whose hooks control telemetry
    (build_telemetry_frame) and video selection (select_cameras). Pass
    ``LeRobotConverter()`` for default behaviour (all telemetry keys, all
    cameras). raw_frames is the episode's raw frames (from
    get_episode_raw_frames); it is reused here to avoid re-reading the parquet.

    output_dir is the directory the ZIP is written to; pass a directory you
    manage (e.g. a ``tempfile.TemporaryDirectory``) so it is cleaned up
    automatically.

    Returns the path to the created ZIP file (written under output_dir).
    The v3 check is not repeated here (``episode_map`` implies it already ran).
    """
    telemetry_frames: list[dict[str, Any]] = [
        converter.build_telemetry_frame(frame) for frame in raw_frames
    ]
    cameras = converter.select_cameras(v3.get_camera_dirs(lerobot_data_path))
    ep_info = v3.resolve_episode(episode_index, episode_map)
    # fps converts each camera's from_timestamp (meta/episodes) into a frame
    # offset within its consolidated video file.
    fps = load_info(lerobot_data_path).get("fps")
    return v3._assemble_episode_zip(
        ep_info, episode_name, cameras, telemetry_frames, output_dir, fps
    )
