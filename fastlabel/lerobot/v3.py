import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, TypedDict

import cv2

from fastlabel.exceptions import FastLabelInvalidException
from fastlabel.lerobot.common import Camera

logger = logging.getLogger(__name__)


class VideoInfo(TypedDict):
    """Per-episode location of one camera's segment within a consolidated
    video file."""

    chunk: str
    file_stem: str
    from_timestamp: float
    to_timestamp: float


class EpisodeInfo(TypedDict):
    """Per-episode location within the v3 layout (from meta/episodes).

    Data and video files are consolidated independently in v3, so each camera
    carries its own chunk/file location under ``videos`` (keyed by the video
    feature key, e.g. ``observation.images.top``).
    """

    data_chunk: str
    data_file_stem: str
    length: int
    videos: dict[str, VideoInfo]


# episode_index -> EpisodeInfo
EpisodeMap = dict[int, EpisodeInfo]
# One frame's parquet columns converted to native Python (JSON-serialisable).
Frame = dict[str, Any]


def _chunk_name(chunk_index: int) -> str:
    return f"chunk-{chunk_index:03d}"


def _file_stem(file_index: int) -> str:
    return f"file-{file_index:03d}"


def _build_episode_map(lerobot_data_path: Path) -> EpisodeMap:
    """Build a mapping of episode_index -> EpisodeInfo.

    Reads meta/episodes/chunk-XXX/file-YYY.parquet, which holds each episode's
    location: ``data/chunk_index``, ``data/file_index``, ``length`` and, per
    camera, ``videos/{key}/chunk_index``, ``videos/{key}/file_index``,
    ``videos/{key}/from_timestamp``, ``videos/{key}/to_timestamp``.

    Video locations cannot be derived from the data file layout: data and
    video files are consolidated independently (different size limits), so
    their chunk/file indices generally differ.
    """
    import pandas as pd

    episodes_dir = lerobot_data_path / "meta" / "episodes"
    parquet_files = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FastLabelInvalidException(
            f"Episode metadata not found: {episodes_dir}/chunk-*/file-*.parquet",
            422,
        )

    episode_map: EpisodeMap = {}
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        video_keys = [
            column.split("/")[1]
            for column in df.columns
            if column.startswith("videos/") and column.endswith("/chunk_index")
        ]
        for _, row in df.iterrows():
            videos: dict[str, VideoInfo] = {}
            for key in video_keys:
                if pd.isna(row[f"videos/{key}/chunk_index"]):
                    continue
                videos[key] = {
                    "chunk": _chunk_name(int(row[f"videos/{key}/chunk_index"])),
                    "file_stem": _file_stem(int(row[f"videos/{key}/file_index"])),
                    "from_timestamp": float(row[f"videos/{key}/from_timestamp"]),
                    "to_timestamp": float(row[f"videos/{key}/to_timestamp"]),
                }
            episode_map[int(row["episode_index"])] = {
                "data_chunk": _chunk_name(int(row["data/chunk_index"])),
                "data_file_stem": _file_stem(int(row["data/file_index"])),
                "length": int(row["length"]),
                "videos": videos,
            }

    return episode_map


def get_episode_indices(lerobot_data_path: Path) -> list[int]:
    """Get all episode indices from a v3 dataset."""
    episode_map = _build_episode_map(lerobot_data_path)
    return sorted(episode_map.keys())


def get_camera_dirs(lerobot_data_path: Path) -> list[Camera]:
    """Get camera directories and their content names (v3 video layout).

    v3: videos/{observation.images.X}/chunk-XXX/file-YYY.mp4
    Returns [(camera_dir, content_name), ...].
    e.g. observation.images.top -> content_name = "images_top"
    """
    videos_dir = lerobot_data_path / "videos"
    if not videos_dir.exists():
        return []

    results: list[Camera] = []
    for obs_dir in sorted(videos_dir.iterdir()):
        if not obs_dir.is_dir():
            continue
        parts = obs_dir.name.split(".")
        if parts[0] != "observation":
            raise FastLabelInvalidException(
                f"Unexpected camera dir name: {obs_dir.name}", 422
            )

        content_name = "_".join(parts[1:])
        results.append(
            Camera(path=obs_dir, key=obs_dir.name, content_name=content_name)
        )
    return results


def _row_to_native(row: Any) -> Frame:
    """Convert a pandas row to a dict of JSON-serialisable native Python values.

    numpy arrays become lists and numpy scalars become Python scalars so the
    result is safe to hand to converter hooks and to json.dumps.
    """
    return {
        key: (value.tolist() if hasattr(value, "tolist") else value)
        for key, value in row.items()
    }


def get_episode_raw_frames(
    lerobot_data_path: Path, episode_index: int, chunk: str, file_stem: str
) -> list[Frame]:
    """Return every frame of a single episode as native-Python dicts.

    Each dict contains all parquet columns for the row (observation.state,
    action, frame_index, timestamp, episode_index, ...). This is the raw data
    passed to the converter hooks.
    """
    import pandas as pd

    parquet_path = lerobot_data_path / "data" / chunk / f"{file_stem}.parquet"
    # Predicate pushdown so a consolidated file holding many episodes only reads
    # the relevant row groups instead of the whole file per episode.
    df = pd.read_parquet(parquet_path, filters=[("episode_index", "==", episode_index)])
    ep_df = df[df["episode_index"] == episode_index]
    return [_row_to_native(row) for _, row in ep_df.iterrows()]


def _extract_video_segment(
    video_path: Path, start_frame: int, num_frames: int, output_path: Path
) -> None:
    """Extract a segment of frames from a video file using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FastLabelInvalidException(f"Could not open video file: {video_path}", 422)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    try:
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
    finally:
        writer.release()
        cap.release()


def _assemble_episode_zip(
    ep_info: EpisodeInfo,
    episode_name: str,
    cameras: list[Camera],
    telemetry_frames: list[dict[str, Any]],
    output_dir: Path,
    fps: float | None,
) -> str:
    """Stage selected video segments + telemetry JSON, then archive as a ZIP.

    ``telemetry_frames`` is the list of frame dicts written to the episode JSON.
    ``fps`` (from meta/info.json) converts each camera's ``from_timestamp``
    into a frame offset within its consolidated video file; it may be None only
    when no selected camera has a video segment (otherwise raises before any
    staging). A camera whose video file is missing is skipped with a warning
    (the ZIP still ships the telemetry JSON).
    The staging directory is removed automatically; the ZIP is written under
    ``output_dir`` (owned by the caller) and its path returned.
    """
    if fps is None and any(camera.key in ep_info["videos"] for camera in cameras):
        raise FastLabelInvalidException(
            "'fps' not found in meta/info.json "
            "(required to locate episode video segments).",
            422,
        )

    length = ep_info["length"]

    with tempfile.TemporaryDirectory() as staging:
        content_dir = Path(staging)

        # Extract video segments
        # v3: videos/{key}/chunk-XXX/file-YYY.mp4, located per camera via
        # meta/episodes (video files are consolidated independently of data).
        for camera in cameras:
            video_info = ep_info["videos"].get(camera.key)
            if video_info is None:
                continue
            video_path = (
                camera.path / video_info["chunk"] / f"{video_info['file_stem']}.mp4"
            )
            if not video_path.exists():
                logger.warning(
                    "Video file not found, skipping camera %s: %s",
                    camera.key,
                    video_path,
                )
                continue
            output_path = content_dir / f"{camera.content_name}.mp4"
            start_frame = round(video_info["from_timestamp"] * fps)
            _extract_video_segment(video_path, start_frame, length, output_path)

        json_path = content_dir / f"{episode_name}.json"
        json_path.write_text(json.dumps(telemetry_frames, ensure_ascii=False))

        # Create ZIP (files at root, ZIP name = episode name)
        return shutil.make_archive(
            base_name=str(output_dir / episode_name),
            format="zip",
            root_dir=str(content_dir),
        )


def resolve_episode(
    episode_index: int,
    episode_map: EpisodeMap,
) -> EpisodeInfo:
    """Look up an episode's info in the episode map (raises if absent)."""
    if episode_index not in episode_map:
        raise FastLabelInvalidException(
            f"Episode index {episode_index} not found in dataset.",
            422,
        )
    return episode_map[episode_index]
