import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, TypedDict

import cv2

from fastlabel.exceptions import FastLabelInvalidException
from fastlabel.lerobot.common import Camera


class EpisodeInfo(TypedDict):
    """Per-episode location within the v3 layout."""

    chunk: str
    file_stem: str
    frame_offset: int
    length: int


# episode_index -> EpisodeInfo
EpisodeMap = dict[int, EpisodeInfo]
# One frame's parquet columns converted to native Python (JSON-serialisable).
Frame = dict[str, Any]


def _build_episode_map(lerobot_data_path: Path) -> EpisodeMap:
    """Build a mapping of episode_index -> EpisodeInfo.

    Reads all data parquet files across all chunks and computes per-episode
    frame offsets within each file (needed for video segment extraction).

    v3 layout: data/chunk-XXX/file-YYY.parquet
    """
    import pandas as pd

    data_dir = lerobot_data_path / "data"
    episode_map: EpisodeMap = {}

    for chunk_dir in sorted(data_dir.iterdir()):
        if not chunk_dir.is_dir() or not chunk_dir.name.startswith("chunk-"):
            continue
        chunk_name = chunk_dir.name

        for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
            file_stem = parquet_file.stem
            df = pd.read_parquet(parquet_file)

            frame_offset = 0
            for ep_idx in sorted(df["episode_index"].unique()):
                ep_df = df[df["episode_index"] == ep_idx]
                length = len(ep_df)
                episode_map[int(ep_idx)] = {
                    "chunk": chunk_name,
                    "file_stem": file_stem,
                    "frame_offset": frame_offset,
                    "length": length,
                }
                frame_offset += length

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
) -> str:
    """Stage selected video segments + telemetry JSON, then archive as a ZIP.

    ``telemetry_frames`` is the list of frame dicts written to the episode JSON.
    The staging directory is removed automatically; the ZIP is written under
    ``output_dir`` (owned by the caller) and its path returned.
    """
    chunk = ep_info["chunk"]
    file_stem = ep_info["file_stem"]
    frame_offset = ep_info["frame_offset"]
    length = ep_info["length"]

    with tempfile.TemporaryDirectory() as staging:
        content_dir = Path(staging)

        # Extract video segments
        # v3: videos/{key}/chunk-XXX/file-YYY.mp4
        for camera in cameras:
            video_path = camera.path / chunk / f"{file_stem}.mp4"
            if not video_path.exists():
                continue
            output_path = content_dir / f"{camera.content_name}.mp4"
            _extract_video_segment(video_path, frame_offset, length, output_path)

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
