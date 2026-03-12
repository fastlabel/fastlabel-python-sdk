import json
import shutil
import tempfile
from pathlib import Path

import cv2

from fastlabel.exceptions import FastLabelInvalidException
from fastlabel.lerobot.common import format_episode_name, get_camera_dirs


def _build_episode_map(lerobot_data_path: Path) -> dict:
    """Build a mapping of episode_index -> {chunk, file_stem, frame_offset, length}.

    Reads all data parquet files across all chunks and computes per-episode
    frame offsets within each file (needed for video segment extraction).

    v3 layout: data/chunk-XXX/file-YYY.parquet
    """
    import pandas as pd

    data_dir = lerobot_data_path / "data"
    episode_map = {}

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


def get_episode_indices(lerobot_data_path: Path) -> list:
    """Get all episode indices from a v3 dataset."""
    episode_map = _build_episode_map(lerobot_data_path)
    return sorted(episode_map.keys())


def _convert_episode_frames(
    lerobot_data_path: Path, episode_index: int, chunk: str, file_stem: str
) -> list:
    """Extract frame dicts for a single episode from a v3 consolidated parquet."""
    import pandas as pd

    parquet_path = lerobot_data_path / "data" / chunk / f"{file_stem}.parquet"
    df = pd.read_parquet(parquet_path)
    ep_df = df[df["episode_index"] == episode_index]

    return [
        {
            "observation.state": row["observation.state"].tolist(),
            "action": row["action"].tolist(),
            "frame_index": int(row["frame_index"]),
            "timestamp": float(row["timestamp"]),
        }
        for _, row in ep_df.iterrows()
    ]


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

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    writer.release()
    cap.release()


def create_episode_zip(lerobot_data_path: Path, episode_index: int) -> str:
    """Create a ZIP for a single v3 episode.

    v3 video layout: videos/{key}/chunk-XXX/file-YYY.mp4
    """
    episode_map = _build_episode_map(lerobot_data_path)

    if episode_index not in episode_map:
        raise FastLabelInvalidException(
            f"Episode index {episode_index} not found in dataset.",  # noqa: E713
            422,
        )

    ep_info = episode_map[episode_index]
    chunk = ep_info["chunk"]
    file_stem = ep_info["file_stem"]
    frame_offset = ep_info["frame_offset"]
    length = ep_info["length"]
    episode_name = format_episode_name(episode_index)

    tmp_dir = tempfile.mkdtemp()
    content_dir = Path(tmp_dir) / "content"
    content_dir.mkdir()

    # Extract video segments
    # v3: videos/{key}/chunk-XXX/file-YYY.mp4
    for camera_dir, content_name in get_camera_dirs(lerobot_data_path):
        video_path = camera_dir / chunk / f"{file_stem}.mp4"
        if not video_path.exists():
            continue
        output_path = content_dir / f"{content_name}.mp4"
        _extract_video_segment(video_path, frame_offset, length, output_path)

    # Convert parquet to JSON
    frames = _convert_episode_frames(lerobot_data_path, episode_index, chunk, file_stem)
    json_path = content_dir / f"{episode_name}.json"
    json_path.write_text(json.dumps(frames, ensure_ascii=False))

    # Create ZIP (files at root, ZIP name = episode name)
    zip_path = shutil.make_archive(
        base_name=str(Path(tmp_dir) / episode_name),
        format="zip",
        root_dir=str(content_dir),
    )
    shutil.rmtree(content_dir)
    return zip_path
