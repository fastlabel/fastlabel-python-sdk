import json
from pathlib import Path
from typing import Any, NamedTuple

from fastlabel.exceptions import FastLabelInvalidException


class Camera(NamedTuple):
    """A dataset camera. Shared contract between data access and converter.

    path is the video directory (v3: videos/observation.images.X).
    key is the observation feature key (e.g. observation.images.cam_high),
        matching ``meta["features"][key]`` so converters can look up resolution
        etc.
    content_name is the mp4 filename stem inside the ZIP (e.g. images_cam_high).
    """

    path: Path
    key: str
    content_name: str


def load_info(lerobot_data_path: Path) -> dict[str, Any]:
    """Load meta/info.json for a LeRobot dataset.

    Returns the parsed dict, or {} when the file is absent (older datasets),
    so name-based selection is opt-in and default behavior is unaffected.
    """
    info_path = lerobot_data_path / "meta" / "info.json"
    if not info_path.exists():
        return {}
    return json.loads(info_path.read_text())


def check_dependencies() -> None:
    try:
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError:
        raise FastLabelInvalidException(
            "pandas and pyarrow are required for LeRobot support. "
            "Install them with: pip install fastlabel[robotics]",
            422,
        )


def detect_version(lerobot_data_path: Path) -> str:
    """Detect LeRobot dataset version (v2 or v3).

    Both versions use data/chunk-XXX/ directories.
    v2: data/chunk-XXX/episode_YYYYYY.parquet
    v3: data/chunk-XXX/file-YYY.parquet
    """
    data_dir = lerobot_data_path / "data"
    if not data_dir.exists():
        raise FastLabelInvalidException(f"Data directory not found: {data_dir}", 422)

    for chunk_dir in data_dir.iterdir():
        if not chunk_dir.is_dir() or not chunk_dir.name.startswith("chunk-"):
            continue
        for f in chunk_dir.iterdir():
            if f.suffix != ".parquet":
                continue
            if f.stem.startswith("episode_"):
                return "v2"
            if f.stem.startswith("file-"):
                return "v3"

    raise FastLabelInvalidException(
        "Could not detect LeRobot dataset version. "
        "Expected data/chunk-XXX/episode_*.parquet (v2) "
        "or data/chunk-XXX/file-*.parquet (v3).",
        422,
    )
