from pathlib import Path

from fastlabel.exceptions import FastLabelInvalidException


def check_dependencies():
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


def format_episode_name(episode_index: int) -> str:
    return f"episode_{episode_index:06d}"


def get_camera_dirs(lerobot_data_path: Path) -> list:
    """Get camera directories and their content names.
    Returns [(camera_dir, content_name), ...].
    e.g. observation.images.top -> content_name = "images_top"
    """
    videos_dir = lerobot_data_path / "videos"
    if not videos_dir.exists():
        return []

    results = []
    for obs_dir in sorted(videos_dir.iterdir()):
        if not obs_dir.is_dir():
            continue
        parts = obs_dir.name.split(".")
        if parts[0] != "observation":
            raise FastLabelInvalidException(
                f"Unexpected camera dir name: {obs_dir.name}"
            )

        content_name = "_".join(parts[1:])
        results.append((obs_dir, content_name))
    return results
