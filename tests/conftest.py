from pathlib import Path

import cv2
import numpy as np
import pytest


def _write_synthetic_video(
    path: Path,
    num_frames: int = 10,
    width: int = 64,
    height: int = 48,
    fps: int = 10,
    fourcc_code: str = "mp4v",
) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        pytest.skip(f"cv2.VideoWriter could not open {path} with codec {fourcc_code}")
    try:
        for i in range(num_frames):
            frame = np.full((height, width, 3), i * 20 % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("Synthetic video could not be created (codec unavailable).")
    return path


@pytest.fixture
def synthetic_video(tmp_path):
    def _factory(
        name: str = "video.mp4",
        num_frames: int = 10,
        width: int = 64,
        height: int = 48,
        fps: int = 10,
        fourcc_code: str = "mp4v",
    ) -> Path:
        return _write_synthetic_video(
            tmp_path / name,
            num_frames=num_frames,
            width=width,
            height=height,
            fps=fps,
            fourcc_code=fourcc_code,
        )

    return _factory
