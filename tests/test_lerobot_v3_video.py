import zipfile

import cv2
import pytest

from fastlabel.exceptions import FastLabelInvalidException
from fastlabel.lerobot import v3
from fastlabel.lerobot.common import Camera


class TestExtractVideoSegment:
    def test_extracts_requested_number_of_frames(self, synthetic_video, tmp_path):
        source = synthetic_video(name="src.mp4", num_frames=20, width=64, height=48)
        output = tmp_path / "segment.mp4"

        v3._extract_video_segment(
            video_path=source,
            start_frame=5,
            num_frames=8,
            output_path=output,
        )

        assert output.is_file()
        cap = cv2.VideoCapture(str(output))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()

        assert count == 8
        assert (width, height) == (64, 48)

    def test_stops_when_source_ends(self, synthetic_video, tmp_path):
        source = synthetic_video(name="src.mp4", num_frames=10)
        output = tmp_path / "segment.mp4"

        v3._extract_video_segment(
            video_path=source,
            start_frame=8,
            num_frames=50,
            output_path=output,
        )

        cap = cv2.VideoCapture(str(output))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

        assert count == 2

    def test_unopenable_file_raises(self, tmp_path):
        bogus = tmp_path / "not_a_video.mp4"
        bogus.write_bytes(b"garbage")

        with pytest.raises(FastLabelInvalidException):
            v3._extract_video_segment(
                video_path=bogus,
                start_frame=0,
                num_frames=1,
                output_path=tmp_path / "out.mp4",
            )


class TestAssembleEpisodeZip:
    """Video location comes from meta/episodes (per camera), not from the
    data file layout: the video chunk/file indices may differ from the data
    ones, and the frame offset is from_timestamp * fps."""

    CAMERA_KEY = "observation.images.top"

    def _episode_info(self, videos):
        return {
            "data_chunk": "chunk-000",
            "data_file_stem": "file-000",
            "length": 4,
            "videos": videos,
        }

    def _camera(self, tmp_path):
        return Camera(
            path=tmp_path / "videos" / self.CAMERA_KEY,
            key=self.CAMERA_KEY,
            content_name="images_top",
        )

    def test_resolves_video_location_and_timestamp_offset(
        self, synthetic_video, tmp_path
    ):
        # The episode's video lives in file-001 even though its data lives in
        # file-000, and starts 0.5s (= frame 5 at fps 10) into that file.
        (tmp_path / "videos" / self.CAMERA_KEY / "chunk-000").mkdir(parents=True)
        synthetic_video(
            name=f"videos/{self.CAMERA_KEY}/chunk-000/file-001.mp4",
            num_frames=10,
            fps=10,
        )
        ep_info = self._episode_info(
            {
                self.CAMERA_KEY: {
                    "chunk": "chunk-000",
                    "file_stem": "file-001",
                    "from_timestamp": 0.5,
                    "to_timestamp": 0.9,
                }
            }
        )
        out = tmp_path / "out"
        out.mkdir()

        zip_path = v3._assemble_episode_zip(
            ep_info,
            "episode_000001",
            [self._camera(tmp_path)],
            telemetry_frames=[],
            output_dir=out,
            fps=10.0,
        )

        with zipfile.ZipFile(zip_path) as zf:
            assert sorted(zf.namelist()) == [
                "episode_000001.json",
                "images_top.mp4",
            ]
            extract_dir = tmp_path / "extracted"
            zf.extract("images_top.mp4", extract_dir)

        cap = cv2.VideoCapture(str(extract_dir / "images_top.mp4"))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        assert count == 4

    def test_missing_video_file_warns_and_skips(self, tmp_path, caplog):
        (tmp_path / "videos" / self.CAMERA_KEY / "chunk-000").mkdir(parents=True)
        ep_info = self._episode_info(
            {
                self.CAMERA_KEY: {
                    "chunk": "chunk-000",
                    "file_stem": "file-009",
                    "from_timestamp": 0.0,
                    "to_timestamp": 0.4,
                }
            }
        )
        out = tmp_path / "out"
        out.mkdir()

        with caplog.at_level("WARNING", logger="fastlabel.lerobot.v3"):
            zip_path = v3._assemble_episode_zip(
                ep_info,
                "episode_000001",
                [self._camera(tmp_path)],
                telemetry_frames=[],
                output_dir=out,
                fps=10.0,
            )

        # The missing camera is skipped with a warning; the ZIP still ships
        # the telemetry JSON.
        assert any("file-009.mp4" in message for message in caplog.messages)
        with zipfile.ZipFile(zip_path) as zf:
            assert zf.namelist() == ["episode_000001.json"]

    def test_missing_fps_raises_when_video_needed(self, synthetic_video, tmp_path):
        (tmp_path / "videos" / self.CAMERA_KEY / "chunk-000").mkdir(parents=True)
        synthetic_video(
            name=f"videos/{self.CAMERA_KEY}/chunk-000/file-000.mp4",
            num_frames=10,
            fps=10,
        )
        ep_info = self._episode_info(
            {
                self.CAMERA_KEY: {
                    "chunk": "chunk-000",
                    "file_stem": "file-000",
                    "from_timestamp": 0.0,
                    "to_timestamp": 0.4,
                }
            }
        )
        out = tmp_path / "out"
        out.mkdir()

        with pytest.raises(FastLabelInvalidException):
            v3._assemble_episode_zip(
                ep_info,
                "episode_000001",
                [self._camera(tmp_path)],
                telemetry_frames=[],
                output_dir=out,
                fps=None,
            )
