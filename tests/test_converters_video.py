import os

import cv2
import pytest

from fastlabel import converters
from fastlabel.exceptions import FastLabelInvalidException


class TestVideoCapture:
    def test_yields_open_capture_and_releases(self, synthetic_video):
        video_path = synthetic_video(name="sample.mp4", num_frames=5)

        with converters.VideoCapture(str(video_path)) as cap:
            assert cap.isOpened()
            ret, frame = cap.read()
            assert ret is True
            assert frame is not None

    def test_releases_capture_on_exit(self, synthetic_video):
        video_path = synthetic_video(name="sample.mp4")

        with converters.VideoCapture(str(video_path)) as cap:
            captured = cap

        # After release, reading should fail or return falsy ret
        ret, _ = captured.read()
        assert ret is False

    def test_releases_capture_on_exception(self, synthetic_video):
        video_path = synthetic_video(name="sample.mp4")

        captured = None
        with pytest.raises(RuntimeError):
            with converters.VideoCapture(str(video_path)) as cap:
                captured = cap
                raise RuntimeError("boom")

        ret, _ = captured.read()
        assert ret is False


class TestExportImageFilesForVideoFile:
    def test_writes_one_jpg_per_frame(self, synthetic_video, tmp_path):
        num_frames = 7
        video_path = synthetic_video(name="sample.mp4", num_frames=num_frames)
        output_dir = tmp_path / "frames"

        names = converters._export_image_files_for_video_file(
            file_path=str(video_path),
            output_dir_path=str(output_dir),
            basename="sample",
        )

        assert len(names) == num_frames
        for name in names:
            assert name.endswith(".jpg")
            assert name.startswith("sample_")
            assert (output_dir / name).is_file()

    def test_zero_padding_matches_total_frame_digits(self, synthetic_video, tmp_path):
        num_frames = 12
        video_path = synthetic_video(name="sample.mp4", num_frames=num_frames)
        output_dir = tmp_path / "frames"

        names = converters._export_image_files_for_video_file(
            file_path=str(video_path),
            output_dir_path=str(output_dir),
            basename="vid",
        )

        # 12 frames -> 2 digit zero padding ("00".."11")
        assert names[0] == "vid_00.jpg"
        assert names[-1] == f"vid_{num_frames - 1:02d}.jpg"

    def test_written_frames_are_readable_images(self, synthetic_video, tmp_path):
        video_path = synthetic_video(
            name="sample.mp4", num_frames=3, width=64, height=48
        )
        output_dir = tmp_path / "frames"

        names = converters._export_image_files_for_video_file(
            file_path=str(video_path),
            output_dir_path=str(output_dir),
            basename="frame",
        )

        for name in names:
            img = cv2.imread(str(output_dir / name))
            assert img is not None
            assert img.shape == (48, 64, 3)

    def test_unopenable_file_raises(self, tmp_path):
        bogus = tmp_path / "not_a_video.mp4"
        bogus.write_bytes(b"not a real video")

        with pytest.raises(FastLabelInvalidException):
            converters._export_image_files_for_video_file(
                file_path=str(bogus),
                output_dir_path=str(tmp_path / "frames"),
                basename="x",
            )

    def test_creates_output_directory_if_missing(self, synthetic_video, tmp_path):
        video_path = synthetic_video(name="sample.mp4", num_frames=2)
        output_dir = tmp_path / "does" / "not" / "exist"

        assert not output_dir.exists()
        converters._export_image_files_for_video_file(
            file_path=str(video_path),
            output_dir_path=str(output_dir),
            basename="frame",
        )
        assert output_dir.is_dir()
        assert len(os.listdir(output_dir)) == 2
