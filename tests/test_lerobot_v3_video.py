import cv2
import pytest

from fastlabel.exceptions import FastLabelInvalidException
from fastlabel.lerobot import v3


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
