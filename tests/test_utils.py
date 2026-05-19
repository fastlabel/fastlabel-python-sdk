from fastlabel import utils


class TestIsVideoSupportedCodec:
    def test_h264_returns_true(self, synthetic_video):
        video_path = synthetic_video(name="sample.mp4", fourcc_code="h264")

        assert utils.get_video_fourcc(str(video_path)) == "h264"
        assert utils.is_video_supported_codec(str(video_path)) is True

    def test_mp4v_returns_false(self, synthetic_video):
        video_path = synthetic_video(name="sample.mp4", fourcc_code="mp4v")

        assert utils.is_video_supported_codec(str(video_path)) is False
