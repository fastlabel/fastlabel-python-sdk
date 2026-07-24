"""Tests for LeRobotConverter hooks and load_info.

These are pure-logic tests (no pandas/opencv needed): the converter operates on
plain frame dicts and camera tuples, independent of the v3 data-access layer.
"""

import json
from pathlib import Path

import pytest

from fastlabel.exceptions import FastLabelInvalidException
from fastlabel.lerobot import LeRobotConverter, load_info
from fastlabel.lerobot.common import Camera

META = {
    "robot_type": "bi_widowxai",
    "features": {
        "observation.state": {"names": ["j0.pos", "sensor_fx", "j1.pos"]},
        "action": {"names": ["a.pos", "b"]},
    },
}

FRAME = {
    "observation.state": [10.0, 20.0, 30.0],
    "action": [7.0, 8.0],
    "frame_index": 3,
    "timestamp": 0.3,
    "episode_index": 0,
}


class TestDefaults:
    def test_passthrough_keeps_all_values(self):
        c = LeRobotConverter(META)
        assert c._state_index == [0, 1, 2]
        assert c._action_index == [0, 1]
        assert c.build_telemetry_frame(FRAME) == {
            "observation.state": [10.0, 20.0, 30.0],
            "action": [7.0, 8.0],
            "frame_index": 3,
            "timestamp": 0.3,
        }

    def test_no_meta_raises(self):
        with pytest.raises(FastLabelInvalidException):
            LeRobotConverter()

    def test_select_cameras_keeps_all_by_default(self):
        cams = [
            Camera(
                Path("videos/observation.images.cam_high"),
                "observation.images.cam_high",
                "images_cam_high",
            )
        ]
        assert LeRobotConverter(META).select_cameras(cams) == cams

    def test_select_episodes_returns_all_sorted_by_default(self):
        assert LeRobotConverter(META).select_episodes({2: 5, 0: 3, 1: 4}) == [0, 1, 2]

    def test_build_episode_name_default_format(self):
        assert LeRobotConverter(META).build_episode_name(7) == "episode_000007"

    def test_build_task_kwargs_default_empty(self):
        assert (
            LeRobotConverter(META).build_task_kwargs(
                episode_index=0, episode_name="episode_000000", frames=[]
            )
            == {}
        )


class TestStaticNameSelection:
    def test_preserves_declared_order(self):
        class Sel(LeRobotConverter):
            OBSERVATION_STATE_NAMES = ("j1.pos", "j0.pos")
            ACTION_NAMES = ("a.pos",)

        c = Sel(META)
        assert c._state_index == [2, 0]
        assert c._action_index == [0]
        assert c.build_telemetry_frame(FRAME) == {
            "observation.state": [30.0, 10.0],
            "action": [7.0],
            "frame_index": 3,
            "timestamp": 0.3,
        }

    def test_missing_name_raises(self):
        class Sel(LeRobotConverter):
            OBSERVATION_STATE_NAMES = ("does_not_exist",)

        with pytest.raises(FastLabelInvalidException):
            Sel(META)

    def test_missing_names_metadata_raises(self):
        class Sel(LeRobotConverter):
            ACTION_NAMES = ("a.pos",)

        with pytest.raises(FastLabelInvalidException):
            Sel({})  # no features/names in meta


class TestDynamicSelection:
    def test_suffix_name_override(self):
        class Pos(LeRobotConverter):
            def select_observation_state_names(self, names):
                return [n for n in names if n.endswith(".pos")]

        c = Pos(META)
        assert c._state_index == [0, 2]
        assert c.build_observation_state(FRAME) == [10.0, 30.0]


class TestExtraTelemetry:
    def test_super_extend(self):
        class WithGripper(LeRobotConverter):
            def build_telemetry_frame(self, frame):
                telemetry = super().build_telemetry_frame(frame)
                telemetry["gripper"] = self.build_action(frame)[-1]
                return telemetry

        assert WithGripper(META).build_telemetry_frame(FRAME)["gripper"] == 8.0


class TestCameraSelection:
    def test_keeps_only_listed_keys(self):
        class Cam(LeRobotConverter):
            CAMERA_KEYS = ("cam_high", "cam_left_wrist")

        def cam(name):
            return Camera(
                Path(f"videos/observation.images.{name}"),
                f"observation.images.{name}",
                f"images_{name}",
            )

        cams = [cam("cam_high"), cam("gel_left_near"), cam("cam_left_wrist")]
        kept = Cam(META).select_cameras(cams)
        assert [c.content_name for c in kept] == [
            "images_cam_high",
            "images_cam_left_wrist",
        ]


class TestEpisodeHooks:
    def test_select_episodes_by_length(self):
        class LongOnly(LeRobotConverter):
            def select_episodes(self, episode_lengths):
                return [i for i, n in episode_lengths.items() if n >= 3]

        episode_lengths = {0: 1, 1: 3, 2: 5}
        assert LongOnly(META).select_episodes(episode_lengths) == [1, 2]

    def test_build_episode_name_override_uses_meta(self):
        class Named(LeRobotConverter):
            def build_episode_name(self, episode_index):
                return f"{self.meta['robot_type']}_{episode_index:03d}"

        assert Named(META).build_episode_name(2) == "bi_widowxai_002"


class TestTaskKwargs:
    def test_keyword_only(self):
        class K(LeRobotConverter):
            def build_task_kwargs(self, *, episode_index, episode_name, frames):
                return {"tags": [self.meta["robot_type"], episode_name]}

        assert K(META).build_task_kwargs(
            episode_index=1, episode_name="episode_000001", frames=[]
        ) == {"tags": ["bi_widowxai", "episode_000001"]}

    def test_positional_call_rejected(self):
        with pytest.raises(TypeError):
            LeRobotConverter(META).build_task_kwargs(0, "episode_000000", [])


class TestLoadInfo:
    def test_missing_returns_empty(self, tmp_path):
        assert load_info(tmp_path) == {}

    def test_parses_info_json(self, tmp_path):
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "info.json").write_text(json.dumps(META))
        assert load_info(tmp_path) == META
