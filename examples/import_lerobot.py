"""
Import a LeRobot dataset into a FastLabel robotics project.

Requires: pip install fastlabel[robotics]

Supports LeRobot v3 dataset format only.
  v3: data/chunk-*/file-*.parquet, videos/.../chunk-*/file-*.mp4
"""

from fastlabel import Client
from fastlabel.lerobot import LeRobotConverter

client = Client()

# Import all episodes (default: full telemetry, all cameras)
results = client.import_lerobot(
    project="your-project-slug",
    lerobot_data_path="/path/to/lerobot/dataset",
)

# Import specific episodes by index
results = client.import_lerobot(
    project="your-project-slug",
    lerobot_data_path="/path/to/lerobot/dataset",
    episode_indices=[0, 1, 2],
)


# --- Customize with a converter --------------------------------------------
# A converter controls which values go into the telemetry JSON, which cameras
# are uploaded, task naming and task keyword args. The SDK constructs it with
# the parsed meta/info.json, so selection can be done by feature name.
#
# The example dataset below has a 12014-dim observation.state; this keeps only
# the joint values (names ending in ".pos") and the 4 main cameras.
class JointsOnlyConverter(LeRobotConverter):
    CAMERA_KEYS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def select_observation_state_names(self, names):
        return [n for n in names if n.endswith(".pos")]

    def select_action_names(self, names):
        return [n for n in names if n.endswith(".pos")]

    def build_telemetry_frame(self, frame):
        telemetry = super().build_telemetry_frame(frame)
        telemetry["gripper"] = self.build_action(frame)[-1]  # add an extra item
        return telemetry

    def build_task_kwargs(self, *, episode_index, episode_name, frames):
        return {
            "tags": ["lerobot", self.meta.get("robot_type", "unknown")],
            "metadatas": [{"key": "num_frames", "value": str(len(frames))}],
        }


# Pass the class itself (the SDK constructs it with meta):
results = client.import_lerobot(
    project="your-project-slug",
    lerobot_data_path="/path/to/lerobot/dataset",
    converter=JointsOnlyConverter,
)

# For constructor arguments, pass a factory:
#   from functools import partial
#   converter=partial(MyConverter, threshold=3)


# Alternatively, declare exact names statically (no method override needed):
class StaticJointsConverter(LeRobotConverter):
    OBSERVATION_STATE_NAMES = (
        "left_joint_0.pos",
        "left_joint_1.pos",
        # ... list the joint names you want ...
        "right_joint_5.pos",
        "right_left_carriage_joint.pos",
    )
    CAMERA_KEYS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")


# Skip short episodes (select_episodes receives {episode_index: frame_num} and
# is only used when episode_indices is not passed):
class LongEpisodesOnly(LeRobotConverter):
    def select_episodes(self, episode_lengths):
        return [i for i, frame_num in episode_lengths.items() if frame_num >= 100]
