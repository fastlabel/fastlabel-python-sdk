"""
Import a LeRobot dataset into a FastLabel robotics project.

Requires: pip install fastlabel[robotics]

Supports both LeRobot v2 and v3 dataset formats.
  v2: data/chunk-*/episode_*.parquet, videos/.../chunk-*/episode_*.mp4
  v3: data/file-*.parquet, videos/.../file-*.mp4
"""

from fastlabel import Client

client = Client()

# Import all episodes
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
