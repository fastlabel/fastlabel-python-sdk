"""
Import a LeRobot dataset into a FastLabel robotics project.

Requires: pip install fastlabel[robotics]

Supports LeRobot v3 dataset format only.
  v3: data/chunk-*/file-*.parquet, videos/.../chunk-*/file-*.mp4
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
