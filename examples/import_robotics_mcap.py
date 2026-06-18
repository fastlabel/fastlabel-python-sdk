"""
Import an mcap zip file into an existing FastLabel robotics task.

This starts an import pipeline, uploads the zip via a signed URL,
and triggers the batch import.
"""

from pprint import pprint

import fastlabel

client = fastlabel.Client()

result = client.import_robotics_mcap(
    project="YOUR_PROJECT_SLUG",
    task_id="YOUR_TASK_ID",
    file_path="ZIP_FILE_PATH",  # Supported extension is .zip
)
pprint(result)
