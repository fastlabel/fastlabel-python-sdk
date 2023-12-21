from pprint import pprint

import fastlabel

# Initialize client
client = fastlabel.Client()

project = "YOUR_PROJECT_SLUG"
name = "YOUR_DATA_NAME"
file_path = "YOUR_DATA_FILE_PATH"  # e.g.) ./cat.jpg
annotations = [
    {
        "type": "bbox",
        "value": "cat",
        "attributes": [{"key": "kind", "value": "Scottish field"}],
        "points": [
            100,  # top-left x
            100,  # top-left y
            200,  # bottom-right x
            200,  # bottom-right y
        ],
    }
]

task_id = client.create_image_task(
    project=project, name=name, file_path=file_path, annotations=annotations
)
pprint(task_id)
