from pprint import pprint

import fastlabel

# Initialize client
client = fastlabel.Client()

project_id = "YOUR_PROJECT_ID"
key = "YOUR_IMAGE_KEY"  # Should be an unique in your project
url = "YOUR_IMAGE_URL"
labels = [
    {
        "type": "bbox",
        "value": "bbox",
        "points": [
            {"x": 100, "y": 100},  # top-left
            {"x": 200, "y": 200},  # bottom-right
        ],
    },
    {
        "type": "line",
        "value": "line",
        "points": [{"x": 200, "y": 200}, {"x": 250, "y": 250}],
    },
    {"type": "keyPoint", "value": "keyPoint", "points": {"x": 10, "y": 10}},
    {
        "type": "polygon",
        "value": "polygon",
        "points": [
            {"x": 300, "y": 300},
            {"x": 320, "y": 320},
            {"x": 340, "y": 220},
            {"x": 310, "y": 200},
        ],
    },
    {
        "type": "polyline",
        "value": "polyline",
        "points": [
            {"x": 100, "y": 300},
            {"x": 120, "y": 320},
            {"x": 140, "y": 220},
            {"x": 110, "y": 200},
        ],
    },
    {
        "type": "segmentation",
        "value": "segmentation",
        "points": [
            [
                {"x": 400, "y": 400},
                {"x": 420, "y": 420},
                {"x": 440, "y": 420},
                {"x": 410, "y": 400},
            ]
        ],
    },
]

task = client.create_image_task(project_id=project_id, key=key, url=url, labels=labels)
pprint(task)
