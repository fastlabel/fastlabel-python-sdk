from pprint import pprint

import fastlabel

client = fastlabel.Client()

task_id = client.update_robotics_task(
    task_id="YOUR_TASK_ID",
    status="approved",
    priority=10,  # none: 0, low: 10, medium: 20, high: 30
    assignee="USER_SLUG",
    tags=["tag1", "tag2"],
    operator="OPERATOR_NAME",
    annotations=[
        {
            "type": "sub_task",
            "value": "grab",
            "start": 13,
            "end": 18,
        }
    ],
    metadatas=[
        {
            "key": "metadata_key",
            "value": "metadata_value",
        }
    ],
)
pprint(task_id)
