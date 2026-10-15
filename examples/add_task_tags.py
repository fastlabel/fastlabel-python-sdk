from pprint import pprint

import fastlabel

client = fastlabel.Client()

updated_task_count = client.add_task_tags(
    project="YOUR_PROJECT_SLUG",
    task_ids=["YOUR_TASK_ID_1", "YOUR_TASK_ID_2"],
    tags=["tag1", "tag2"],
)
pprint(updated_task_count)
