from pprint import pprint

import fastlabel

client = fastlabel.Client()

comments = client.get_task_comments(project="YOUR_PROJECT_SLUG", task_id="YOUR_TASK_ID")
pprint(comments)
