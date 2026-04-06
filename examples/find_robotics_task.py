from pprint import pprint

import fastlabel

client = fastlabel.Client()

task = client.find_robotics_task(task_id="YOUR_TASK_ID")
pprint(task)
