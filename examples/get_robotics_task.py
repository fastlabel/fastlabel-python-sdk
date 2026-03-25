from pprint import pprint

import fastlabel

client = fastlabel.Client()

tasks = client.get_robotics_task(project="YOUR_PROJECT_SLUG")
pprint(tasks)
