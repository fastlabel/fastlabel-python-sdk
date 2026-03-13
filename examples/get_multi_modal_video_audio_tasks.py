from pprint import pprint

import fastlabel

client = fastlabel.Client()

tasks = client.get_multi_modal_video_audio_tasks(project="YOUR_PROJECT_SLUG")
pprint(tasks)
