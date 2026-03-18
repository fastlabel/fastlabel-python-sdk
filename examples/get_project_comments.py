from pprint import pprint

import fastlabel

client = fastlabel.Client()

comment_threads = client.get_project_comments(project="YOUR_PROJECT_SLUG")
pprint(comment_threads)
