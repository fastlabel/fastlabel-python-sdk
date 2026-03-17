from pprint import pprint

import fastlabel

client = fastlabel.Client()

comments = client.get_project_comments(project="YOUR_PROJECT_SLUG")
pprint(comments)
