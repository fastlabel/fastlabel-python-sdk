from pprint import pprint

import fastlabel

client = fastlabel.Client()

auto_annotation_jobs = client.get_auto_annotation_jobs(project="YOUR_PROJECT_SLUG")
pprint(auto_annotation_jobs)
