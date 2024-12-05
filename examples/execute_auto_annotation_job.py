from pprint import pprint

import fastlabel

client = fastlabel.Client()

auto_annotation_job = client.execute_auto_annotation_job(
    project="YOUR_PROJECT_SLUG",
    model_name="MODEL_NAME",
)
pprint(auto_annotation_job)
