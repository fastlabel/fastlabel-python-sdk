from pprint import pprint

import fastlabel

client = fastlabel.Client()

dataset_version = client.create_dataset_version(
    dataset_id="YOUR_DATASET_ID",
    version="version",
)
pprint(dataset_version)
