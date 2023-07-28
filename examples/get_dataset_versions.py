from pprint import pprint

import fastlabel

client = fastlabel.Client()

dataset_versions = client.get_dataset_versions(dataset_id="YOUR_DATASET_ID")
pprint(dataset_versions)
