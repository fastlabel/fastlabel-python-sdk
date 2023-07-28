from pprint import pprint

import fastlabel

client = fastlabel.Client()

dataset_version = client.find_dataset_version(id="YOUR_DATASET_VERSION_ID")
pprint(dataset_version)
