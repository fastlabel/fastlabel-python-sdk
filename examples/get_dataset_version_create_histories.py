from pprint import pprint

import fastlabel

client = fastlabel.Client()

histories = client.get_dataset_version_create_histories(dataset_id="YOUR_DATASET_ID")
pprint(histories)
