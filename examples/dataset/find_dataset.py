from pprint import pprint

import fastlabel

client = fastlabel.Client()

dataset = client.find_dataset(dataset_id="YOUR_DATASET_ID")
pprint(dataset)
