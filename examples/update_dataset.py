from pprint import pprint

import fastlabel

client = fastlabel.Client()

dataset = client.update_dataset(dataset_id="YOUR_DATASET_ID", name="object-detection")
pprint(dataset)
