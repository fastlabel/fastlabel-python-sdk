from pprint import pprint

import fastlabel

client = fastlabel.Client()

dataset_object = client.find_dataset_object(dataset_object_id="YOUR_DATASET_OBJECT_ID")
pprint(dataset_object)
