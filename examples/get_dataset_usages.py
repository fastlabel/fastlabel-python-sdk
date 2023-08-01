from pprint import pprint

import fastlabel

client = fastlabel.Client()

dataset_usages = client.get_dataset_usages()
pprint(dataset_usages)
