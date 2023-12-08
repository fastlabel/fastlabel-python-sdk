from pprint import pprint

import fastlabel

client = fastlabel.Client()

datasets = client.get_datasets()
pprint(datasets)
