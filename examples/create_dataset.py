from pprint import pprint

import fastlabel

client = fastlabel.Client()

dataset = client.create_dataset(name="object-detection")

pprint(dataset)
