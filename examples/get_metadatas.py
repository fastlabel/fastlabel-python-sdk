from pprint import pprint

import fastlabel

client = fastlabel.Client()

metadatas = client.get_metadatas(project="robotics-task-classification")
pprint(metadatas)
