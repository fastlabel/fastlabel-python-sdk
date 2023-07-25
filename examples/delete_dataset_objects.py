import fastlabel

client = fastlabel.Client()

client.delete_dataset_objects(
    dataset_id="YOUR_DATASET_ID",
    dataset_object_ids=[
        "YOUR_DATASET_OBJECT_ID_1",
        "YOUR_DATASET_OBJECT_ID_2",
    ],
)
