import fastlabel

client = fastlabel.Client()

client.delete_dataset_object(
    dataset_id="YOUR_DATASET_OBJECT_ID", object_name="YOUR_OBJECT_NAME"
)
