import fastlabel

client = fastlabel.Client()

client.delete_dataset_object(
    dataset_id="YOUR_DATASET_ID", object_name="YOUR_OBJECT_NAME"
)
