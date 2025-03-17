import fastlabel

client = fastlabel.Client()

project_slug = "YOUR_PROJECT_SLUG"
tasks = client.get_image_tasks(project=project_slug)
annotations = client.get_annotations(project=project_slug)

client.export_coco(
    project=project_slug,
    tasks=tasks,
    annotations=annotations,
    output_dir="./export_coco/",
)
