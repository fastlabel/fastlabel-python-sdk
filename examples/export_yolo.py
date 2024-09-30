import fastlabel

client = fastlabel.Client()

project_slug = "YOUR_PROJECT_SLUG"
tasks = client.get_image_tasks(project=project_slug)

client.export_yolo(project=project_slug, tasks=tasks, output_dir="./export_yolo/")
