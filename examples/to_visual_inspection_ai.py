import json
import os

from visual_inspection_ai import VisualInspectionAi


class VisualInspectionAiConverter:
    def convert_from_json(
        self, file_path: str, export_file_path: str, project_type: str
    ) -> dict:
        json = self.open_file(file_path)
        if project_type == "bbox":
            visual_inspection_ai = VisualInspectionAi.from_bbox(json)
        else:
            visual_inspection_ai = VisualInspectionAi.from_segmentation(json)

        os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
        with open(export_file_path, "w") as file:
            file.write(visual_inspection_ai.as_jsonl())

    def open_file(self, file_path: str) -> any:
        return json.load(open(file_path, "r"))


converter = VisualInspectionAiConverter()

export_file_path = (
    "./export/visual_inspection_ai/example.jsonl"  # 出力先のファイルパスを指定
)

# bboxの場合
# file_path = (
#     "./import/visual_inspection_ai/bbox/annotations.json"  # 入力元のファイルパスを指定
# )
# project_type = "bbox"

# segmentationの場合
file_path = "./import/visual_inspection_ai/segmentation/annotations.json"  # 入力元のファイルパスを指定
project_type = "segmentation"

result = converter.convert_from_json(file_path, export_file_path, project_type)
