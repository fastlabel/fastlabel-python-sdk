import json
import os

from visual_inspection_ai import VisualInspectionAi


class VisualInspectionAiConverter:
    def convert_from_bbox(self, file_path: str, export_file_path: str) -> dict:
        bbox_json = self.open_bbox(file_path)
        visual_inspection_ai = VisualInspectionAi.from_bbox(bbox_json)

        os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
        with open(export_file_path, "w") as file:
            file.write(visual_inspection_ai.as_jsonl())

    def open_bbox(self, file_path: str) -> any:
        return json.load(open(file_path, "r"))


converter = VisualInspectionAiConverter()

file_path = "./import/visual_inspection_ai/annotations.json"  # 入力元のファイルパスを指定
export_file_path = "./export/visual_inspection_ai/test.jsonl"  # 出力先のファイルパスを指定
result = converter.convert_from_bbox(file_path, export_file_path)
