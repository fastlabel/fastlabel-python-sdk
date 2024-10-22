import json


# いろいろなフォーマットを受け取って、VisualInspectionAiのjsonlで出力するのが責務
class VisualInspectionAi:
    def __init__(self, vertex, project_type):
        self.vertex = vertex
        self.project_type = project_type

    @classmethod
    def from_bbox(cls, bbox_json):
        # TODO bboxからvertexへの変換処理
        vertex = [{"x": 0.1, "y": 0.1}]
        return cls(vertex, "bbox")

    @classmethod
    def from_segmentation(cls, segmentation_json, project_type):
        # TODO segmentationからvertexへの変換処理
        vertex = [{"x": 0.1, "y": 0.1}]
        return cls(vertex, "segmentation")

    # VisualInspectionAiのjsonlを作成する。
    def __make_jsonl(self):
        via = []
        via_el = {
            "image_gcs_uri": "gs://ffod-98sm/20240613/NG/test/BQ1334B__11-7_18-19_0004.png",
            "vi_annotations": self.__make_vi_annotations_jsonl(),
            "dataItemResourceLabels": {"goog_vi_ml_use": "test"},
        }
        via.append(via_el)
        return via

    def __make_vi_annotations_jsonl(self):
        return {
            "viBoundingPoly": {"vertex": self.vertex},
            "annotationSpec": "defect",
            "annotationSet": "Polygons Regions"
            if self.project_type == "bbox"
            else "TODO segmentationの場合の表記調査",
        }

    def as_jsonl(self):
        return json.dumps(self.__make_jsonl(), ensure_ascii=False, indent=4) + "\n"
