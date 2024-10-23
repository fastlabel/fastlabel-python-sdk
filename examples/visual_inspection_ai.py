import json

# いろいろなフォーマットを受け取って、VisualInspectionAiのjsonlで出力するのが責務


class VisualInspectionAiFigure:
    def __init__(self, vertex, project_type):
        self.vertex = vertex
        self.project_type = project_type

    @classmethod
    def fromBbox(cls, width, height, points):
        vertices = [
            (points[0] / width, points[1] / height),  # 左上
            (points[2] / width, points[1] / height),  # 右上
            (points[0] / width, points[3] / height),  # 左下
            (points[2] / width, points[3] / height),  # 右下
        ]
        return VisualInspectionAiFigure(vertices, "bbox")

    # VisualInspectionAiのjsonlを作成する。
    def make_jsonl(self) -> dict:
        return {
            "image_gcs_uri": "gs://ffod-98sm/20240613/NG/test/BQ1334B__11-7_18-19_0004.png",
            "vi_annotations": self.__make_vi_annotations_jsonl(),
            "dataItemResourceLabels": {"goog_vi_ml_use": "test"},
        }

    def __make_vi_annotations_jsonl(self) -> dict:
        return {
            "viBoundingPoly": {"vertex": self.vertex},
            "annotationSpec": "defect",
            "annotationSet": "Polygons Regions"
            if self.project_type == "bbox"
            else "TODO segmentationの場合の表記調査",
        }


class VisualInspectionAi:
    def __init__(self, figures):
        self.figures = figures

    @classmethod
    def from_bbox(cls, bbox_json):
        figures = []
        for bbox in bbox_json:
            figures.extend(VisualInspectionAi.parseBboxJson(bbox))
        return cls(figures)

    @classmethod
    def parseBboxJson(cls, bbox_json) -> list[VisualInspectionAiFigure]:
        result = []
        width = bbox_json["width"]
        height = bbox_json["height"]
        for annotation in bbox_json["annotations"]:
            points = annotation["points"]
            result.append(VisualInspectionAiFigure.fromBbox(width, height, points))
        return result

    @classmethod
    def from_segmentation(cls, segmentation_json, project_type):
        # TODO segmentationからvertexへの変換処理
        return cls([])

    def __make_jsonl(self) -> list:
        print(self.figures)
        print(self.figures[0].make_jsonl())
        print(list(map(lambda it: it.make_jsonl(), self.figures)))
        return list(map(lambda it: it.make_jsonl(), self.figures))

    def as_jsonl(self):
        return json.dumps(self.__make_jsonl(), ensure_ascii=False, indent=4) + "\n"
