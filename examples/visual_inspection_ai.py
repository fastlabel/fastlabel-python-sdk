import json


# visualInspectionAiの一要素として表せる図形の情報。
# jsonl型として出力するのが責務
class VisualInspectionAiFigure:
    def __init__(self, vertex, project_type):
        self.vertex = vertex
        self.project_type = project_type

    @classmethod
    def from_bbox(cls, width, height, points):
        vertices = [
            (points[0] / width, points[1] / height),  # 左上
            (points[2] / width, points[1] / height),  # 右上
            (points[0] / width, points[3] / height),  # 左下
            (points[2] / width, points[3] / height),  # 右下
        ]
        return VisualInspectionAiFigure(vertices, "bbox")

    @classmethod
    def from_segmentation(cls, width, height, points):
        # FIXME サンプルのjsonには矩形しかなかったので、とりあえずで変換
        figure_points = points[0][
            0
        ]  # FIXME fastlabelでは飛地がある場合に配列に複数の要素がある場合があるが、VisualInspectionAiで飛地をどのように扱うか不明なため、一旦最初の図形だけパース。
        vertices = [
            {"x": figure_points[i] / width, "y": figure_points[i + 1] / height}
            for i in range(0, len(figure_points), 2)
        ]
        return VisualInspectionAiFigure(vertices, "segmentation")

    # VisualInspectionAiのjsonlを作成する。
    def make_jsonl(self) -> dict:
        return {
            "image_gcs_uri": "gs://ffod-98sm/20240613/NG/test/BQ1334B__11-7_18-19_0004.png",  # TODO ここに入れ込めそうな値はjsonにはないので、調査
            "vi_annotations": self.__make_vi_annotations_jsonl(),
            "dataItemResourceLabels": {
                "goog_vi_ml_use": "test"
            },  # TODO この値が何を指しているのか調査
        }

    def __make_vi_annotations_jsonl(self) -> dict:
        return {
            "viBoundingPoly": {"vertex": self.vertex},
            "annotationSpec": "defect",  # TODO ここにクラスが入る場合は入れる。
            "annotationSet": (
                "Polygons Regions"
                if self.project_type == "bbox"
                else "TODO segmentationの場合の表記調査"
            ),
        }


# いろいろなフォーマットを受け取って、VisualInspectionAiのjsonlで出力するのが責務


class VisualInspectionAi:
    def __init__(self, figures):
        self.figures = figures

    @classmethod
    def from_bbox(cls, bbox_json):
        figures = []
        for bbox in bbox_json:
            figures.extend(VisualInspectionAi.parse_bbox_json(bbox))
        return cls(figures)

    @classmethod
    def parse_bbox_json(cls, bbox_json) -> list[VisualInspectionAiFigure]:
        result = []
        width = bbox_json["width"]
        height = bbox_json["height"]
        for annotation in bbox_json["annotations"]:
            points = annotation["points"]
            result.append(VisualInspectionAiFigure.from_bbox(width, height, points))
        return result

    @classmethod
    def from_segmentation(cls, segmentation_json):
        figures = []
        for bbox in segmentation_json:
            figures.extend(VisualInspectionAi.parse_segmentation_json(bbox))
        return cls(figures)

    @classmethod
    def parse_segmentation_json(cls, json) -> list[VisualInspectionAiFigure]:
        result = []
        width = json["width"]
        height = json["height"]
        for annotation in json["annotations"]:
            points = annotation["points"]
            result.append(
                VisualInspectionAiFigure.from_segmentation(width, height, points)
            )
        return result

    def __make_jsonl(self) -> list:
        print(self.figures)
        print(self.figures[0].make_jsonl())
        print(list(map(lambda it: it.make_jsonl(), self.figures)))
        return list(map(lambda it: it.make_jsonl(), self.figures))

    def as_jsonl(self):
        return json.dumps(self.__make_jsonl(), ensure_ascii=False, indent=4) + "\n"
