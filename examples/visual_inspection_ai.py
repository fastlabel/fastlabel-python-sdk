import json
import random

# visualInspectionAiの一要素として表せる図形のValue Object。
# jsonl型として出力するのが責務


class VisualInspectionAiFigure:
    def __init__(
        self,
        vertex,
        project_type,
        image_gcs_uri,
        label,
        goog_vi_ml_use,
        annotationId,
        annotationSpec,
        annotationSpecId,
    ):
        self.vertex = vertex
        self.project_type = project_type
        self.image_gcs_uri = image_gcs_uri
        self.label = label
        self.goog_vi_ml_use = goog_vi_ml_use
        self.annotationId = annotationId
        self.annotationSpec = annotationSpec
        self.annotationSpecId = annotationSpecId

    @classmethod
    def from_bbox(cls, width, height, points, image_gcs_uri, goog_vi_ml_use):
        vertices = [
            (points[0] / width, points[1] / height),  # 左上
            (points[2] / width, points[1] / height),  # 右上
            (points[0] / width, points[3] / height),  # 左下
            (points[2] / width, points[3] / height),  # 右下
        ]
        return VisualInspectionAiFigure(
            vertex=vertices,
            project_type="bbox",
            image_gcs_uri=image_gcs_uri,
            goog_vi_ml_use=goog_vi_ml_use,
        )

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
        vi_annotations = (
            self.__make_bbox_vi_annotations_jsonl()
            if self.project_type == "bbox"
            else self.__make_segmentation_vi_annotations_jsonl()
        )
        return {
            "image_gcs_uri": self.image_gcs_uri,
            "vi_annotations": vi_annotations,
            "dataItemResourceLabels": {
                "label": self.label,
                "goog_vi_ml_use": self.goog_vi_ml_use,
            },
        }

    def __make_bbox_vi_annotations_jsonl(self) -> dict:
        return {
            "viBoundingPoly": {"vertex": self.vertex},
            "annotationSpec": self.annotationSpec,
            "annotationSet": "Polygons Regions",
        }

    def __make_segmentation_vi_annotations_jsonl(self) -> dict:
        return {
            "viBoundingPoly": {"vertex": self.vertex},
            "annotationId": self.annotationId,
            "annotationSpecId": self.annotationSpecId,
            "annotationSpec": self.annotationSpec,
            "annotationSet": "Polygons Regions",
            "annotationSetId": "",
            "annotationResourceLabels": {"goog_vi_annotation_set_name": ""},
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

    # image_gcs_uri　タスク名をマッピング
    # dataItemResourceLabels タグ情報をマッピング
    # goog_vi_ml_use 全量に対して7:3でtrainingとtestがわかれるようにマッピング（分割ロジックがよくわからんので割合変えれるようにきりだしとく）

    @classmethod
    def parse_bbox_json(
        cls, bbox_json, goog_mi_use_test_probability=0.3
    ) -> list[VisualInspectionAiFigure]:
        result = []
        width = bbox_json["width"]
        height = bbox_json["height"]
        image_gcs_uri = bbox_json["name"]
        goog_vi_ml_use = (
            "test" if random.random() < goog_mi_use_test_probability else "training"
        )
        for annotation in bbox_json["annotations"]:
            points = annotation["points"]
            result.append(
                VisualInspectionAiFigure.from_bbox(
                    width, height, points, image_gcs_uri, goog_vi_ml_use
                )
            )
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
        # TODO 削除
        print(self.figures)
        print(self.figures[0].make_jsonl())
        print(list(map(lambda it: it.make_jsonl(), self.figures)))
        return list(map(lambda it: it.make_jsonl(), self.figures))

    def as_jsonl(self):
        return json.dumps(self.__make_jsonl(), ensure_ascii=False, indent=4) + "\n"
