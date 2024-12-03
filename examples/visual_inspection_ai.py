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
        goog_vi_ml_use,
        label=None,
        annotationId=None,
        annotationSpec=None,
        annotationSpecId=None,
    ):
        self.vertex = vertex
        self.project_type = project_type
        self.image_gcs_uri = image_gcs_uri
        self.goog_vi_ml_use = goog_vi_ml_use
        self.label = label
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
    def from_segmentation(
        cls,
        width,
        height,
        points,
        image_gcs_uri,
        goog_vi_ml_use,
        label,
        annotationId,
        annotationSpec,
        annotationSpecId,
    ):
        # FIXME サンプルのjsonには矩形しかなかったので、とりあえずで変換
        figure_points = points[0][
            0
        ]  # FIXME fastlabelでは飛地がある場合に配列に複数の要素がある場合があるが、VisualInspectionAiで飛地をどのように扱うか不明なため、一旦最初の図形だけパース。
        vertices = [
            {"x": figure_points[i] / width, "y": figure_points[i + 1] / height}
            for i in range(0, len(figure_points), 2)
        ]
        return VisualInspectionAiFigure(
            vertex=vertices,
            project_type="segmentation",
            image_gcs_uri=image_gcs_uri,
            goog_vi_ml_use=goog_vi_ml_use,
            label=label,
            annotationId=annotationId,
            annotationSpec=annotationSpec,
            annotationSpecId=annotationSpecId,
        )

    # VisualInspectionAiのjsonlを作成する。
    def make_jsonl(self) -> dict:
        vi_annotations = (
            self.__make_bbox_vi_annotations_jsonl()
            if self.project_type == "bbox"
            else self.__make_segmentation_vi_annotations_jsonl()
        )
        dataItemResourceLabels = {
            "goog_vi_ml_use": self.goog_vi_ml_use,
        }

        if self.project_type == "segmentation":
            dataItemResourceLabels["label"] = self.label
        return {
            "image_gcs_uri": self.image_gcs_uri,
            "vi_annotations": vi_annotations,
            "dataItemResourceLabels": dataItemResourceLabels,
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

    @classmethod
    def parse_bbox_json(
        cls, bbox_json, goog_mi_use_test_probability=0.3
    ) -> list[VisualInspectionAiFigure]:
        result = []
        width = bbox_json["width"]
        height = bbox_json["height"]
        image_gcs_uri = bbox_json["name"]

        for annotation in bbox_json["annotations"]:
            points = annotation["points"]
            goog_vi_ml_use = (
                "test" if random.random() < goog_mi_use_test_probability else "training"
            )
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
    def parse_segmentation_json(
        cls, json, goog_mi_use_test_probability=0.3
    ) -> list[VisualInspectionAiFigure]:
        result = []
        width = json["width"]
        height = json["height"]
        # TODO 配列と文字で型の互換がないが問題ないか確認
        label = ", ".join(json["tags"])
        image_gcs_uri = json["name"]
        for annotation in json["annotations"]:
            annotationId = annotation["id"]
            annotationSpec = annotation["title"]
            annotationSpecId = annotation["value"]
            points = annotation["points"]
            goog_vi_ml_use = (
                "test" if random.random() < goog_mi_use_test_probability else "training"
            )
            result.append(
                VisualInspectionAiFigure.from_segmentation(
                    width,
                    height,
                    points,
                    image_gcs_uri,
                    goog_vi_ml_use,
                    label,
                    annotationId,
                    annotationSpec,
                    annotationSpecId,
                )
            )
        return result

    def __make_jsonl(self) -> list:
        return list(map(lambda it: it.make_jsonl(), self.figures))

    def as_jsonl(self):
        return json.dumps(self.__make_jsonl(), ensure_ascii=False, indent=4) + "\n"
