from enum import Enum
from concurrent.futures import ThreadPoolExecutor

import geojson
import numpy as np


class AnnotationType(Enum):
    bbox = "bbox"
    polygon = "polygon"
    keypoint = "keypoint"
    classification = "classification"
    line = "line"
    segmentation = "segmentation"


def to_coco(tasks: list) -> dict:
    """
    Convert tasks to COCO format.
    """
    # Get categories
    categories = __get_categories(tasks)

    # Get images and annotations
    images = []
    annotations = []
    annotation_id = 0
    image_id = 0
    for task in tasks:
        if task["height"] == 0 or task["width"] == 0:
            continue

        image_id += 1
        image = {
            "file_name": task["name"],
            "height": task["height"],
            "width": task["width"],
            "id": image_id,
        }
        images.append(image)

        data = [{"annotation": annotation, "categories": categories,
                 "image": image} for annotation in task["annotations"]]
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(__to_annotation, data)

        for result in results:
            annotation_id += 1
            if not result:
                continue
            result["id"] = annotation_id
            annotations.append(result)

    return {
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }


def __get_categories(tasks: list) -> list:
    values = []
    for task in tasks:
        for annotation in task["annotations"]:
            if annotation["type"] != AnnotationType.bbox.value and annotation["type"] != AnnotationType.polygon.value:
                continue
            values.append(annotation["value"])
    values = list(set(values))

    categories = []
    for index, value in enumerate(values):
        category = {
            "supercategory": value,
            "id": index + 1,
            "name": value
        }
        categories.append(category)
    return categories


def __to_annotation(data: dict) -> dict:
    annotation = data["annotation"]
    categories = data["categories"]
    image = data["image"]
    points = annotation["points"]
    annotation_type = annotation["type"]
    annotation_id = 0

    if annotation_type != AnnotationType.bbox.value and annotation_type != AnnotationType.polygon.value:
        return None
    if not points or len(points) == 0:
        return None
    if annotation_type == AnnotationType.bbox.value and (int(points[0]) == int(points[2]) or int(points[1]) == int(points[3])):
        return None

    category = __get_category_by_name(categories, annotation["value"])

    return __get_annotation(
        annotation_id, points, category["id"], image, annotation_type)


def __get_category_by_name(categories: list, name: str) -> str:
    category = [
        category for category in categories if category["name"] == name][0]
    return category


def __get_annotation(id_: int, points: list, category_id: int, image: dict, annotation_type: str) -> dict:
    annotation = {}
    annotation["segmentation"] = [points]
    annotation["iscrowd"] = 0
    annotation["area"] = __calc_area(annotation_type, points)
    annotation["image_id"] = image["id"]
    annotation["bbox"] = __to_bbox(points)
    annotation["category_id"] = category_id
    annotation["id"] = id_
    return annotation


def __to_bbox(points: list) -> list:
    points_splitted = [points[idx:idx + 2]
                       for idx in range(0, len(points), 2)]
    polygon_geo = geojson.Polygon(points_splitted)
    coords = np.array(list(geojson.utils.coords(polygon_geo)))
    left_top_x = coords[:, 0].min()
    left_top_y = coords[:, 1].min()
    right_bottom_x = coords[:, 0].max()
    right_bottom_y = coords[:, 1].max()

    return [
        left_top_x,  # x
        left_top_y,  # y
        right_bottom_x - left_top_x,  # width
        right_bottom_y - left_top_y,  # height
    ]


def __calc_area(annotation_type: str, points: list) -> float:
    area = 0
    if annotation_type == AnnotationType.bbox.value:
        width = points[0] - points[2]
        height = points[1] - points[3]
        area = width * height
    elif annotation_type == AnnotationType.polygon.value:
        x = points[0::2]
        y = points[1::2]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) -
                            np.dot(y, np.roll(x, 1)))
    return area