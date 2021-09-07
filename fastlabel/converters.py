from concurrent.futures import ThreadPoolExecutor
from typing import List

import copy
import geojson
import numpy as np
import math
from fastlabel.const import AnnotationType

# COCO


def to_coco(tasks: list) -> dict:
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


# YOLO


def to_yolo(tasks: list, classes: list) -> tuple:
    if len(classes) == 0:
        coco = to_coco(tasks)
        return __coco2yolo(coco)
    else:
        return __to_yolo(tasks, classes)


def __coco2yolo(coco: dict) -> tuple:
    categories = coco["categories"]

    annos = []
    for image in coco["images"]:
        dw = 1. / image["width"]
        dh = 1. / image["height"]

        # Get objects
        objs = []
        for annotation in coco["annotations"]:
            if image["id"] != annotation["image_id"]:
                continue

            category_index = "0"
            for index, category in enumerate(categories):
                if category["id"] == annotation["category_id"]:
                    category_index = str(index)
                    break

            xmin = annotation["bbox"][0]
            ymin = annotation["bbox"][1]
            xmax = annotation["bbox"][0] + annotation["bbox"][2]
            ymax = annotation["bbox"][1] + annotation["bbox"][3]

            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin

            x = str(_truncate(x * dw, 7))
            y = str(_truncate(y * dh, 7))
            w = str(_truncate(w * dw, 7))
            h = str(_truncate(h * dh, 7))

            obj = [category_index, x, y, w, h]
            objs.append(" ".join(obj))

        # get annotation
        anno = {
            "filename": image["file_name"],
            "object": objs
        }
        annos.append(anno)

    return annos, categories


def __to_yolo(tasks: list, classes: list) -> tuple:
    annos = []
    for task in tasks:
        if task["height"] == 0 or task["width"] == 0:
            continue
        objs = []
        data = [{"annotation": annotation, "task": task, "classes": classes}
                for annotation in task["annotations"]]
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(__get_yolo_annotation, data)
            for result in results:
                if not result:
                    continue
                objs.append(" ".join(result))
            anno = {
                "filename": task["name"],
                "object": objs
            }
            annos.append(anno)

    categories = map(lambda val: {"name": val}, classes)

    return annos, categories


def __get_yolo_annotation(data: dict) -> dict:
    annotation = data["annotation"]
    points = annotation["points"]
    annotation_type = annotation["type"]
    value = annotation["value"]
    classes = list(data["classes"])
    task = data["task"]
    if annotation_type != AnnotationType.bbox.value and annotation_type != AnnotationType.polygon.value:
        return None
    if not points or len(points) == 0:
        return None
    if annotation_type == AnnotationType.bbox.value and (int(points[0]) == int(points[2]) or int(points[1]) == int(points[3])):
        return None
    if not annotation["value"] in classes:
        return None

    dw = 1. / task["width"]
    dh = 1. / task["height"]

    bbox = __to_bbox(points)
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[0] + bbox[2]
    ymax = bbox[1] + bbox[3]

    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin

    x = str(_truncate(x * dw, 7))
    y = str(_truncate(y * dh, 7))
    w = str(_truncate(w * dw, 7))
    h = str(_truncate(h * dh, 7))
    category_index = str(classes.index(value))
    return [category_index, x, y, w, h]


def _truncate(n, decimals=0) -> float:
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


# Pascal VOC


def to_pascalvoc(tasks: list) -> list:
    pascalvoc = []
    for task in tasks:
        if task["height"] == 0 or task["width"] == 0:
            continue

        pascal_objs = []
        data = [{"annotation": annotation}
                for annotation in task["annotations"]]
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(__get_pascalvoc_obj, data)

        for result in results:
            if not result:
                continue
            pascal_objs.append(result)

        voc = {
            "annotation": {
                "filename": task["name"],
                "size": {
                    "width": task["width"],
                    "height": task["height"],
                    "depth": 3,
                },
                "segmented": 0,
                "object": pascal_objs
            }
        }
        pascalvoc.append(voc)
    return pascalvoc

def __get_pascalvoc_obj(data: dict) -> dict:
    annotation = data["annotation"]
    points = annotation["points"]
    annotation_type = annotation["type"]
    if annotation_type != AnnotationType.bbox.value and annotation_type != AnnotationType.polygon.value:
        return None
    if not points or len(points) == 0:
        return None
    if annotation_type == AnnotationType.bbox.value and (int(points[0]) == int(points[2]) or int(points[1]) == int(points[3])):
        return None
    bbox = __to_bbox(points)
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]

    return {
        "name": annotation["value"],
        "pose": "Unspecified",
                "truncated": __get_pascalvoc_tag_value(annotation, "truncated"),
                "occluded": __get_pascalvoc_tag_value(annotation, "occluded"),
                "difficult": __get_pascalvoc_tag_value(annotation, "difficult"),
                "bndbox": {
                    "xmin": math.floor(x),
                    "ymin": math.floor(y),
                    "xmax": math.floor(x + w),
                    "ymax": math.floor(y + h),
        },
    }

def __get_pascalvoc_tag_value(annotation: dict, target_tag_name: str) -> int:
    attributes = annotation["attributes"]
    if not attributes:
        return 0
    related_attr = next(
        (attribute for attribute in attributes if attribute["type"] == "switch" and attribute["key"] == target_tag_name), None)
    return int(related_attr["value"]) if related_attr else 0


# labelme


def to_labelme(tasks: list) -> list:
    labelmes =[]
    for task in tasks:
        shapes = []
        for annotation in task["annotations"]:
                shape_type = __to_labelme_shape_type(annotation["type"])
                if not shape_type:
                    continue
                points = annotation["points"]
                if len(points) == 0:
                    continue

                shape_points = []
                if annotation["type"] == "segmentation":
                    for i in range(int(len(points[0][0]) / 2)):
                        shape_points.append([points[0][0][i * 2], points[0][0][(i * 2) + 1]]) 
                else:
                    for i in range(int(len(points) / 2)):
                        shape_points.append([points[i * 2], points[(i * 2) + 1]]) 

                shape = {
                        "label": annotation["value"],
                        "points": shape_points,
                        "group_id": None,
                        "shape_type": shape_type,
                        "flags": {}
                }
                shapes.append(shape)
        labelmes.append({
                "version": "4.5.9",
                "flags": {},
                "shapes": shapes,
                "imagePath": task["name"],
                "imageData": None,
                "imageHeight": task["height"],
                "imageWidth": task["width"],
        })
    return labelmes

def __to_labelme_shape_type(annotation_type: str) -> str:
    if annotation_type == "polygon" or annotation_type == "segmentation":
        return "polygon"
    if annotation_type == "bbox":
        return "rectangle"
    if annotation_type == "keypoint":
        return "point"
    if annotation_type == "line":
        return "line"
    return None


def to_pixel_coordinates(tasks: list) -> list:
    """
    Remove diagonal coordinates and return pixel outline coordinates.
    Only support bbox, polygon, and segmentation annotation types.
    """
    tasks = copy.deepcopy(tasks)
    for task in tasks:
        for annotation in task["annotations"]:
            if annotation["type"] == AnnotationType.segmentation.value:
                new_regions = []
                for region in annotation["points"]:
                    new_region = []
                    for points in region:
                        new_points = __get_pixel_coordinates(points)
                        new_region.append(new_points)
                    new_regions.append(new_region)
                annotation["points"] = new_regions
            elif annotation["type"] == AnnotationType.polygon.value:
                new_points = __get_pixel_coordinates(annotation["points"])
                annotation["points"] = new_points
            elif annotation["type"] == AnnotationType.bbox.value:
                points = annotation["points"]
                points = [int(point) for point in points]
                xmin = min([points[0], points[2]])
                ymin = min([points[1], points[3]])
                xmax = max([points[0], points[2]])
                ymax = max([points[1], points[3]])
                annotation["points"] = [
                    xmin, ymin,
                    xmax, ymin,
                    xmax, ymax,
                    xmin, ymax,
                ]
            else:
                continue

    # Remove duplicate points
    for task in tasks:
        for annotation in task["annotations"]:
            if annotation["type"] == AnnotationType.segmentation.value:
                new_regions = []
                for region in annotation["points"]:
                    new_region = []
                    for points in region:
                        new_points = __remove_duplicated_coordinates(points)
                        new_region.append(new_points)
                    new_regions.append(new_region)
                annotation["points"] = new_regions
            elif annotation["type"] == AnnotationType.polygon.value:
                new_points = __remove_duplicated_coordinates(annotation["points"])
                annotation["points"] = new_points
    return tasks

def __remove_duplicated_coordinates(points: List[int]) -> List[int]:
    """
    Remove duplicated coordinates.
    """
    if len(points) == 0:
        return []

    new_points = []
    for i in range(int(len(points) / 2)):
        if i == 0:
            new_points.append(points[i*2])
            new_points.append(points[i*2 + 1])

        if new_points[-2] == points[i*2] and new_points[-1] == points[i*2 + 1]:
            continue

        if len(new_points) <= 2:
            new_points.append(points[i*2])
            new_points.append(points[i*2 + 1])
        else:
            if new_points[-4] == new_points[-2] and new_points[-2] == points[i*2]:
                new_points.pop()
                new_points.pop()
                new_points.append(points[i*2])
                new_points.append(points[i*2 + 1])
            elif new_points[-3] == new_points[-1] and new_points[-1] == points[i*2 + 1]:
                new_points.pop()
                new_points.pop()
                new_points.append(points[i*2])
                new_points.append(points[i*2 + 1])
            else:
                new_points.append(points[i*2])
                new_points.append(points[i*2 + 1])
    return new_points

def __get_pixel_coordinates(points: List[int or float]) -> List[int]:
    """
    Remove diagonal coordinates and return pixel outline coordinates.
    """
    if len(points) == 0:
        return []

    new_points = []
    new_points.append(int(points[0]))
    new_points.append(int(points[1]))
    for i in range(int(len(points) / 2)):
        if i == 0:
            continue

        prev_x = int(points[(i-1) * 2])
        prev_y = int(points[(i-1) * 2 + 1])
        x = int(points[i * 2])
        y = int(points[i * 2 + 1])

        if prev_x == x or prev_y == y:
            # just add x, y coordinates if not diagonal
            new_points.append(x)
            new_points.append(y)
        else:
            # remove diagonal
            xdiff = x - prev_x
            ydiff = y - prev_y
            mindiff = min([abs(xdiff), abs(ydiff)])
            for i in range(mindiff):
                new_points.append(int(prev_x + int(xdiff / mindiff * i)))
                new_points.append(int(prev_y + int(ydiff / mindiff * (i + 1))))
                new_points.append(int(prev_x + int(xdiff / mindiff * (i + 1))))
                new_points.append(int(prev_y + int(ydiff / mindiff * (i + 1))))
    return new_points
