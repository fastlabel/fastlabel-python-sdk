import copy
import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from operator import itemgetter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Dict, Optional

import cv2
import geojson
import numpy as np
import requests

from fastlabel.const import AnnotationType, AttributeValue
from fastlabel.exceptions import FastLabelInvalidException
from fastlabel.utils import is_video_project_type

# COCO


def to_coco(
    project_type: str, tasks: list, output_dir: str, annotations: list = []
) -> dict:
    # Get categories
    categories = __get_coco_categories(tasks, annotations)

    # Get images and annotations
    images = []
    annotations = []
    annotation_id = 0
    image_index = 0
    for task in tasks:
        if task["height"] == 0 or task["width"] == 0:
            continue

        if is_video_project_type(project_type):
            image_file_names = _export_image_files_for_video_task(
                task, str((Path(output_dir) / "images").resolve())
            )
            task_images = _generate_coco_images(
                image_file_names=image_file_names,
                height=task["height"],
                width=task["width"],
                offset=image_index,
            )
            image_index = len(task_images) + image_index

            def get_annotation_points(anno, index):
                return _get_annotation_points_for_video_annotation(anno, index)

        else:
            image_index += 1
            task_images = [
                {
                    "file_name": task["name"],
                    "height": task["height"],
                    "width": task["width"],
                    "id": image_index,
                }
            ]

            def get_annotation_points(anno, _):
                return _get_annotation_points_for_image_annotation(anno)

        for index, task_image in enumerate(task_images, 1):
            images.append(task_image)
            params = [
                {
                    "annotation_value": annotation["value"],
                    "annotation_type": annotation["type"],
                    "annotation_points": get_annotation_points(annotation, index),
                    "annotation_rotation": annotation.get("rotation", 0),
                    "annotation_keypoints": annotation.get("keypoints"),
                    "annotation_attributes": _get_coco_annotation_attributes(
                        annotation
                    ),
                    "categories": categories,
                    "image_id": task_image["id"],
                }
                for annotation in task["annotations"]
            ]

            with ThreadPoolExecutor(max_workers=8) as executor:
                image_annotations = executor.map(__to_coco_annotation, params)

            filtered_image_annotations = list(filter(None, image_annotations))
            if len(filtered_image_annotations) <= 0:
                continue

            for image_annotation in sorted(
                filtered_image_annotations,
                key=itemgetter("image_id", "category_id", "area"),
            ):
                annotation_id += 1
                if not image_annotation:
                    continue
                image_annotation["id"] = annotation_id
                annotations.append(image_annotation)

    return {
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }


def _generate_coco_images(
    image_file_names: str, height: int, width: int, offset: int = 0
):
    return [
        {
            "file_name": file_name,
            "height": height,
            "width": width,
            "id": (index + 1) + offset,
        }
        for index, file_name in enumerate(image_file_names)
    ]


def __get_coco_skeleton(keypoints: list) -> list:
    keypoint_id_skeleton_index_map = {}
    for index, keypoint in enumerate(keypoints, 1):
        keypoint_id_skeleton_index_map[keypoint["id"]] = index

    skeleton = []
    filtered_skeleton_indexes = []
    for keypoint in keypoints:
        id = keypoint["id"]
        skeleton_index = keypoint_id_skeleton_index_map[id]
        edges = keypoint["edges"]
        for edge in edges:
            edge_skeleton_index = keypoint_id_skeleton_index_map[edge]
            if edge_skeleton_index not in filtered_skeleton_indexes:
                skeleton.append([skeleton_index, edge_skeleton_index])
            filtered_skeleton_indexes.append(skeleton_index)
    return skeleton


def __get_coco_categories(tasks: list, annotations: list) -> list:
    categories = []
    values = []
    for task in tasks:
        for task_annotation in task["annotations"]:
            if task_annotation["type"] not in [
                AnnotationType.bbox.value,
                AnnotationType.polygon.value,
                AnnotationType.segmentation.value,
                AnnotationType.pose_estimation.value,
            ]:
                continue
            values.append(task_annotation["value"])
    values = sorted(list(set(values)))

    # Create categories from task annotations (not support pose esitimation)
    if not annotations:
        for index, value in enumerate(values, 1):
            category = {
                "skeleton": [],
                "keypoints": [],
                "keypoint_colors": [],
                # BUG: All are set to the same color.
                "color": task_annotation["color"],
                "supercategory": value,
                "id": index,
                "name": value,
            }
            categories.append(category)
        return categories

    # Create categories from passed annotations (support pose esitimation)
    index = 1
    for annotation in annotations:
        if not annotation["value"] in values:
            continue
        coco_skeleton = []
        coco_keypoints = []
        coco_keypoint_colors = []
        if annotation["type"] == AnnotationType.pose_estimation.value:
            keypoints = annotation["keypoints"]
            for keypoint in keypoints:
                coco_keypoints.append(keypoint["key"])
                coco_keypoint_colors.append(keypoint["color"])
            coco_skeleton = __get_coco_skeleton(keypoints)
        category = {
            "skeleton": coco_skeleton,
            "keypoints": coco_keypoints,
            "keypoint_colors": coco_keypoint_colors,
            "color": annotation["color"],
            "supercategory": annotation["value"],
            "id": index,
            "name": annotation["value"],
        }
        index += 1
        categories.append(category)
    return categories


def __to_coco_annotation(data: dict) -> dict:
    categories = data["categories"]
    image_id = data["image_id"]
    points = data["annotation_points"]
    keypoints = data["annotation_keypoints"]
    rotation = data["annotation_rotation"]
    annotation_type = data["annotation_type"]
    annotation_value = data["annotation_value"]
    annotation_id = 0
    annotation_attributes = data["annotation_attributes"]

    if annotation_type not in [
        AnnotationType.bbox.value,
        AnnotationType.polygon.value,
        AnnotationType.segmentation.value,
        AnnotationType.pose_estimation.value,
    ]:
        return None
    if annotation_type != AnnotationType.pose_estimation.value and (
        not points or (len(points) == 0)
    ):
        return None
    if annotation_type == AnnotationType.bbox.value and (
        int(points[0]) == int(points[2]) or int(points[1]) == int(points[3])
    ):
        return None

    category = __get_coco_category_by_name(categories, annotation_value)
    if category is None:
        return None

    return __get_coco_annotation(
        annotation_id,
        points,
        keypoints,
        category,
        image_id,
        annotation_type,
        annotation_attributes,
        rotation
    )


def __get_coco_category_by_name(categories: list, name: str) -> Optional[dict]:
    matched_categories = [
        category for category in categories if category["name"] == name
    ]
    if len(matched_categories) >= 1:
        return matched_categories[0]
    return None


def __get_coco_annotation_keypoints(keypoints: list, category_keypoints: list) -> list:
    coco_annotation_keypoints = []
    keypoint_values = {keypoint["key"]: keypoint["value"] for keypoint in keypoints if keypoint["value"]}
    for category_key in category_keypoints:
        value = keypoint_values.get(category_key, [0, 0, 0])
        # Adjust fastlabel data definition to coco format
        visibility = 2 if value[2] == 1 else 1
        coco_annotation_keypoints.extend([value[0], value[1], visibility])
    return coco_annotation_keypoints


def __get_coco_annotation(
    id_: int,
    points: list,
    keypoints: list,
    category: dict,
    image_id: str,
    annotation_type: str,
    annotation_attributes: Dict[str, AttributeValue],
    rotation: int
) -> dict:
    annotation = {}
    annotation["num_keypoints"] = len(keypoints) if keypoints else 0
    annotation["keypoints"] = (
        __get_coco_annotation_keypoints(keypoints, category["keypoints"]) if keypoints else []
    )
    annotation["segmentation"] = __to_coco_segmentation(annotation_type, points)
    annotation["iscrowd"] = 0
    annotation["area"] = __to_area(annotation_type, points)
    annotation["image_id"] = image_id
    annotation["bbox"] = (
        __get_coco_bbox(points, rotation) 
        if annotation_type == AnnotationType.bbox 
        else __to_bbox(annotation_type, points)
    )
    annotation["rotation"] = rotation
    annotation["category_id"] = category["id"]
    annotation["id"] = id_
    annotation["attributes"] = annotation_attributes
    return annotation


def __rotate_point(
    cx: float, cy: float, angle: float, px: float, py: float
) -> np.ndarray:
    px -= cx
    py -= cy

    x_new = px * math.cos(angle) - py * math.sin(angle)
    y_new = px * math.sin(angle) + py * math.cos(angle)

    px = x_new + cx
    py = y_new + cy
    return np.array([px, py])


def __get_rotated_rectangle_coordinates(
    coords: np.ndarray, rotation: int
) -> np.ndarray:
    top_left = coords[0]
    bottom_right = coords[1]

    cx = (top_left[0] + bottom_right[0]) / 2
    cy = (top_left[1] + bottom_right[1]) / 2

    top_right = np.array([bottom_right[0], top_left[1]])
    bottom_left = np.array([top_left[0], bottom_right[1]])

    corners = np.array([top_left, top_right, bottom_right, bottom_left])

    angle_rad = math.radians(rotation)
    rotated_corners = np.array(
        [__rotate_point(cx, cy, angle_rad, x, y) for x, y in corners]
    )

    return rotated_corners

def __get_coco_bbox(
    points: list,
    rotation: int,
) -> list[float]:
    if not points:
        return []
    points_splitted = [points[idx : idx + 2] for idx in range(0, len(points), 2)]
    polygon_geo = geojson.Polygon(points_splitted)
    coords = np.array(list(geojson.utils.coords(polygon_geo)))
    rotated_coords = __get_rotated_rectangle_coordinates(coords, rotation)
    x_min = rotated_coords[:, 0].min()
    y_min = rotated_coords[:, 1].min()
    x_max = rotated_coords[:, 0].max()
    y_max = rotated_coords[:, 1].max()
    return [
        x_min,  # x
        y_min,  # y
        x_max - x_min,  # width
        y_max - y_min,  # height
    ]


def __get_without_hollowed_points(points: list) -> list:
    return [region[0] for region in points]


def __to_coco_segmentation(annotation_type: str, points: list) -> list:
    if not points:
        return []
    if annotation_type == AnnotationType.segmentation.value:
        # Remove hollowed points
        return __get_without_hollowed_points(points)
    if annotation_type == AnnotationType.bbox.value:
        x1, y1, x2, y2 = points
        rectangle_points = [x1, y1, x2, y1, x2, y2, x1, y2, x1, y1]
        return [rectangle_points]
    return [points]


def __to_bbox(annotation_type: str, points: list) -> list:
    if not points:
        return []
    base_points = []
    if annotation_type == AnnotationType.segmentation.value:
        base_points = sum(__get_without_hollowed_points(points), [])
    else:
        base_points = points
    points_splitted = [
        base_points[idx : idx + 2] for idx in range(0, len(base_points), 2)
    ]
    polygon_geo = geojson.Polygon(points_splitted)
    coords = np.array(list(geojson.utils.coords(polygon_geo)))
    left_top_x = coords[:, 0].min()
    left_top_y = coords[:, 1].min()
    right_bottom_x = coords[:, 0].max()
    right_bottom_y = coords[:, 1].max()
    width = right_bottom_x - left_top_x
    height = right_bottom_y - left_top_y
    return [__serialize(point) for point in [left_top_x, left_top_y, width, height]]


def __to_area(annotation_type: str, points: list) -> float:
    if not points:
        return 0
    area = 0
    if annotation_type == AnnotationType.segmentation.value:
        for region in __get_without_hollowed_points(points):
            area += __calc_area(annotation_type, region)
    else:
        area = __calc_area(annotation_type, points)
    return __serialize(area)


def __calc_area(annotation_type: str, points: list) -> float:
    if not points:
        return 0
    if annotation_type in [
        AnnotationType.bbox.value,
        AnnotationType.pose_estimation.value,
    ]:
        width = points[0] - points[2]
        height = points[1] - points[3]
        return width * height
    elif annotation_type in [
        AnnotationType.polygon.value,
        AnnotationType.segmentation.value,
    ]:
        x = points[0::2]
        y = points[1::2]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    else:
        raise Exception(f"Unsupported annotation type: {annotation_type}")


def __serialize(value: any) -> any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        float_value = float(value)
        if float_value.is_integer():
            return int(value)
        else:
            return float_value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        float_value = float(value)
        if float_value.is_integer():
            return int(value)
        else:
            return float_value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# YOLO


def to_yolo(project_type: str, tasks: list, classes: list, output_dir: str) -> tuple:
    if len(classes) == 0:
        coco = to_coco(project_type=project_type, tasks=tasks, output_dir=output_dir)
        return __coco2yolo(coco)
    else:
        return __to_yolo(
            project_type=project_type,
            tasks=tasks,
            classes=classes,
            output_dir=output_dir,
        )


def __coco2yolo(coco: dict) -> tuple:
    categories = coco["categories"]

    annos = []
    for image in coco["images"]:
        dw = 1.0 / image["width"]
        dh = 1.0 / image["height"]

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
        anno = {"filename": image["file_name"], "object": objs}
        annos.append(anno)

    return annos, categories


def __to_yolo(project_type: str, tasks: list, classes: list, output_dir: str) -> tuple:
    annos = []
    for task in tasks:
        if task["height"] == 0 or task["width"] == 0:
            continue

        if is_video_project_type(project_type):
            image_file_names = _export_image_files_for_video_task(
                task, str((Path(output_dir) / "images").resolve())
            )

            def get_annotation_points(anno, index):
                return _get_annotation_points_for_video_annotation(anno, index)

        else:
            image_file_names = [task["name"]]

            def get_annotation_points(anno, _):
                return _get_annotation_points_for_image_annotation(anno)

        for index, image_file_name in enumerate(image_file_names, 1):
            params = [
                {
                    "annotation_value": annotation["value"],
                    "annotation_type": annotation["type"],
                    "annotation_points": get_annotation_points(annotation, index),
                    "width": task["width"],
                    "height": task["height"],
                    "classes": classes,
                }
                for annotation in task["annotations"]
            ]
            with ThreadPoolExecutor(max_workers=8) as executor:
                image_anno_dicts = executor.map(__get_yolo_annotation, params)

            filtered_image_anno_dicts = list(filter(None, image_anno_dicts))

            anno = {"filename": image_file_name}

            if len(filtered_image_anno_dicts) > 0:
                anno["object"] = [
                    " ".join(anno)
                    for anno in sorted(filtered_image_anno_dicts, key=itemgetter(0))
                    if anno
                ]

            annos.append(anno)

    categories = map(lambda val: {"name": val}, sorted(classes))

    return annos, categories


def __get_yolo_annotation(data: dict) -> dict:
    points = data["annotation_points"]
    annotation_type = data["annotation_type"]
    value = data["annotation_value"]
    classes = list(data["classes"])
    if (
        annotation_type != AnnotationType.bbox.value
        and annotation_type != AnnotationType.polygon.value
    ):
        return None
    if not points or len(points) == 0:
        return None
    if annotation_type == AnnotationType.bbox.value and (
        int(points[0]) == int(points[2]) or int(points[1]) == int(points[3])
    ):
        return None
    if value not in classes:
        return None

    dw = 1.0 / data["width"]
    dh = 1.0 / data["height"]

    bbox = __to_bbox(annotation_type, points)
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
    multiplier = 10**decimals
    return int(n * multiplier) / multiplier


# Pascal VOC


def to_pascalvoc(project_type: str, tasks: list, output_dir: str) -> list:
    pascalvoc = []
    for task in tasks:
        if task["height"] == 0 or task["width"] == 0:
            continue

        if is_video_project_type(project_type):
            image_file_names = _export_image_files_for_video_task(
                task, str((Path(output_dir) / "images").resolve())
            )

            def get_annotation_points(anno, index):
                return _get_annotation_points_for_video_annotation(anno, index)

        else:
            image_file_names = [task["name"]]

            def get_annotation_points(anno, _):
                return _get_annotation_points_for_image_annotation(anno)

        for index, image_file_name in enumerate(image_file_names, 1):
            params = [
                {
                    "annotation_type": annotation["type"],
                    "annotation_value": annotation["value"],
                    "annotation_points": get_annotation_points(annotation, index),
                    "annotation_attributes": annotation["attributes"],
                }
                for annotation in task["annotations"]
            ]

            with ThreadPoolExecutor(max_workers=8) as executor:
                pascalvoc_objs = executor.map(__get_pascalvoc_obj, params)

            filtered_pascalvoc_objs = list(filter(None, pascalvoc_objs))

            voc = {
                "annotation": {
                    "filename": image_file_name,
                    "size": {
                        "width": task["width"],
                        "height": task["height"],
                        "depth": 3,
                    },
                    "segmented": 0,
                }
            }

            if len(filtered_pascalvoc_objs) > 0:
                voc["annotation"]["object"] = list(
                    sorted(filtered_pascalvoc_objs, key=itemgetter("name"))
                )

            pascalvoc.append(voc)
    return pascalvoc


def __get_pascalvoc_obj(data: dict) -> dict:
    points = data["annotation_points"]
    type = data["annotation_type"]
    value = data["annotation_value"]
    attributes = data["annotation_attributes"]
    if type != AnnotationType.bbox.value and type != AnnotationType.polygon.value:
        return None
    if not points or len(points) == 0:
        return None
    if type == AnnotationType.bbox.value and (
        int(points[0]) == int(points[2]) or int(points[1]) == int(points[3])
    ):
        return None
    bbox = __to_bbox(type, points)
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]

    return {
        "name": value,
        "pose": "Unspecified",
        "truncated": __get_pascalvoc_tag_value(attributes, "truncated"),
        "occluded": __get_pascalvoc_tag_value(attributes, "occluded"),
        "difficult": __get_pascalvoc_tag_value(attributes, "difficult"),
        "bndbox": {
            "xmin": math.floor(x),
            "ymin": math.floor(y),
            "xmax": math.floor(x + w),
            "ymax": math.floor(y + h),
        },
    }


def __get_pascalvoc_tag_value(attributes: list, target_tag_name: str) -> int:
    if not attributes:
        return 0
    related_attr = next(
        (
            attribute
            for attribute in attributes
            if attribute["type"] == "switch" and attribute["key"] == target_tag_name
        ),
        None,
    )
    return int(related_attr["value"]) if related_attr else 0


# labelme


def to_labelme(tasks: list) -> list:
    labelmes = []
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
                    shape_points.append(
                        [points[0][0][i * 2], points[0][0][(i * 2) + 1]]
                    )
            else:
                for i in range(int(len(points) / 2)):
                    shape_points.append([points[i * 2], points[(i * 2) + 1]])

            shape = {
                "label": annotation["value"],
                "points": shape_points,
                "group_id": None,
                "shape_type": shape_type,
                "flags": {},
            }
            shapes.append(shape)
        labelmes.append(
            {
                "version": "4.5.9",
                "flags": {},
                "shapes": shapes,
                "imagePath": task["name"],
                "imageData": None,
                "imageHeight": task["height"],
                "imageWidth": task["width"],
            }
        )
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
                    xmin,
                    ymin,
                    xmax,
                    ymin,
                    xmax,
                    ymax,
                    xmin,
                    ymax,
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
            new_points.append(points[i * 2])
            new_points.append(points[i * 2 + 1])

        if new_points[-2] == points[i * 2] and new_points[-1] == points[i * 2 + 1]:
            continue

        if len(new_points) <= 2:
            new_points.append(points[i * 2])
            new_points.append(points[i * 2 + 1])
        else:
            if new_points[-4] == new_points[-2] and new_points[-2] == points[i * 2]:
                new_points.pop()
                new_points.pop()
                new_points.append(points[i * 2])
                new_points.append(points[i * 2 + 1])
            elif (
                new_points[-3] == new_points[-1] and new_points[-1] == points[i * 2 + 1]
            ):
                new_points.pop()
                new_points.pop()
                new_points.append(points[i * 2])
                new_points.append(points[i * 2 + 1])
            else:
                new_points.append(points[i * 2])
                new_points.append(points[i * 2 + 1])
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

        prev_x = int(points[(i - 1) * 2])
        prev_y = int(points[(i - 1) * 2 + 1])
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


def execute_coco_to_fastlabel(coco: dict, annotation_type: str) -> dict:
    coco_images = {}
    for c in coco["images"]:
        coco_images[c["id"]] = c["file_name"]

    coco_categories = {}
    coco_categories_keypoints = {}
    for c in coco["categories"]:
        coco_categories[c["id"]] = c["name"] if c.get("name") else c["supercategory"]
        coco_categories_keypoints[c["id"]] = (
            c["keypoints"] if c.get("keypoints") else []
        )

    coco_annotations = coco["annotations"]

    results = {}
    for coco_image_key in coco_images:
        target_coco_annotations = filter(
            lambda annotation: annotation["image_id"] == coco_image_key,
            coco_annotations,
        )
        if not target_coco_annotations:
            return

        annotations = []
        for target_coco_annotation in target_coco_annotations:
            attributes_items = target_coco_annotation.get("attributes", {})
            attributes = [
                {"key": attribute_key, "value": attribute_value}
                for attribute_key, attribute_value in attributes_items.items()
            ]
            category_name = coco_categories[target_coco_annotation["category_id"]]
            if not category_name:
                return

            if annotation_type in [
                AnnotationType.bbox.value,
                AnnotationType.polygon.value,
            ]:
                segmentation = target_coco_annotation["segmentation"][0]
                annotation_type = ""
                if len(segmentation) == 4:
                    annotation_type = AnnotationType.bbox.value
                if len(segmentation) > 4:
                    annotation_type = AnnotationType.polygon.value
                annotations.append(
                    {
                        "value": category_name,
                        "points": segmentation,
                        "type": annotation_type,
                        "attributes": attributes,
                    }
                )
            elif annotation_type == AnnotationType.pose_estimation.value:
                keypoints = []
                target_coco_annotation_keypoints = target_coco_annotation["keypoints"]
                keypoint_keys = coco_categories_keypoints[
                    target_coco_annotation["category_id"]
                ]
                # coco keypoint style [100,200,1,300,400,1,500,600,2] convert to [[100,200,1],[300,400,1],[500,600,2]]
                keypoint_values = [
                    target_coco_annotation_keypoints[i : i + 3]
                    for i in range(0, len(target_coco_annotation_keypoints), 3)
                ]
                for index, keypoint_key in enumerate(keypoint_keys):
                    keypoint_value = keypoint_values[index]
                    if keypoint_value[2] == 0:
                        continue
                    if not keypoint_value[2] in [1, 2]:
                        raise FastLabelInvalidException(
                            f"Visibility flag must be 0 or 1, 2 . annotation_id: {target_coco_annotation['id']}",
                            422,
                        )
                    # fastlabel occulusion is 0 or 1 . coco occulusion is 1 or 2.
                    keypoint_value[2] = keypoint_value[2] - 1
                    keypoints.append({"key": keypoint_key, "value": keypoint_value})

                annotations.append(
                    {
                        "value": category_name,
                        "type": annotation_type,
                        "keypoints": keypoints,
                        "attributes": attributes,
                    }
                )
            else:
                raise FastLabelInvalidException(
                    "Annotation type must be bbox or polygon ,pose_estimation.", 422
                )

        results[coco_images[coco_image_key]] = annotations
    return results


def execute_labelme_to_fastlabel(labelme: dict, file_path: str = None) -> tuple:
    file_name = ""
    if file_path:
        file_name = file_path.replace(
            ".json", os.path.splitext(labelme["imagePath"])[1]
        )
    else:
        file_name = labelme["imagePath"]
    labelme_annotations = labelme["shapes"]

    annotations = []
    for labelme_annotation in labelme_annotations:
        label = labelme_annotation["label"]
        if not label:
            return

        points = np.ravel(labelme_annotation["points"])
        annotation_type = __get_annotation_type_by_labelme(
            labelme_annotation["shape_type"]
        )
        annotations.append(
            {"value": label, "points": points.tolist(), "type": annotation_type}
        )

    return (file_name, annotations)


def execute_pascalvoc_to_fastlabel(pascalvoc: dict, file_path: str = None) -> tuple:
    target_pascalvoc = pascalvoc["annotation"]
    file_name = ""  # file_path if file_path else target_pascalvoc["filename"]
    if file_path:
        file_name = file_path.replace(
            ".xml", os.path.splitext(target_pascalvoc["filename"])[1]
        )
    else:
        file_name = target_pascalvoc["filename"]
    pascalvoc_annotations = target_pascalvoc["object"]
    if not isinstance(pascalvoc_annotations, list):
        pascalvoc_annotations = [pascalvoc_annotations]

    annotations = []
    for pascalvoc_annotation in pascalvoc_annotations:
        category_name = pascalvoc_annotation["name"]
        if not category_name:
            return

        points = [
            int(pascalvoc_annotation["bndbox"][item])
            for item in pascalvoc_annotation["bndbox"]
        ]
        annotations.append(
            {
                "value": category_name,
                "points": points,
                "type": AnnotationType.bbox.value,
            }
        )

    return (file_name, annotations)


def execute_yolo_to_fastlabel(
    classes: dict,
    image_sizes: dict,
    yolo_annotations: dict,
    dataset_folder_path: str = None,
) -> dict:
    results = {}
    for yolo_anno_key in yolo_annotations:
        annotations = []
        for each_image_annotation in yolo_annotations[yolo_anno_key]:
            (
                yolo_class_id,
                yolo_center_x_ratio,
                yolo_center_y_ratio,
                yolo_anno_width_ratio,
                yolo_anno_height_ratio,
            ) = each_image_annotation
            image_width, image_height = image_sizes[yolo_anno_key]["size"]

            classs_name = classes[str(yolo_class_id)]

            yolo_center_x_point = float(image_width) * float(yolo_center_x_ratio)
            yolo_center_y_point = float(image_height) * float(yolo_center_y_ratio)
            yolo_anno_width_size = float(image_width) * float(yolo_anno_width_ratio)
            yolo_anno_height_size = float(image_height) * float(yolo_anno_height_ratio)

            points = []
            points.append(yolo_center_x_point - (yolo_anno_width_size / 2))  # x1
            points.append(yolo_center_y_point - (yolo_anno_height_size / 2))  # y1
            points.append(yolo_center_x_point + (yolo_anno_width_size / 2))  # x2
            points.append(yolo_center_y_point + (yolo_anno_height_size / 2))  # y2
            annotations.append(
                {
                    "value": classs_name,
                    "points": points,
                    "type": AnnotationType.bbox.value,
                }
            )

        file_path = (
            image_sizes[yolo_anno_key]["image_file_path"].replace(
                os.path.join(*[dataset_folder_path, ""]), ""
            )
            if dataset_folder_path
            else image_sizes[yolo_anno_key]["image_file_path"]
        )
        results[file_path] = annotations

    return results


def __get_annotation_type_by_labelme(shape_type: str) -> str:
    if shape_type == "rectangle":
        return "bbox"
    if shape_type == "polygon":
        return "polygon"
    if shape_type == "point":
        return "keypoint"
    if shape_type == "line":
        return "line"
    return None


@contextmanager
def VideoCapture(*args, **kwds):
    videoCapture = cv2.VideoCapture(*args, **kwds)
    try:
        yield videoCapture
    finally:
        videoCapture.release()


def _download_file(url: str, output_file_path: str, chunk_size: int = 8192) -> str:
    with requests.get(url, stream=True) as stream:
        stream.raise_for_status()
        with open(file=output_file_path, mode="wb") as file:
            for chunk in stream.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
            return file.name


def _export_image_files_for_video_file(
    file_path: str,
    output_dir_path: str,
    basename: str,
):
    image_file_names = []
    with VideoCapture(file_path) as cap:
        if not cap.isOpened():
            raise FastLabelInvalidException(
                (
                    "Video to image conversion failed. Video could not be opened.",
                    " Download may have failed or there is a problem with the video.",
                ),
                422,
            )
        digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_file_name = f"{basename}_{str(frame_num).zfill(digit)}.jpg"
            image_file_path = os.path.join(output_dir_path, image_file_name)
            os.makedirs(output_dir_path, exist_ok=True)
            cv2.imwrite(image_file_path, frame)
            frame_num += 1
            image_file_names.append(image_file_name)
    return image_file_names


def _export_image_files_for_video_task(video_task: dict, output_dir_path: str):
    with NamedTemporaryFile(prefix="fastlabel-sdk-") as video_file:
        video_file_path = _download_file(
            url=video_task["url"], output_file_path=video_file.name
        )
        return _export_image_files_for_video_file(
            file_path=video_file_path,
            output_dir_path=output_dir_path,
            basename=Path(video_task["name"]).stem,
        )


def _get_annotation_points_for_video_annotation(annotation: dict, index: int):
    points = annotation.get("points")
    if not points:
        return None
    video_point_datum = points.get(str(index))
    if not video_point_datum:
        return None
    return video_point_datum["value"]


def _get_annotation_points_for_image_annotation(annotation: dict):
    return annotation.get("points")


def _get_coco_annotation_attributes(annotation: dict) -> Dict[str, AttributeValue]:
    coco_attributes = {}
    attributes = annotation.get("attributes")
    if not attributes:
        return coco_attributes
    for attribute in attributes:
        coco_attributes[attribute["key"]] = attribute["value"]
    return coco_attributes

