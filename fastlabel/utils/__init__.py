import base64
import json
import os
from typing import List
from uuid import uuid4

import cv2
import geojson
import numpy as np

from fastlabel import const

from .mask_image_util import mask_to_segmentation  # noqa: F401


def base64_encode(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def is_image_supported_ext(file_path: str) -> bool:
    return file_path.lower().endswith((".png", ".jpg", ".jpeg"))


def is_video_supported_codec(file_path: str) -> bool:
    return get_video_fourcc(file_path) in const.SUPPORTED_FOURCC


def is_video_supported_ext(file_path: str) -> bool:
    return file_path.lower().endswith(".mp4")


def is_text_supported_ext(file_path: str) -> bool:
    return file_path.lower().endswith(".txt")


def is_audio_supported_ext(file_path: str) -> bool:
    return file_path.lower().endswith((".mp3", ".wav", ".m4a"))


def is_dicom_supported_ext(file_path: str) -> bool:
    return file_path.lower().endswith((".zip"))


def is_appendix_supported_ext(file_path: str) -> bool:
    return file_path.lower().endswith((".zip"))


def is_pcd_supported_ext(file_path: str) -> bool:
    # .ply is not yet supported. To support it, modification of the API
    # needs to be considered as well.
    return file_path.lower().endswith((".pcd"))


def is_image_supported_size(file_path: str) -> bool:
    return os.path.getsize(file_path) <= const.SUPPORTED_IMAGE_SIZE


def is_image_supported_size_for_inference(file_path: str) -> bool:
    return os.path.getsize(file_path) <= const.SUPPORTED_INFERENCE_IMAGE_SIZE


def is_video_supported_size(file_path: str) -> bool:
    return os.path.getsize(file_path) <= const.SUPPORTED_VIDEO_SIZE


def is_text_supported_size(file_path: str) -> bool:
    return os.path.getsize(file_path) <= const.SUPPORTED_TEXT_SIZE


def is_audio_supported_size(file_path: str) -> bool:
    return os.path.getsize(file_path) <= const.SUPPORTED_AUDIO_SIZE


def is_dicom_supported_size(file_path: str) -> bool:
    return os.path.getsize(file_path) <= const.SUPPORTED_DICOM_SIZE


def is_pcd_supported_size(file_path: str) -> bool:
    return os.path.getsize(file_path) <= const.SUPPORTED_PCD_SIZE


def is_object_supported_size(file_path: str) -> bool:
    return os.path.getsize(file_path) <= const.SUPPORTED_OBJECT_SIZE


def is_video_project_type(project_type: str):
    return type(project_type) is str and project_type.startswith("video_")


def is_json_ext(file_name: str) -> bool:
    return file_name.lower().endswith(".json")


def get_basename(file_path: str) -> str:
    """
    e.g.) file.jpg -> file
          path/to/file.jpg -> path/to/file
    """
    return os.path.splitext(file_path)[0]


def get_supported_image_ext() -> list:
    return ["jpg", "jpeg", "png"]


def reverse_points(points: List[int]) -> List[int]:
    """
    e.g.)
    [4, 5, 4, 9, 8, 9, 8, 5, 4, 5] => [4, 5, 8, 5, 8, 9, 4, 9, 4, 5]
    """
    reversed_points = []
    for index, _ in enumerate(points):
        if index % 2 == 0:
            reversed_points.insert(0, points[index + 1])
            reversed_points.insert(0, points[index])
    return reversed_points


def sort_segmentation_points(points: List[int]) -> List[int]:
    """
    e.g.)
    [1, 2, 1, 1, 2, 1, 2, 2, 1, 2] => [1, 1, 2, 1, 2, 2, 1, 2, 1, 1]
    """
    points_array = np.array(points).reshape((-1, 2))[1:]
    base_point_index = 0
    points_list = points_array.tolist()
    for index, val in enumerate(points_list):
        if index == 0:
            continue
        if (
            val[1] <= points_list[base_point_index][1]
            and val[0] <= points_list[base_point_index][0]
        ):
            base_point_index = index
    new_points_array = np.vstack(
        [
            points_array[base_point_index:],
            points_array[:base_point_index],
            np.array([points_array[base_point_index]]),
        ]
    )
    return new_points_array.ravel().tolist()


def is_clockwise(points: list) -> bool:
    """
    points: [x1, y1, x2, y2, x3, y3, ... xn, yn]
    Sum over the edges, (x2 − x1)(y2 + y1).
    If the result is positive the curve is clockwise,
    if it's negative the curve is counter-clockwise.

    The above is assumes a normal Cartesian coordinate system.
    HTML5 canvas, use an inverted Y-axis.
    Therefore If the area is negative, the curve is clockwise.
    """
    points_splitted = [points[idx : idx + 2] for idx in range(0, len(points), 2)]
    polygon_geo = geojson.Polygon(points_splitted)
    coords = np.array(list(geojson.utils.coords(polygon_geo)))
    xs, ys = map(list, zip(*coords))
    xs.append(xs[0])
    ys.append(ys[0])
    sum_edges = (
        sum(
            (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1])
            for i in range(1, len(points_splitted))
        )
        / 2.0
    )

    if sum_edges < 0:
        return True
    return False


def get_json_length(value) -> int:
    json_str = json.dumps(value)
    return len(json_str)


def get_video_fourcc(video_path: str) -> str:
    cap = cv2.VideoCapture(video_path)
    fourcc_code = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)])
    cap.release()
    return fourcc_str


def get_uuid() -> str:
    return str(uuid4())
