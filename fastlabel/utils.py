import os
import base64
from typing import List


def base64_encode(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def is_image_supported_ext(file_path: str) -> bool:
    return file_path.lower().endswith(('.png', '.jpg', '.jpeg'))


def is_video_supported_ext(file_path: str) -> bool:
    return file_path.lower().endswith(('.mp4'))


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
            reversed_points.insert(
                0, points[index + 1])
            reversed_points.insert(
                0, points[index])
    return reversed_points
