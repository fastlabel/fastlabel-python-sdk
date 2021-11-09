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

def color_code_to_rgb(color_code: str) -> tuple[int, int, int]:
    R = int(color_code[1:3], 16)
    G = int(color_code[3:5], 16)
    B = int(color_code[5:7], 16)
    return (R, G, B)


def color_code_to_bgr(color_code: str) -> tuple[int, int, int]:
    R, G, B = color_code_to_rgb(color_code)
    return (B, G, R)
