import os
import base64
import numpy as np
import geojson
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

def is_clockwise(points: list) -> bool:
    """
    points: [x1, y1, x2, y2, x3, y3, ... xn, yn]
    Sum over the edges, (x2 − x1)(y2 + y1). 
    If the result is positive the curve is clockwise, if it's negative the curve is counter-clockwise.
    
    The above is assumes a normal Cartesian coordinate system. 
    HTML5 canvas, use an inverted Y-axis.
    Therefore If the area is negative, the curve is clockwise.
    """
    points_splitted = [points[idx:idx + 2]
                                    for idx in range(0, len(points), 2)]
    polygon_geo = geojson.Polygon(points_splitted)
    coords = np.array(list(geojson.utils.coords(polygon_geo)))
    xs, ys = map(list, zip(*coords))
    xs.append(xs[0])
    ys.append(ys[0])
    sum_edges = sum((xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) for i in range(1, len(points_splitted))) / 2.0

    if sum_edges < 0:
        return True
    return False 
