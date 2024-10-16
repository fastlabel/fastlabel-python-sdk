from typing import Any, Optional, Sequence, Union, cast

import cv2
import numpy as np
from cv2 import Mat, UMat
from PIL.Image import Image


def mask_to_polygon(mask_image: Union[str, np.ndarray]) -> list[list[list[int]]]:
    MIN_POLYGON_POINTS_LENGTH = 6
    if isinstance(mask_image, str):
        mask_image_path = mask_image
        mask_image = cv2.imread(mask_image_path)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    points = []
    contours, hierarchy = cv2.findContours(
        mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contours = list(map(lambda x: cv2.approxPolyDP(x, 3, True), contours))
    if len(contours) > 0:
        for i in range(0, len(contours)):
            points_array = contours[i][::-1]
            points = points_array.ravel().tolist()
            if len(points) < MIN_POLYGON_POINTS_LENGTH:
                continue
    return points


def mask_to_segmentation(
    mask_image: Union[str, np.ndarray, Mat, UMat, Image]
) -> list[list[list[int]]]:
    if isinstance(mask_image, str):
        mask_image_path = mask_image
        mask_image = cv2.imread(mask_image_path)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(
        image=mask_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
    )
    if len(contours) == 0:
        return []

    segmentation_list = __detect_segmentation_list(hierarchy, contours)
    points = [
        __convert_fastlabel_segmentation(segmentation)
        for segmentation in segmentation_list
        if len(segmentation) > 0
    ]
    return points


def __detect_segmentation_list(
    hierarchy: np.ndarray, contours: Sequence[Union[np.ndarray, Mat]]
) -> list[Any]:

    outer_polygon_hierarchy_indexes = list(
        filter(lambda x: hierarchy[0][x][3] == -1, range(hierarchy[0].shape[0]))
    )
    if len(outer_polygon_hierarchy_indexes) == 0:
        return []
    segmentation_list = []

    for outer_polygon_hierarchy_index in outer_polygon_hierarchy_indexes:
        segmentation, separate_place_hierarchy_indexes = __detect_segmentation(
            outer_polygon_hierarchy_index, hierarchy, contours, []
        )
        segmentation_list.append(segmentation)

        for separate_place_hierarchy_index in separate_place_hierarchy_indexes:
            (segmentation, _,) = __detect_segmentation(
                separate_place_hierarchy_index, hierarchy, contours
            )
            segmentation_list.append(segmentation)
    return segmentation_list


def __detect_segmentation(
    parent_polygon_hierarchy_index: int,
    hierarchy: np.ndarray,
    contours: Sequence[np.ndarray],
    hierarchy_indexes: Optional[list[int]] = None,
    is_recursive: bool = True,
) -> tuple[list, list]:
    separate_place_hierarchy_indexes: list[int] = hierarchy_indexes or []
    base_polygon = contours[parent_polygon_hierarchy_index].reshape(-1, 2)
    segmentation: list[np.ndarray] = [base_polygon]
    parent_hierarchy = hierarchy[0][parent_polygon_hierarchy_index]
    is_outer_hierarchy = parent_hierarchy[3] == -1
    child_hierarchy_index = (
        parent_hierarchy[2] if is_outer_hierarchy else parent_hierarchy[0]
    )
    has_child = child_hierarchy_index != -1
    if not is_recursive or (
        not has_child or hierarchy[0][child_hierarchy_index][3] == -1
    ):
        return segmentation, separate_place_hierarchy_indexes
    else:
        next_hierarchy_index = child_hierarchy_index
        while True:
            has_child = hierarchy[0][next_hierarchy_index][2] != -1
            hole, separate_place_hierarchy_indexes = __detect_segmentation(
                cast(int, next_hierarchy_index),
                hierarchy,
                contours,
                separate_place_hierarchy_indexes,
                is_recursive=False,
            )
            segmentation.extend(hole)
            if has_child:
                separate_place_hierarchy_indexes.append(
                    int(hierarchy[0][next_hierarchy_index][2])
                )
            next_hierarchy_index = hierarchy[0][next_hierarchy_index][0]
            is_last_child = next_hierarchy_index == -1
            if is_last_child:
                break
        return segmentation, separate_place_hierarchy_indexes


def __convert_fastlabel_segmentation(segmentation: list) -> list[Any]:
    MIN_POLYGON_POINTS_LENGTH = 10
    new_points = []
    for annotation in segmentation:
        converted_points = __convert_fastlabel_points(np.array(annotation))
        if len(converted_points) >= MIN_POLYGON_POINTS_LENGTH:
            new_points.append(converted_points)
    return new_points


# TODO: maybe return points number is not correct. (ex. only 2 points)
def __convert_fastlabel_points(cv2_polygon_array: np.ndarray) -> list[int]:
    # Notices
    #   polygon_array → anticlockwise coordinates
    #   return value → clockwise coordinates
    polygon_array = (
        np.append(cv2_polygon_array, [cv2_polygon_array[0]], axis=0)
        if not np.array_equal(cv2_polygon_array[0], cv2_polygon_array[-1])
        else cv2_polygon_array
    )
    x_array = polygon_array[:, 0]
    y_array = polygon_array[:, 1]
    # Rectangle with height of 1px
    if np.all(x_array == x_array[0]):
        x_min = x_array[0]
        x_max = x_array[0] + 1
        y_max = y_array.max() + 1
        y_min = y_array.min()
        left_up = [x_min, y_min]
        left_down = [x_min, y_max]
        right_down = [x_max, y_max]
        right_up = [x_max, y_min]
        return (
            np.flipud(np.array([left_up, left_down, right_down, right_up, left_up]))
            .flatten()
            .tolist()
        )
    # Rectangle with width of 1px
    elif np.all(y_array == y_array[0]):
        x_min = x_array.min()
        x_max = x_array.max() + 1
        y_max = y_array[0] + 1
        y_min = y_array[0]
        left_up = [x_min, y_min]
        left_down = [x_min, y_max]
        right_down = [x_max, y_max]
        right_up = [x_max, y_min]
        return (
            np.flipud(np.array([left_up, left_down, right_down, right_up, left_up]))
            .flatten()
            .tolist()
        )

    new_array = []
    for current_index, current_value in enumerate(polygon_array.tolist()):
        prev_index = current_index - 1
        next_index = current_index + 1
        x_current_value = current_value[0]
        y_current_value = current_value[1]

        if current_index == 0:
            prev_index = (
                len(polygon_array) - 1
                if np.all(polygon_array[1] == polygon_array[len(polygon_array) - 1])
                else len(polygon_array) - 2
            )
            next_index = current_index + 1
        elif current_index == len(polygon_array) - 1:
            prev_index = current_index - 1
            next_index = 0

        prev_value = polygon_array[prev_index]
        [x_prev_value, y_prev_value] = [prev_value[0], prev_value[1]]
        next_value = polygon_array[next_index]
        [x_next_value, y_next_value] = [next_value[0], next_value[1]]

        # The pixel coordinates of the contour are shifted diagonally
        if x_prev_value != x_current_value and y_prev_value != y_current_value:
            # 　Diagonal left down
            if x_prev_value > x_current_value and y_prev_value < y_current_value:
                new_array.append([x_current_value + 1, y_current_value])
                if x_next_value == x_current_value and y_next_value > y_current_value:
                    new_array.append([x_current_value, y_current_value])
                if x_next_value > x_current_value and y_next_value == y_current_value:
                    new_array.append([x_current_value, y_current_value + 1])
                continue
            # 　Diagonal right up
            elif x_prev_value < x_current_value and y_prev_value > y_current_value:
                if len(new_array) and not np.all(prev_value <= np.array(new_array[-1])):
                    new_array.extend(
                        [
                            [
                                x_prev_value,
                                y_prev_value + 1,
                            ],
                            (prev_value + 1).tolist(),
                        ]
                    )
                new_array.append([x_current_value, y_current_value + 1])
                if x_next_value < x_current_value and y_next_value < y_current_value:
                    new_array.append([x_current_value + 1, y_current_value + 1])
                    new_array.append([x_current_value + 1, y_current_value])
                continue
            # 　Diagonal left up
            elif x_prev_value > x_current_value and y_prev_value > y_current_value:
                new_array.append([x_current_value + 1, y_current_value])
                if x_next_value == x_current_value and y_next_value > y_current_value:
                    new_array.append([x_current_value, y_current_value])
                continue
            # 　Diagonal right down
            elif x_prev_value < x_current_value and y_prev_value < y_current_value:
                new_array.append([x_current_value - 1, y_current_value])
                if x_next_value > x_current_value and y_next_value == y_current_value:
                    new_array.append([x_current_value, y_current_value + 1])
                continue
            else:
                continue
        # 1px straight line
        elif x_prev_value == x_next_value and y_prev_value == y_next_value:
            diff = np.array(current_value) - next_value
            if diff[0] > 0:
                new_array.extend(
                    [
                        [current_value[0] + 1, current_value[1] + 1],
                        [current_value[0] + 1, current_value[1]],
                    ]
                )
                continue
            elif diff[0] < 0:
                new_array.extend(
                    [
                        [current_value[0], current_value[1]],
                        [current_value[0], current_value[1] + 1],
                    ]
                )
                continue
            if diff[1] > 0:
                new_array.extend(
                    [
                        [current_value[0], current_value[1] + 1],
                        [current_value[0] + 1, current_value[1] + 1],
                    ]
                )
                continue
            elif diff[1] < 0:
                new_array.extend(
                    [
                        [current_value[0] + 1, current_value[1]],
                        [current_value[0], current_value[1]],
                    ]
                )
                continue
        # 　Turn 90 degrees
        else:
            if (
                x_prev_value < x_current_value and y_prev_value == y_current_value
            ) and (x_next_value == x_current_value and y_next_value < y_current_value):
                new_array.append([x_current_value + 1, y_current_value + 1])
                continue
            elif (
                x_prev_value > x_current_value and y_prev_value == y_current_value
            ) and (x_next_value == x_current_value and y_next_value > y_current_value):
                new_array.append([x_current_value, y_current_value])
                continue
            elif (
                x_prev_value == x_current_value and y_prev_value > y_current_value
            ) and (x_next_value < x_current_value and y_next_value <= y_current_value):
                new_array.append([x_current_value + 1, y_current_value])
                continue
            elif (
                x_prev_value == x_current_value and y_prev_value < y_current_value
            ) and (x_next_value > x_current_value and y_next_value == y_current_value):
                new_array.append([x_current_value, y_current_value + 1])
                continue
            elif (
                x_prev_value > x_current_value and y_prev_value == y_current_value
            ) and (x_next_value == x_current_value and y_next_value > y_current_value):
                new_array.append([x_current_value + 1, y_current_value])
                continue
    new_point_array = np.flipud(np.array(new_array)).flatten().tolist()
    if not (
        (new_point_array[0] == new_point_array[-2])
        and (new_point_array[1] == new_point_array[-1])
    ):
        new_point_array.extend([new_point_array[0], new_point_array[1]])
    return new_point_array
