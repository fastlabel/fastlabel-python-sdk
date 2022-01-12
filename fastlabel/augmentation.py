import json
import os
import cv2
import random
import numpy as np
import math
import skimage
import copy
import codecs
from typing import Tuple, Dict, Any, List


class Augmentation:

    AUGMENTATION_LIST = [
        "90_degree_rotation_image_level",
        "crop_image_level",
        "rotation_image_level",
        "brightness_image_level",
        "blur_image_level",
        "noise_image_level",
        "exposure_image_level",
        "flip_bounding_box_level",
        "90_degree_rotation_bounding_box_level",
        "crop_bounding_box_level",
        "rotation_bounding_box_level",
        "shear_bounding_box_level",
        "brightness_bounding_box_level",
        "exposure_bounding_box_level",
        "blur_bounding_box_level",
        "noise_bounding_box_level"
    ]

    def __init__(
            self,
            max_rotation_angle: int = 360,
            max_kernel_size_for_blur: int = 25,
            keep_image_ratio_for_crop: float = 0.5,
            shear_angle_range: Tuple[int, int] = (0, 200),
            alpha_range_for_brightness: Tuple[float, float] = (0.2, 2),
            beta_range_for_brightness: Tuple[int, int] = (-100, 100),
            gamma_range_for_exposure: Tuple[float, float]=(0, 1),
            is_fill_average_color: bool = True) -> None:
        self._max_rotation_angle = max_rotation_angle
        self._max_kernel_size_for_blur = max_kernel_size_for_blur
        self._keep_image_ratio_for_crop = keep_image_ratio_for_crop
        self._shear_angle_range = shear_angle_range
        self._alpha_range_for_brightness = alpha_range_for_brightness
        self._beta_range_for_brightness = beta_range_for_brightness
        self._gamma_range_for_exposure = gamma_range_for_exposure
        self._is_fill_average_color = is_fill_average_color

    def execute(
            self,
            image_dir_path: str,
            annotation_json_path: str,
            output_dir_path: str = "./augmentation",
            is_debug: bool = False) -> None:

        annotation_list = json.load(open(annotation_json_path))

        for augmentation_name in self.AUGMENTATION_LIST:
            augmentation_method = getattr(self, f"_{augmentation_name}")

            self._create_output_image_dir(f"{output_dir_path}/{augmentation_name}", is_debug)

            processed_annotation_list = []
            for annotation in copy.deepcopy(annotation_list):
                image_name = annotation["name"]
                image_path = f"{image_dir_path}/{image_name}"

                if not os.path.isfile(image_path):
                    continue

                image = cv2.imread(image_path)

                processed_image, processed_annotation = augmentation_method(
                    image, annotation)

                if is_debug:
                    image_draw = self._debugger(processed_image, processed_annotation)
                    self._save_debug_image(image_draw, image_name, output_dir_path, augmentation_name)

                self._save_image(
                    processed_image, image_name, output_dir_path, augmentation_name)

                processed_annotation_list.append(processed_annotation)

            self._save_json(processed_annotation_list, output_dir_path, augmentation_name)

    def _90_degree_rotation_image_level(
            self,
            image: np.ndarray,
            annotation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        rotation_angle = random.choice([90, 180, 270])
        processed_image = self._rotation_image(image, rotation_angle)

        processed_annotation = annotation.copy()
        width, height = processed_annotation["width"], processed_annotation["height"]
        origin_before = (width / 2, height / 2)

        if not rotation_angle == 180:
            processed_annotation["width"], processed_annotation["height"] = height, width

        origin_after = (processed_annotation["width"] / 2, processed_annotation["height"] / 2)

        processed_annotation_object_list = []
        for annotation_object in processed_annotation["annotations"]:
            point_list_flatten = annotation_object["points"]

            processed_point_list_flatten = []
            for point_index in range(0, len(point_list_flatten) - 1, 2):
                x, y = point_list_flatten[point_index], point_list_flatten[point_index + 1]
                processed_x , processed_y = self._rotate_point(
                    origin_before, origin_after, (x, y), rotation_angle)
                processed_point_list_flatten.append(processed_x)
                processed_point_list_flatten.append(processed_y)

            annotation_object["points"] = processed_point_list_flatten

            processed_annotation_object_list.append(annotation_object)
        processed_annotation["annotations"] = processed_annotation_object_list

        return processed_image, processed_annotation

    def _rotation_image_level(self, image: np.ndarray, annotation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        rotation_angle = random.randint(0, self._max_rotation_angle)
        processed_image = self._rotation_image(image, rotation_angle)

        processed_annotation = annotation.copy()

        origin_before = (processed_annotation["width"] / 2, processed_annotation["height"] / 2)

        processed_annotation["height"], processed_annotation["width"] = processed_image.shape[:2]
        origin_after = (processed_annotation["width"] / 2, processed_annotation["height"] / 2)

        processed_annotation_object_list = []
        for annotation_object in processed_annotation["annotations"]:
            point_list_flatten = annotation_object["points"]

            annotation_type = annotation_object["type"]

            if annotation_type == "bbox":
                left_top = (point_list_flatten[0], point_list_flatten[1])
                right_bottom = (point_list_flatten[2], point_list_flatten[3])

                point_list_flatten = [
                    left_top[0],
                    left_top[1],
                    right_bottom[0],
                    left_top[1],
                    right_bottom[0],
                    right_bottom[1],
                    left_top[0],
                    right_bottom[1]
                ]

                annotation_object["type"] = "polygon"

            processed_point_list_flatten = []
            for point_index in range(0, len(point_list_flatten) - 1, 2):
                x, y = point_list_flatten[point_index], point_list_flatten[point_index + 1]
                processed_x, processed_y = self._rotate_point(
                    origin_before, origin_after, (x, y), rotation_angle)
                processed_point_list_flatten.append(processed_x)
                processed_point_list_flatten.append(processed_y)

            annotation_object["points"] = processed_point_list_flatten

            processed_annotation_object_list.append(annotation_object)
        processed_annotation["annotations"] = processed_annotation_object_list

        return processed_image, processed_annotation

    def _crop_image_level(self, image: np.ndarray, annotation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        processed_image, cropped_image_size, start_point = self._random_crop(
            image)

        processed_annotation = annotation.copy()

        processed_annotation["height"], processed_annotation["width"] = cropped_image_size[1], cropped_image_size[0]

        processed_annotation_object_list = []
        for annotation_object in processed_annotation["annotations"]:
            point_list_flatten = annotation_object["points"]

            processed_point_list_flatten = []
            is_out_of_frame_x = is_out_of_frame_y = True
            for point_index in range(0, len(point_list_flatten) - 1, 2):
                x, y = point_list_flatten[point_index], point_list_flatten[point_index + 1]

                processed_x = x - start_point[0]
                processed_y = y - start_point[1]

                if processed_x <= 0:
                    processed_x = 0
                else:
                    is_out_of_frame_x = False

                if processed_y <= 0:
                    processed_y = 0
                else:
                    is_out_of_frame_y = False

                processed_point_list_flatten.append(processed_x)
                processed_point_list_flatten.append(processed_y)

            if is_out_of_frame_x or is_out_of_frame_y:
                continue

            annotation_object["points"] = processed_point_list_flatten
            processed_annotation_object_list.append(annotation_object)
        processed_annotation["annotations"] = processed_annotation_object_list

        return processed_image, processed_annotation

    def _brightness_image_level(self, image: np.ndarray, annotation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self._random_brightness(image.copy()), annotation

    def _blur_image_level(self, image: np.ndarray, annotation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self._random_blur(image.copy()), annotation

    def _noise_image_level(self, image: np.ndarray, annotation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self._random_noise(image.copy()), annotation

    def _exposure_image_level(self, image: np.ndarray, annotation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self._random_exposure(image.copy()), annotation

    def _brightness_bounding_box_level(self, image: np.ndarray, annotation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        processed_image = image.copy()

        for annotation_object in annotation["annotations"]:
            point_list_flatten = annotation_object["points"]

            if annotation_object["type"] == "bbox":
                x_min, y_min, x_max, y_max = point_list_flatten
            else:
                x_min, y_min, x_max, y_max = self._get_bbox(point_list_flatten)

            cropped_image = processed_image[y_min: y_max, x_min: x_max]

            cropped_image_brightness = self._random_brightness(cropped_image)

            if annotation_object["type"] == "bbox":
                processed_image[y_min: y_max, x_min: x_max] = cropped_image_brightness
            else:
                processed_image = self._update_polygon_area(
                    processed_image, cropped_image_brightness, point_list_flatten)

        return processed_image, annotation

    def _exposure_bounding_box_level(self, image: np.ndarray, annotation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        processed_image = image.copy()

        for annotation_object in annotation["annotations"]:
            point_list_flatten = annotation_object["points"]

            if annotation_object["type"] == "bbox":
                x_min, y_min, x_max, y_max = point_list_flatten
            else:
                x_min, y_min, x_max, y_max = self._get_bbox(point_list_flatten)

            cropped_image = processed_image[y_min: y_max, x_min: x_max]

            cropped_image_exposure = self._random_exposure(cropped_image)

            if annotation_object["type"] == "bbox":
                processed_image[y_min: y_max, x_min: x_max] = cropped_image_exposure
            else:
                processed_image = self._update_polygon_area(
                    processed_image, cropped_image_exposure, point_list_flatten)

        return processed_image, annotation

    def _blur_bounding_box_level(self, image: np.ndarray, annotation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        processed_image = image.copy()

        for annotation_object in annotation["annotations"]:
            point_list_flatten = annotation_object["points"]

            if annotation_object["type"] == "bbox":
                x_min, y_min, x_max, y_max = point_list_flatten
            else:
                x_min, y_min, x_max, y_max = self._get_bbox(point_list_flatten)

            cropped_image = processed_image[y_min: y_max, x_min: x_max]

            cropped_image_blur = self._random_blur(cropped_image)

            if annotation_object["type"] == "bbox":
                processed_image[y_min: y_max, x_min: x_max] = cropped_image_blur
            else:
                processed_image = self._update_polygon_area(
                    processed_image, cropped_image_blur, point_list_flatten)

        return processed_image, annotation

    def _noise_bounding_box_level(self, image: np.ndarray, annotation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        processed_image = image.copy()

        for annotation_object in annotation["annotations"]:
            point_list_flatten = annotation_object["points"]

            if annotation_object["type"] == "bbox":
                x_min, y_min, x_max, y_max = point_list_flatten
            else:
                x_min, y_min, x_max, y_max = self._get_bbox(point_list_flatten)

            cropped_image = processed_image[y_min: y_max, x_min: x_max]

            cropped_image_noise = self._random_noise(cropped_image)

            if annotation_object["type"] == "bbox":
                processed_image[y_min: y_max, x_min: x_max] = cropped_image_noise
            else:
                processed_image = self._update_polygon_area(
                    processed_image, cropped_image_noise, point_list_flatten)

        return processed_image, annotation

    def _flip_bounding_box_level(self, image: np.ndarray, annotation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        processed_image = image.copy()

        processed_annotation = annotation.copy()

        processed_annotation_object_list = []
        for annotation_object in processed_annotation["annotations"]:
            point_list_flatten = annotation_object["points"]

            if annotation_object["type"] == "bbox":
                x_min, y_min, x_max, y_max = point_list_flatten
            else:
                x_min, y_min, x_max, y_max = self._get_bbox(point_list_flatten)

            x_center = (x_max + x_min) / 2
            y_center = (y_max + y_min) / 2

            cropped_image = processed_image[y_min: y_max, x_min: x_max]

            cropped_image_flip, flip_direction = self._random_flip(cropped_image)

            if annotation_object["type"] == "bbox":
                processed_image[y_min: y_max, x_min: x_max] = cropped_image_flip
            else:
                point_list_flatten = annotation_object["points"]
                processed_point_list_flatten = []
                for point_index in range(0, len(point_list_flatten) - 1, 2):
                    x, y = point_list_flatten[point_index], point_list_flatten[point_index + 1]

                    if flip_direction == "horizontal":
                        x = 2 * x_center - x
                        y = y

                    if flip_direction == "vertical":
                        x = x
                        y = 2 * y_center - y

                    processed_point_list_flatten.append(int(x))
                    processed_point_list_flatten.append(int(y))

                processed_image = self._update_polygon_area(
                    processed_image, cropped_image_flip, processed_point_list_flatten)

                annotation_object["points"] = processed_point_list_flatten

            processed_annotation_object_list.append(annotation_object)

        processed_annotation["annotations"] = processed_annotation_object_list

        return processed_image, processed_annotation

    def _90_degree_rotation_bounding_box_level(self, image: np.ndarray, annotation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        processed_image = image.copy()
        image_height, image_width = processed_image.shape[:2]

        processed_annotation = annotation.copy()

        processed_annotation_object_list = []
        for annotation_object in processed_annotation["annotations"]:
            point_list_flatten = annotation_object["points"]

            if annotation_object["type"] == "bbox":
                x_min, y_min, x_max, y_max = point_list_flatten
            else:
                x_min, y_min, x_max, y_max = self._get_bbox(point_list_flatten)

            cropped_image = processed_image[y_min: y_max, x_min: x_max]

            rotation_angle = random.choice([90, 180, 270])
            cropped_image_90_degree_rotation = self._rotation_image(cropped_image, rotation_angle)

            if annotation_object["type"] == "bbox":
                if rotation_angle == 180:
                    processed_image[y_min: y_max, x_min: x_max] = cropped_image_90_degree_rotation
                else:
                    x_center, y_center = (x_max + x_min)/2, (y_max + y_min)/2
                    rotation_x_min = int(x_center - (y_max - y_min) / 2)
                    rotation_x_max = int(x_center + (y_max - y_min) / 2)
                    rotation_y_min = int(y_center - (x_max - x_min) / 2)
                    rotation_y_max = int(y_center + (x_max - x_min) / 2)
                    processed_image[rotation_y_min: rotation_y_max,rotation_x_min: rotation_x_max] = cropped_image_90_degree_rotation
                    annotation_object["points"] = [rotation_x_min, rotation_y_min, rotation_x_max, rotation_y_max]
            else:
                point_list_flatten = annotation_object["points"]
                processed_point_list_flatten = []
                for point_index in range(0, len(point_list_flatten) - 1, 2):
                    x, y = point_list_flatten[point_index], point_list_flatten[point_index + 1]

                    origin_after = origin_before = ((x_max + x_min) / 2, (y_max + y_min) / 2)

                    processed_x, processed_y = self._rotate_point(
                        origin_before, origin_after, (x, y), rotation_angle)

                    if image_height < processed_y:
                        processed_y = image_height
                    if processed_y < 0:
                        processed_y = 0

                    if image_width < processed_x:
                        processed_x = image_width
                    if processed_x < 0:
                        processed_x = 0

                    processed_point_list_flatten.append(processed_x)
                    processed_point_list_flatten.append(processed_y)

                processed_image = self._update_polygon_area(
                    processed_image, cropped_image_90_degree_rotation, processed_point_list_flatten)

                annotation_object["points"] = processed_point_list_flatten

            processed_annotation_object_list.append(annotation_object)

        processed_annotation["annotations"] = processed_annotation_object_list

        return processed_image, processed_annotation

    def _crop_bounding_box_level(self, image: np.ndarray, annotation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        processed_image = image.copy()

        processed_annotation = annotation.copy()

        processed_annotation_object_list = []
        for annotation_object in processed_annotation["annotations"]:
            point_list_flatten = annotation_object["points"]

            if annotation_object["type"] == "bbox":
                x_min, y_min, x_max, y_max = point_list_flatten
            else:
                x_min, y_min, x_max, y_max = self._get_bbox(point_list_flatten)

            cropped_image = processed_image[y_min: y_max, x_min: x_max]

            cropped_image, cropped_image_size, start_point = self._random_crop(
                cropped_image)

            processed_image[
                y_min + start_point[1]: y_min + start_point[1] + cropped_image_size[1],
                x_min + start_point[0]: x_min + start_point[0] + cropped_image_size[0]] = cropped_image

            if self._is_fill_average_color:
                fill_color = np.average(cropped_image, axis=(0, 1))
            else:
                fill_color = (0, 0, 0)

            processed_image[y_min: y_max, x_min: x_min + start_point[0]] = fill_color
            processed_image[y_min: y_max, x_min + start_point[0] + cropped_image_size[0]: x_max] = fill_color
            processed_image[y_min: y_min + start_point[1], x_min: x_max] = fill_color
            processed_image[y_min + start_point[1] + cropped_image_size[1]: y_max, x_min: x_max] = fill_color

            if annotation_object["type"] == "polygon":
                point_list_flatten = annotation_object["points"]
                processed_point_list_flatten = []
                for point_index in range(0, len(point_list_flatten) - 1, 2):
                    x, y = point_list_flatten[point_index], point_list_flatten[point_index + 1]

                    if x < x_min + start_point[0]:
                        x = x_min + start_point[0]

                    if y < y_min + start_point[1]:
                        y = y_min + start_point[1]

                    if x > x_min + start_point[0] + cropped_image_size[0]:
                        x = x_min + start_point[0] + cropped_image_size[0]

                    if y > y_min + start_point[1] + cropped_image_size[1]:
                        y = y_min + start_point[1] + cropped_image_size[1]

                    processed_point_list_flatten.append(x)
                    processed_point_list_flatten.append(y)

                annotation_object["points"] = processed_point_list_flatten

            processed_annotation_object_list.append(annotation_object)

        processed_annotation["annotations"] = processed_annotation_object_list

        return processed_image, processed_annotation

    def _rotation_bounding_box_level(self, image: np.ndarray, annotation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        processed_image = image.copy()

        processed_annotation = annotation.copy()

        processed_annotation_object_list = []
        for annotation_object in processed_annotation["annotations"]:
            point_list_flatten = annotation_object["points"]

            if annotation_object["type"] == "bbox":
                x_min, y_min, x_max, y_max = point_list_flatten
            else:
                x_min, y_min, x_max, y_max = self._get_bbox(point_list_flatten)

            cropped_image = processed_image[y_min: y_max, x_min: x_max]

            rotation_angle = random.randint(0, self._max_rotation_angle)
            cropped_rotation_image = self._rotation_image(cropped_image, rotation_angle)

            rotation_x_min = int(x_min - (cropped_rotation_image.shape[1] - cropped_image.shape[1]) / 2)
            rotation_x_max = rotation_x_min + cropped_rotation_image.shape[1]
            rotation_y_min = int(y_min - (cropped_rotation_image.shape[0] - cropped_image.shape[0]) / 2)
            rotation_y_max = rotation_y_min + cropped_rotation_image.shape[0]

            if rotation_x_min < 0:
                rotation_x_min = 0
                cropped_rotation_image = cropped_rotation_image[:, -rotation_x_min: ].copy()

            if rotation_y_min < 0:
                rotation_y_min = 0
                cropped_rotation_image = cropped_rotation_image[-rotation_y_min:, ].copy()

            if rotation_x_max > processed_image.shape[1]:
                rotation_x_max = processed_image.shape[1]
                cropped_rotation_image = cropped_rotation_image[:, :rotation_x_max - rotation_x_min].copy()

            if rotation_y_max > processed_image.shape[0]:
                rotation_y_max = processed_image.shape[0]
                cropped_rotation_image = cropped_rotation_image[:rotation_y_max - rotation_y_min, :].copy()

            processed_image[rotation_y_min: rotation_y_max, rotation_x_min: rotation_x_max] = cropped_rotation_image

            if annotation_object["type"] == "bbox":
                annotation_object["points"] = [rotation_x_min, rotation_y_min, rotation_x_max, rotation_y_max]
            else:
                point_list_flatten = annotation_object["points"]
                processed_point_list_flatten = []
                for point_index in range(0, len(point_list_flatten) - 1, 2):
                    x, y = point_list_flatten[point_index], point_list_flatten[point_index + 1]

                    origin_before = origin_after = ((x_max + x_min) / 2, (y_max + y_min) / 2)

                    x, y = self._rotate_point(origin_before, origin_after, (x, y), rotation_angle)

                    processed_point_list_flatten.append(x)
                    processed_point_list_flatten.append(y)

                annotation_object["points"] = processed_point_list_flatten

            processed_annotation_object_list.append(annotation_object)

            processed_annotation["annotations"] = processed_annotation_object_list

        return processed_image, processed_annotation

    def _shear_bounding_box_level(self, image: np.ndarray, annotation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        processed_image = image.copy()

        processed_annotation = annotation.copy()

        processed_annotation_object_list = []
        for annotation_object in processed_annotation["annotations"]:
            point_list_flatten = annotation_object["points"]

            if annotation_object["type"] == "bbox":
                x_min, y_min, x_max, y_max = point_list_flatten
            else:
                x_min, y_min, x_max, y_max = self._get_bbox(point_list_flatten)

            cropped_image = processed_image[y_min: y_max, x_min: x_max]

            cropped_shear_image, shear_matrix = self._random_shear(cropped_image)

            rotation_x_min = int(x_min - (cropped_shear_image.shape[1] - cropped_image.shape[1]) / 2)
            rotation_x_max = rotation_x_min + cropped_shear_image.shape[1]
            rotation_y_min = int(y_min - (cropped_shear_image.shape[0] - cropped_image.shape[0]) / 2)
            rotation_y_max = rotation_y_min + cropped_shear_image.shape[0]

            if rotation_x_min < 0:
                rotation_x_min = 0
                cropped_shear_image = cropped_shear_image[:, -rotation_x_min:].copy()

            if rotation_y_min < 0:
                rotation_y_min = 0
                cropped_shear_image = cropped_shear_image[-rotation_y_min:, ].copy()

            if rotation_x_max > processed_image.shape[1]:
                rotation_x_max = processed_image.shape[1]
                cropped_shear_image = cropped_shear_image[:, :rotation_x_max - rotation_x_min].copy()

            if rotation_y_max > processed_image.shape[0]:
                rotation_y_max = processed_image.shape[0]
                cropped_shear_image = cropped_shear_image[:rotation_y_max - rotation_y_min, :].copy()

            processed_image[rotation_y_min: rotation_y_max, rotation_x_min: rotation_x_max] = cropped_shear_image

            if annotation_object["type"] == "bbox":
                annotation_object["points"] = [rotation_x_min, rotation_y_min, rotation_x_max, rotation_y_max]
            else:
                point_list_flatten = annotation_object["points"]
                processed_point_list_flatten = []
                for point_index in range(0, len(point_list_flatten) - 1, 2):
                    x, y = point_list_flatten[point_index], point_list_flatten[point_index + 1]
                    origin_before = origin_after = ((x_max + x_min) / 2, (y_max + y_min) / 2)
                    x, y = self._shear_point(origin_before, origin_after, (x, y), shear_matrix)
                    processed_point_list_flatten.append(x)
                    processed_point_list_flatten.append(y)

                annotation_object["points"] = processed_point_list_flatten

            processed_annotation_object_list.append(annotation_object)

            processed_annotation["annotations"] = processed_annotation_object_list

        return processed_image, processed_annotation

    def _random_blur(self, image: np.ndarray) -> np.ndarray:
        # https://blog.roboflow.com/using-blur-in-computer-vision-preprocessing/
        kernel_size = random.randint(1, self._max_kernel_size_for_blur)
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
        return cv2.blur(image.copy(), (kernel_size, kernel_size))

    def _random_shear(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # https://qiita.com/koshian2/items/c133e2e10c261b8646bf#%E3%81%9B%E3%82%93%E6%96%ADshear

        direction = random.choice(["horizontal", "vertical"])

        shear_angle = random.uniform(self._shear_angle_range[0], self._shear_angle_range[1])
        h, w = image.shape[:2]

        if direction == "horizontal":
            mat = np.array([[1, np.radians(shear_angle), 0], [0, 1, 0]], dtype=np.float32)
            shear_image = cv2.warpAffine(image, mat, (int(w + h * np.radians(shear_angle)), h))
        else:
            mat = np.array([[1, 0, 0], [np.radians(shear_angle), 1, 0]], dtype=np.float32)
            shear_image = cv2.warpAffine(image, mat, (w, int(h + w * np.radians(shear_angle))))

        return shear_image, mat

    def _random_crop(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        # https://blog.roboflow.com/why-and-how-to-implement-random-crop-data-augmentation/
        height, width = image.shape[:2]
        start_point_x = random.randint(0, int(width * self._keep_image_ratio_for_crop))
        start_point_y = random.randint(0, int(height * self._keep_image_ratio_for_crop))

        cropped_width = random.randint(int(width * self._keep_image_ratio_for_crop), (width - start_point_x))
        cropped_height = random.randint(int(height * self._keep_image_ratio_for_crop), (height - start_point_y))

        processed_image = image[
                          start_point_y: start_point_y + cropped_height,
                          start_point_x: start_point_x + cropped_width
                          ]
        return processed_image, (cropped_width, cropped_height), (start_point_x, start_point_y)

    @staticmethod
    def _random_flip(image: np.ndarray) -> Tuple[np.ndarray, str]:
        rotation_flip_direction = random.choice(["horizontal", "vertical"])
        if rotation_flip_direction == "horizontal":
            processed_image = np.fliplr(image)
        else:
            processed_image = np.flipud(image)

        return processed_image, rotation_flip_direction

    def _random_brightness(self, image: np.ndarray) -> np.ndarray:
        # https://pystyle.info/opencv-change-contrast-and-brightness/
        alpha = random.uniform(self._alpha_range_for_brightness[0], self._alpha_range_for_brightness[1])
        beta = random.uniform(self._beta_range_for_brightness[0], self._beta_range_for_brightness[1])

        processed_image = alpha * image.copy() + beta

        processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)

        return processed_image

    def _update_polygon_area(
            self, processed_image: np.ndarray,
            cropped_image_brightness: np.ndarray, point_list_flatten: List[int]) -> np.ndarray:

        x_min, y_min, x_max, y_max = self._get_bbox(point_list_flatten)

        point_list = []
        for point_index in range(0, len(point_list_flatten) - 1, 2):
            x, y = point_list_flatten[point_index], point_list_flatten[point_index + 1]
            point_list.append([x, y])

        black_mask_image = processed_image.copy()
        cv2.fillPoly(black_mask_image, [np.array(point_list).reshape(-1, 1, 2)], (0, 0, 0))
        black_mask_image = black_mask_image[y_min: y_max, x_min: x_max]

        white_mask_image = np.zeros(processed_image.shape, np.uint8)
        cv2.fillPoly(white_mask_image, [np.array(point_list).reshape(-1, 1, 2)], [255, 255, 255])

        if y_max - y_min != cropped_image_brightness.shape[0]:
            y_max = cropped_image_brightness.shape[0] + y_min

        if x_max - x_min != cropped_image_brightness.shape[1]:
            x_max = cropped_image_brightness.shape[1] + x_min

        white_mask_image = white_mask_image[y_min: y_max, x_min: x_max]

        cropped_image_brightness = cv2.bitwise_and(
            cropped_image_brightness,
            white_mask_image
        )
        processed_image[y_min: y_max, x_min: x_max] = cropped_image_brightness + black_mask_image

        return processed_image

    def _random_exposure(self, image: np.ndarray) -> np.ndarray:
        # https://docs.roboflow.com/image-transformations/image-augmentation#exposure
        # https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_gamma
        gamma = random.uniform(self._gamma_range_for_exposure [0], self._gamma_range_for_exposure [1])
        processed_image = skimage.exposure.adjust_gamma(image.copy(), gamma=gamma)

        return processed_image

    @staticmethod
    def _random_noise(image: np.ndarray) -> np.ndarray:
        # https://blog.roboflow.com/why-to-add-noise-to-images-for-machine-learning/

        r = np.random.rand(1)

        if r < 0.15:
            processed_image = skimage.util.random_noise(image.copy(), mode='localvar')
        elif r < 0.30:
            processed_image = skimage.util.random_noise(image.copy(), mode='salt')
        elif r < 0.45:
            processed_image = skimage.util.random_noise(image.copy(), mode='s&p')
        elif r < 0.60:
            processed_image = skimage.util.random_noise(image.copy(), mode='speckle', var=0.01)
        elif r < 0.75:
            processed_image = skimage.util.random_noise(image.copy(), mode='poisson')
        else:
            processed_image = skimage.util.random_noise(image.copy(), mode='gaussian', var=0.01)

        processed_image = processed_image * 255
        processed_image = processed_image.astype(np.uint8)

        return processed_image


    @staticmethod
    def _get_bbox(point_list_flatten: List[int]) -> List[int]:
        x_list = []
        y_list = []
        for point_index in range(0, len(point_list_flatten) - 1, 2):
            x_list.append(point_list_flatten[point_index])
            y_list.append(point_list_flatten[point_index + 1])
        return [min(x_list), min(y_list), max(x_list), max(y_list)]


    def _rotation_image(self, image: np.ndarray, angle_degree: float) -> np.ndarray:

        if angle_degree == 0:
            processed_image = image.copy()
        elif angle_degree == 90:
            processed_image = np.rot90(image.copy())
        elif angle_degree == 180:
            processed_image = np.rot90(image.copy(), k=2)
        elif angle_degree == 270:
            processed_image = np.rot90(image.copy(), k=3)
        else:
            (h, w) = image.shape[:2]
            (cX, cY) = (w // 2, h // 2)

            M = cv2.getRotationMatrix2D((cX, cY), angle_degree, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY

            if self._is_fill_average_color :
                fill_color = np.average(image, axis=(0, 1))
            else:
                fill_color = (0, 0, 0)

            processed_image = cv2.warpAffine(image, M, (nW, nH), borderValue=fill_color)

        return processed_image

    @staticmethod
    def _debugger(image: np.ndarray, annotation: Dict[str, Any]) -> np.ndarray:
        image_draw = image.copy()
        label_color_dict = {}
        for annotation_object in annotation["annotations"]:
            point_list_flatten = annotation_object["points"]
            point_type = annotation_object["type"]
            label = annotation_object["title"]
            if label in label_color_dict.keys():
                color = label_color_dict[label]
            else:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                label_color_dict[label] = color

            if point_type == "bbox":
                image_draw = cv2.rectangle(
                    image_draw,
                    (point_list_flatten[0], point_list_flatten[1]),
                    (point_list_flatten[2], point_list_flatten[3]), color, 10)
            else:
                point_list = []
                for point_index in range(0, len(point_list_flatten) - 1, 2):
                    x, y = point_list_flatten[point_index], point_list_flatten[point_index + 1]
                    point_list.append([x, y])
                points = np.array(point_list, np.int32)
                points = points.reshape((-1, 1, 2))
                image_draw = cv2.polylines(image_draw, [points], True, color, 10)

        return image_draw

    @staticmethod
    def _shear_point(
            origin_before: Tuple[int, int],
            origin_after: Tuple[int, int],
            point: Tuple[int, int],
            shear_matrix: np.ndarray) -> Tuple[int, int]:
        x = shear_matrix[0][0] * (point[0] - origin_before[0]) + shear_matrix[0][1] * (point[1] - origin_before[1]) + shear_matrix[0][2] + origin_after[0]
        y = shear_matrix[1][0] * (point[0] - origin_before[0]) + shear_matrix[1][1] * (point[1] - origin_before[1]) + shear_matrix[1][2] + origin_after[1]

        return int(x), int(y)

    @staticmethod
    def _rotate_point(
            origin_before: Tuple[int, int],
            origin_after: Tuple[int, int],
            point: Tuple[int, int],
            angle_degree: float) -> Tuple[int, int]:
        ox_before, oy_before = origin_before
        ox_after, oy_after = origin_after
        px, py = point

        angle_radian = - angle_degree * math.pi / 180

        qx = ox_after + math.cos(angle_radian) * (px - ox_before) - math.sin(angle_radian) * (py - oy_before)
        qy = oy_after + math.sin(angle_radian) * (px - ox_before) + math.cos(angle_radian) * (py - oy_before)
        return int(qx), int(qy)

    @staticmethod
    def _create_output_image_dir(augmentation_dir_path: str, is_debug: bool) -> None:
        os.makedirs(f"{augmentation_dir_path}/images", exist_ok=True)
        if is_debug:
            os.makedirs(f"{augmentation_dir_path}/debugs", exist_ok=True)

    @staticmethod
    def _save_image(image: np.ndarray, image_name: str, output_dir_path: str, augmentation_name: str) -> None:
        cv2.imwrite(f"{output_dir_path}/{augmentation_name}/images/{image_name}", image)

    @staticmethod
    def _save_debug_image(image: np.ndarray, image_name: str, output_dir_path: str, augmentation_name: str) -> None:
        cv2.imwrite(f"{output_dir_path}/{augmentation_name}/debugs/{image_name}", image)

    @staticmethod
    def _save_json(annotation: List[Dict[str, Any]], output_dir_path: str, augmentation_name: str) -> None:

        augmentation_dir_path = f"{output_dir_path}/{augmentation_name}"

        os.makedirs(f"{augmentation_dir_path}", exist_ok=True)

        with codecs.open(f"{augmentation_dir_path}/annotation.json", 'w', 'utf-8') as f:
            json.dump(annotation, f, indent=4, ensure_ascii=False)