import json
import os
import cv2
import random
import numpy as np
import math
import skimage
import copy


class Augmentation:

    AUGMENTATION_LIST = [
        "90_degree_rotation_image_level",
        "crop_image_level",
        "rotation_image_level",
        "brightness_image_level",
        "blur_image_level",
        "noise_image_level",
        "exposure_image_level",
        "brightness_bounding_box_level",
        "exposure_bounding_box_level",
        "blur_bounding_box_level",
        "noise_bounding_box_level"
    ]

    @classmethod
    def execute(
            cls,
            image_dir_path: str,
            annotation_json_path: str,
            output_dir_path: str = "./augmentation",
            is_debug: bool = False) -> None:

        annotation_list = json.load(open(annotation_json_path))

        for augmentation_name in cls.AUGMENTATION_LIST:
            augmentation_method = getattr(cls, f"_{augmentation_name}")

            augmentation_dir_path = f"{output_dir_path}/{augmentation_name}"
            os.makedirs(f"{augmentation_dir_path}/images", exist_ok=True)
            if is_debug:
                os.makedirs(f"{augmentation_dir_path}/debugs", exist_ok=True)

            processed_annotation_list = []
            for annotation in copy.deepcopy(annotation_list):
                image_name = annotation["name"]
                image_path = f"{image_dir_path}/{image_name}"

                if not os.path.isfile(image_path):
                    continue

                image = cv2.imread(image_path)

                processed_image, processed_annotation = augmentation_method(
                    cls, image, annotation)

                if is_debug:
                    image_draw = cls._debugger(processed_image, processed_annotation)
                    cls._save_debug_image(image_draw, image_name, output_dir_path, augmentation_name)

                cls._save_image(
                    processed_image, image_name, output_dir_path, augmentation_name)

                processed_annotation_list.append(processed_annotation)

            cls._save_json(processed_annotation_list, output_dir_path, augmentation_name)

    def _90_degree_rotation_image_level(self, image, annotation):
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

    def _rotation_image_level(self, image, annotation):
        rotation_angle = random.randint(0, 359)
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

    def _crop_image_level(self, image, annotation):
        # https://blog.roboflow.com/why-and-how-to-implement-random-crop-data-augmentation/

        keep_image_ratio = 0.5

        width, height = annotation["width"], annotation["height"]
        start_point_x = random.randint(0, int(width * keep_image_ratio))
        start_point_y = random.randint(0, int(height * keep_image_ratio))

        cropped_width = random.randint(int(width * keep_image_ratio), (width - start_point_x))
        cropped_height = random.randint(int(height * keep_image_ratio), (height - start_point_y))

        processed_image = image[
                          start_point_y: start_point_y + cropped_height,
                          start_point_x: start_point_x + cropped_width
                          ]

        processed_annotation = annotation.copy()

        processed_annotation["height"], processed_annotation["width"] = cropped_height, cropped_width

        processed_annotation_object_list = []
        for annotation_object in processed_annotation["annotations"]:
            point_list_flatten = annotation_object["points"]

            processed_point_list_flatten = []
            is_out_of_frame_x = is_out_of_frame_y = True
            for point_index in range(0, len(point_list_flatten) - 1, 2):
                x, y = point_list_flatten[point_index], point_list_flatten[point_index + 1]

                processed_x = x - start_point_x
                processed_y = y - start_point_y

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

    def _brightness_image_level(self, image, annotation):
        return self._random_brightness(image.copy()), annotation

    def _blur_image_level(self, image, annotation):
        return self._random_blur(image.copy()), annotation

    def _noise_image_level(self, image, annotation):
        return self._random_noise(image.copy()), annotation

    def _exposure_image_level(self, image, annotation):
        return self._random_exposure(image.copy()), annotation

    def _brightness_bounding_box_level(self, image, annotation):
        processed_image = image.copy()

        for annotation_object in annotation["annotations"]:
            point_list_flatten = annotation_object["points"]

            if annotation_object["type"] == "bbox":
                x_min, y_min, x_max, y_max = point_list_flatten
            else:
                x_min, y_min, x_max, y_max = self._get_bbox(point_list_flatten)

            cropped_image = processed_image[y_min: y_max, x_min: x_max]

            cropped_image_brightness = self._random_brightness(cropped_image)

            processed_image[y_min: y_max, x_min: x_max] = cropped_image_brightness

        return processed_image, annotation

    def _exposure_bounding_box_level(self, image, annotation):
        processed_image = image.copy()

        for annotation_object in annotation["annotations"]:
            point_list_flatten = annotation_object["points"]

            if annotation_object["type"] == "bbox":
                x_min, y_min, x_max, y_max = point_list_flatten
            else:
                x_min, y_min, x_max, y_max = self._get_bbox(point_list_flatten)

            cropped_image = processed_image[y_min: y_max, x_min: x_max]

            cropped_image_brightness = self._random_exposure(cropped_image)

            processed_image[y_min: y_max, x_min: x_max] = cropped_image_brightness

        return processed_image, annotation

    def _blur_bounding_box_level(self, image, annotation):
        processed_image = image.copy()

        for annotation_object in annotation["annotations"]:
            point_list_flatten = annotation_object["points"]

            if annotation_object["type"] == "bbox":
                x_min, y_min, x_max, y_max = point_list_flatten
            else:
                x_min, y_min, x_max, y_max = self._get_bbox(point_list_flatten)

            cropped_image = processed_image[y_min: y_max, x_min: x_max]

            cropped_image_brightness = self._random_blur(cropped_image)

            processed_image[y_min: y_max, x_min: x_max] = cropped_image_brightness

        return processed_image, annotation

    def _noise_bounding_box_level(self, image, annotation):
        processed_image = image.copy()

        for annotation_object in annotation["annotations"]:
            point_list_flatten = annotation_object["points"]

            if annotation_object["type"] == "bbox":
                x_min, y_min, x_max, y_max = point_list_flatten
            else:
                x_min, y_min, x_max, y_max = self._get_bbox(point_list_flatten)

            cropped_image = processed_image[y_min: y_max, x_min: x_max]

            cropped_image_brightness = self._random_noise(cropped_image)

            processed_image[y_min: y_max, x_min: x_max] = cropped_image_brightness

        return processed_image, annotation

    @staticmethod
    def _random_blur(image):
        # https://blog.roboflow.com/using-blur-in-computer-vision-preprocessing/
        kernel_size = random.randint(0, 12) * 2 + 1
        return cv2.blur(image.copy(), (kernel_size, kernel_size))

    @staticmethod
    def _random_brightness(image):
        # https://pystyle.info/opencv-change-contrast-and-brightness/

        alpha = random.uniform(0.2, 2)
        beta = random.uniform(-100, 100)

        processed_image = alpha * image.copy() + beta

        processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)

        return processed_image

    @staticmethod
    def _random_exposure(image):
        # https://docs.roboflow.com/image-transformations/image-augmentation#exposure
        # https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_gamma

        gamma = random.uniform(0, 1)

        processed_image = skimage.exposure.adjust_gamma(image.copy(), gamma=gamma)

        return processed_image

    @staticmethod
    def _random_noise(image):
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
    def _get_bbox(point_list_flatten):
        x_list = []
        y_list = []
        for point_index in range(0, len(point_list_flatten) - 1, 2):
            x_list.append(point_list_flatten[point_index])
            y_list.append(point_list_flatten[point_index + 1])
        return min(x_list), min(y_list), max(x_list), max(y_list)


    @staticmethod
    def _rotation_image(image, angle_degree):

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

            processed_image = cv2.warpAffine(image, M, (nW, nH))

        return processed_image

    @staticmethod
    def _debugger(image, annotation):
        image_draw = image.copy()
        draw_line_thinness = 10
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
                    (point_list_flatten[2], point_list_flatten[3]), color, draw_line_thinness)
            else:
                point_list = []
                for point_index in range(0, len(point_list_flatten) - 1, 2):
                    x, y = point_list_flatten[point_index], point_list_flatten[point_index + 1]
                    point_list.append([x, y])
                points = np.array(point_list, np.int32)
                points = points.reshape((-1, 1, 2))
                image_draw = cv2.polylines(image_draw, [points], True, color, draw_line_thinness)

        return image_draw

    @staticmethod
    def _save_image(image, image_name, output_dir_path, augmentation_name):
        cv2.imwrite(f"{output_dir_path}/{augmentation_name}/images/{image_name}", image)

    @staticmethod
    def _save_debug_image(image, image_name, output_dir_path, augmentation_name):
        cv2.imwrite(f"{output_dir_path}/{augmentation_name}/debugs/{image_name}", image)

    @staticmethod
    def _save_json(annotation, output_dir_path, augmentation_name):

        augmentation_dir_path = f"{output_dir_path}/{augmentation_name}"

        os.makedirs(f"{augmentation_dir_path}", exist_ok=True)

        with open(f"{augmentation_dir_path}/annotation.json", 'w') as f:
            json.dump(annotation, f, indent=4)

    @staticmethod
    def _rotate_point(origin_before, origin_after, point, angle_degree):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
        :param angle: <float> Angle in degrees.
            Positive angle is counterclockwise.
        """
        ox_before, oy_before = origin_before
        ox_after, oy_after = origin_after
        px, py = point

        angle_radian = - angle_degree * math.pi / 180

        qx = ox_after + math.cos(angle_radian) * (px - ox_before) - math.sin(angle_radian) * (py - oy_before)
        qy = oy_after + math.sin(angle_radian) * (px - ox_before) + math.cos(angle_radian) * (py - oy_before)
        return int(qx), int(qy)