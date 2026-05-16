import cv2
import numpy as np

from fastlabel.utils import mask_image_util


def _make_rect_mask(
    width: int, height: int, x1: int, y1: int, x2: int, y2: int
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def _make_rect_mask_with_hole(
    width: int, height: int, outer: tuple, hole: tuple
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    ox1, oy1, ox2, oy2 = outer
    hx1, hy1, hx2, hy2 = hole
    mask[oy1:oy2, ox1:ox2] = 255
    mask[hy1:hy2, hx1:hx2] = 0
    return mask


class TestMaskToPolygon:
    def test_empty_mask_returns_empty(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        assert mask_image_util.mask_to_polygon(mask) == []

    def test_single_rect_returns_flat_int_list(self):
        mask = _make_rect_mask(100, 100, 20, 30, 60, 70)
        points = mask_image_util.mask_to_polygon(mask)

        assert isinstance(points, list)
        assert len(points) >= 6
        assert len(points) % 2 == 0
        assert all(isinstance(p, int) for p in points)

        xs = points[0::2]
        ys = points[1::2]
        assert min(xs) >= 19 and max(xs) <= 60
        assert min(ys) >= 29 and max(ys) <= 70

    def test_accepts_file_path(self, tmp_path):
        mask = _make_rect_mask(80, 80, 10, 10, 50, 50)
        path = tmp_path / "mask.png"
        cv2.imwrite(str(path), mask)

        points = mask_image_util.mask_to_polygon(str(path))
        assert isinstance(points, list)
        assert len(points) >= 6


class TestMaskToSegmentation:
    def test_empty_mask_returns_empty(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        assert mask_image_util.mask_to_segmentation(mask) == []

    def test_single_rect_returns_one_polygon(self):
        mask = _make_rect_mask(100, 100, 20, 30, 60, 70)
        result = mask_image_util.mask_to_segmentation(mask)

        assert isinstance(result, list)
        assert len(result) == 1
        polygon = result[0][0]
        assert len(polygon) >= 10
        assert len(polygon) % 2 == 0
        assert all(isinstance(v, (int, np.integer)) for v in polygon)

    def test_two_separate_rects_returns_two_polygons(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        mask[60:90, 120:160] = 255

        result = mask_image_util.mask_to_segmentation(mask)
        assert len(result) == 2

    def test_rect_with_hole_includes_inner_contour(self):
        mask = _make_rect_mask_with_hole(
            100, 100, outer=(10, 10, 80, 80), hole=(30, 30, 60, 60)
        )
        result = mask_image_util.mask_to_segmentation(mask)

        assert len(result) >= 1
        assert len(result[0]) >= 2

    def test_accepts_file_path(self, tmp_path):
        mask = _make_rect_mask(80, 80, 10, 10, 50, 50)
        path = tmp_path / "mask.png"
        cv2.imwrite(str(path), mask)

        result = mask_image_util.mask_to_segmentation(str(path))
        assert len(result) == 1


class TestMaskToFastlabelSegmentationPointsAtUtils:
    def test_module_level_import(self):
        from fastlabel.utils import mask_to_segmentation

        assert mask_to_segmentation is mask_image_util.mask_to_segmentation
