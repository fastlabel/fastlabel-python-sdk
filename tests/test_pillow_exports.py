"""Smoke tests for Pillow-using code paths.

These exercise the Image.open/new/fromarray/composite, convert, putpalette,
save, ImageDraw, and ImageColor calls inside fastlabel/__init__.py so that
Pillow major-version bumps surface API breakage in CI.
"""
import os

import numpy as np
import pytest
from PIL import Image

import fastlabel
from fastlabel import const


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("FASTLABEL_ACCESS_TOKEN", "dummy-token")
    return fastlabel.Client()


def _bbox_task(name="task1.png", w=64, h=48):
    return {
        "name": name,
        "width": w,
        "height": h,
        "annotations": [
            {
                "type": "bbox",
                "value": "cat",
                "color": "#ff0000",
                "points": [10, 10, 40, 30],
            }
        ],
    }


def _polygon_task(name="task2.png", w=64, h=48):
    return {
        "name": name,
        "width": w,
        "height": h,
        "annotations": [
            {
                "type": "polygon",
                "value": "dog",
                "color": "#00ff00",
                "points": [5, 5, 50, 5, 50, 40, 5, 40],
            }
        ],
    }


def _segmentation_task(name="task3.png", w=64, h=48):
    return {
        "name": name,
        "width": w,
        "height": h,
        "annotations": [
            {
                "type": "segmentation",
                "value": "bird",
                "color": "#0000ff",
                "points": [[[5, 5, 50, 5, 50, 40, 5, 40]]],
            }
        ],
    }


class TestExportIndexColorImage:
    """Covers Image.new, Image.fromarray, convert('P'), putpalette, save."""

    def _call(self, client, task, output_dir, **kwargs):
        client._Client__export_index_color_image(
            task=task,
            output_dir=str(output_dir),
            pallete=const.COLOR_PALETTE,
            **kwargs,
        )

    def _assert_indexed_png(self, path):
        assert os.path.exists(path)
        with Image.open(path) as img:
            assert img.mode == "P"
            assert img.getpalette() is not None
            assert img.size == (64, 48)

    def test_bbox_instance(self, client, tmp_path):
        task = _bbox_task()
        self._call(client, task, tmp_path, is_instance_segmentation=True)
        self._assert_indexed_png(tmp_path / "task1.png")

    def test_polygon_semantic(self, client, tmp_path):
        task = _polygon_task()
        self._call(
            client,
            task,
            tmp_path,
            is_instance_segmentation=False,
            classes=["dog"],
        )
        self._assert_indexed_png(tmp_path / "task2.png")

    def test_segmentation_instance(self, client, tmp_path):
        task = _segmentation_task()
        self._call(client, task, tmp_path, is_instance_segmentation=True)
        self._assert_indexed_png(tmp_path / "task3.png")


class TestCreateImageWithAnnotation:
    """Covers Image.open, ImageDraw.Draw, ImageColor.getcolor, Image.composite."""

    def _make_source_image(self, path, w=64, h=48):
        arr = np.full((h, w, 3), 200, dtype=np.uint8)
        Image.fromarray(arr).save(path)

    def _call(self, client, img_path, task, output_dir):
        client._Client__create_image_with_annotation(
            [str(img_path), task, str(output_dir)]
        )

    def test_bbox(self, client, tmp_path):
        src = tmp_path / "src.png"
        self._make_source_image(src)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        task = _bbox_task(name="src.png")
        self._call(client, src, task, out_dir)
        result = out_dir / "src.png"
        assert result.exists()
        with Image.open(result) as img:
            assert img.size == (64, 48)

    def test_polygon(self, client, tmp_path):
        src = tmp_path / "p.png"
        self._make_source_image(src)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        task = _polygon_task(name="p.png")
        self._call(client, src, task, out_dir)
        assert (out_dir / "p.png").exists()

    def test_segmentation_triggers_composite(self, client, tmp_path):
        src = tmp_path / "s.png"
        self._make_source_image(src)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        task = _segmentation_task(name="s.png")
        self._call(client, src, task, out_dir)
        result = out_dir / "s.png"
        assert result.exists()
        with Image.open(result) as img:
            assert img.mode in ("RGB", "RGBA")

    def test_segmentation_jpeg_converts_rgb(self, client, tmp_path):
        src = tmp_path / "s.jpg"
        arr = np.full((48, 64, 3), 200, dtype=np.uint8)
        Image.fromarray(arr).save(src, format="JPEG")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        task = _segmentation_task(name="s.jpg")
        self._call(client, src, task, out_dir)
        result = out_dir / "s.jpg"
        assert result.exists()
        with Image.open(result) as img:
            assert img.mode == "RGB"
