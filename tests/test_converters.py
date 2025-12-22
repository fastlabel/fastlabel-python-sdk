from fastlabel import converters


class TestToCoco:
    """Tests for to_coco converter function."""

    def test_to_coco_with_zero_height_task(self, tmp_path):
        """Test that tasks with height=0 are included in output."""
        tasks = [
            {
                "name": "image1.jpg",
                "height": 0,
                "width": 100,
                "annotations": [
                    {
                        "type": "bbox",
                        "value": "cat",
                        "points": [10, 10, 50, 50],
                        "color": "#FF0000",
                    }
                ],
            },
            {
                "name": "image2.jpg",
                "height": 100,
                "width": 100,
                "annotations": [
                    {
                        "type": "bbox",
                        "value": "cat",
                        "points": [10, 10, 50, 50],
                        "color": "#FF0000",
                    }
                ],
            },
        ]

        result = converters.to_coco(
            project_type="image_bbox",
            tasks=tasks,
            output_dir=str(tmp_path),
        )

        assert len(result["images"]) == 2
        assert result["images"][0]["height"] == 0
        assert result["images"][1]["height"] == 100

    def test_to_coco_with_zero_width_task(self, tmp_path):
        """Test that tasks with width=0 are included in output."""
        tasks = [
            {
                "name": "image1.jpg",
                "height": 100,
                "width": 0,
                "annotations": [
                    {
                        "type": "bbox",
                        "value": "dog",
                        "points": [10, 10, 50, 50],
                        "color": "#00FF00",
                    }
                ],
            },
        ]

        result = converters.to_coco(
            project_type="image_bbox",
            tasks=tasks,
            output_dir=str(tmp_path),
        )

        assert len(result["images"]) == 1
        assert result["images"][0]["width"] == 0

    def test_to_coco_with_zero_dimensions_task(self, tmp_path):
        """Test that tasks with both height=0 and width=0 are included in output."""
        tasks = [
            {
                "name": "image1.jpg",
                "height": 0,
                "width": 0,
                "annotations": [
                    {
                        "type": "bbox",
                        "value": "bird",
                        "points": [10, 10, 50, 50],
                        "color": "#0000FF",
                    }
                ],
            },
        ]

        result = converters.to_coco(
            project_type="image_bbox",
            tasks=tasks,
            output_dir=str(tmp_path),
        )

        assert len(result["images"]) == 1
        assert result["images"][0]["height"] == 0
        assert result["images"][0]["width"] == 0


class TestToPascalVoc:
    """Tests for to_pascalvoc converter function."""

    def test_to_pascalvoc_with_zero_height_task(self, tmp_path):
        """Test that tasks with height=0 are included in output."""
        tasks = [
            {
                "name": "image1.jpg",
                "height": 0,
                "width": 100,
                "annotations": [
                    {
                        "type": "bbox",
                        "value": "cat",
                        "points": [10, 10, 50, 50],
                        "attributes": [],
                    }
                ],
            },
            {
                "name": "image2.jpg",
                "height": 100,
                "width": 100,
                "annotations": [
                    {
                        "type": "bbox",
                        "value": "cat",
                        "points": [10, 10, 50, 50],
                        "attributes": [],
                    }
                ],
            },
        ]

        result = converters.to_pascalvoc(
            project_type="image_bbox",
            tasks=tasks,
            output_dir=str(tmp_path),
        )

        assert len(result) == 2
        assert result[0]["annotation"]["size"]["height"] == 0
        assert result[1]["annotation"]["size"]["height"] == 100

    def test_to_pascalvoc_with_zero_width_task(self, tmp_path):
        """Test that tasks with width=0 are included in output."""
        tasks = [
            {
                "name": "image1.jpg",
                "height": 100,
                "width": 0,
                "annotations": [
                    {
                        "type": "bbox",
                        "value": "dog",
                        "points": [10, 10, 50, 50],
                        "attributes": [],
                    }
                ],
            },
        ]

        result = converters.to_pascalvoc(
            project_type="image_bbox",
            tasks=tasks,
            output_dir=str(tmp_path),
        )

        assert len(result) == 1
        assert result[0]["annotation"]["size"]["width"] == 0

    def test_to_pascalvoc_with_zero_dimensions_task(self, tmp_path):
        """Test that tasks with both height=0 and width=0 are included in output."""
        tasks = [
            {
                "name": "image1.jpg",
                "height": 0,
                "width": 0,
                "annotations": [
                    {
                        "type": "bbox",
                        "value": "bird",
                        "points": [10, 10, 50, 50],
                        "attributes": [],
                    }
                ],
            },
        ]

        result = converters.to_pascalvoc(
            project_type="image_bbox",
            tasks=tasks,
            output_dir=str(tmp_path),
        )

        assert len(result) == 1
        assert result[0]["annotation"]["size"]["height"] == 0
        assert result[0]["annotation"]["size"]["width"] == 0
