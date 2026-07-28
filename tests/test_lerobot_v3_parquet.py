"""Tests for v3 pandas/pyarrow code paths.

Covers _build_episode_map, get_episode_indices, get_episode_raw_frames, and
check_dependencies so that pandas/pyarrow major-version bumps surface
breakage in CI.
"""

import json

import pytest

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")

from fastlabel.lerobot import common, v3  # noqa: E402


def _write_parquet(path, rows):
    df = pd.DataFrame(rows)
    df.to_parquet(path)


@pytest.fixture
def v3_dataset(tmp_path):
    """Create a minimal v3 layout with two chunks and two episodes per file."""
    data_dir = tmp_path / "data"
    chunk0 = data_dir / "chunk-000"
    chunk0.mkdir(parents=True)

    rows = [
        {
            "episode_index": ep,
            "frame_index": f,
            "timestamp": float(f) * 0.1,
            "observation.state": [0.1 * f, 0.2 * f],
            "action": [1.0, 2.0],
        }
        for ep in (0, 1)
        for f in range(3)
    ]
    _write_parquet(chunk0 / "file-000.parquet", rows)

    chunk1 = data_dir / "chunk-001"
    chunk1.mkdir(parents=True)
    rows = [
        {
            "episode_index": 2,
            "frame_index": f,
            "timestamp": float(f) * 0.1,
            "observation.state": [0.0, 0.0],
            "action": [0.0, 0.0],
        }
        for f in range(2)
    ]
    _write_parquet(chunk1 / "file-000.parquet", rows)

    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.state": {"names": ["s0", "s1"]},
                    "action": {"names": ["a0", "a1"]},
                }
            }
        )
    )

    return tmp_path


class TestBuildEpisodeMap:
    def test_returns_offsets_per_episode(self, v3_dataset):
        result = v3._build_episode_map(v3_dataset)

        assert set(result.keys()) == {0, 1, 2}
        assert result[0] == {
            "chunk": "chunk-000",
            "file_stem": "file-000",
            "frame_offset": 0,
            "length": 3,
        }
        assert result[1] == {
            "chunk": "chunk-000",
            "file_stem": "file-000",
            "frame_offset": 3,
            "length": 3,
        }
        assert result[2] == {
            "chunk": "chunk-001",
            "file_stem": "file-000",
            "frame_offset": 0,
            "length": 2,
        }

    def test_get_episode_indices_sorted(self, v3_dataset):
        assert v3.get_episode_indices(v3_dataset) == [0, 1, 2]


class TestGetEpisodeRawFrames:
    def test_extracts_all_columns_as_native(self, v3_dataset):
        frames = v3.get_episode_raw_frames(
            v3_dataset, episode_index=1, chunk="chunk-000", file_stem="file-000"
        )

        assert len(frames) == 3
        for i, frame in enumerate(frames):
            assert frame["episode_index"] == 1
            assert frame["frame_index"] == i
            assert frame["timestamp"] == pytest.approx(i * 0.1)
            assert frame["action"] == [1.0, 2.0]
            assert isinstance(frame["observation.state"], list)


class TestCheckDependencies:
    def test_returns_when_available(self):
        common.check_dependencies()


class TestCreateEpisodeZip:
    def test_writes_zip_into_output_dir_without_leaking_staging(
        self, v3_dataset, tmp_path
    ):
        import zipfile
        from pathlib import Path

        from fastlabel.lerobot import LeRobotConverter, create_episode_zip

        out = tmp_path / "out"
        out.mkdir()
        episode_map = v3._build_episode_map(v3_dataset)
        raw_frames = v3.get_episode_raw_frames(
            v3_dataset, 1, episode_map[1]["chunk"], episode_map[1]["file_stem"]
        )
        zip_path = create_episode_zip(
            v3_dataset,
            episode_index=1,
            episode_name="episode_000001",
            converter=LeRobotConverter(common.load_info(v3_dataset)),
            output_dir=out,
            episode_map=episode_map,
            raw_frames=raw_frames,
        )

        # ZIP is written under output_dir; the staging dir is not leaked there.
        assert Path(zip_path).parent == out
        assert [p.name for p in out.iterdir()] == ["episode_000001.zip"]

        # No videos in this fixture, so the ZIP holds only the telemetry JSON.
        with zipfile.ZipFile(zip_path) as zf:
            assert zf.namelist() == ["episode_000001.json"]
