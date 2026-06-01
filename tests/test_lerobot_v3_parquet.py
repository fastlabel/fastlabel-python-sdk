"""Tests for v3 pandas/pyarrow code paths.

Covers _build_episode_map, get_episode_indices, _convert_episode_frames, and
check_dependencies so that pandas/pyarrow major-version bumps surface
breakage in CI.
"""
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


class TestConvertEpisodeFrames:
    def test_extracts_frame_dicts(self, v3_dataset):
        frames = v3._convert_episode_frames(
            v3_dataset, episode_index=1, chunk="chunk-000", file_stem="file-000"
        )

        assert len(frames) == 3
        for i, frame in enumerate(frames):
            assert frame["frame_index"] == i
            assert frame["timestamp"] == pytest.approx(i * 0.1)
            assert frame["action"] == [1.0, 2.0]
            assert isinstance(frame["observation.state"], list)

    def test_missing_required_columns_returns_empty(self, tmp_path):
        chunk = tmp_path / "data" / "chunk-000"
        chunk.mkdir(parents=True)
        _write_parquet(
            chunk / "file-000.parquet",
            [{"episode_index": 0, "frame_index": 0}],
        )

        assert (
            v3._convert_episode_frames(
                tmp_path, episode_index=0, chunk="chunk-000", file_stem="file-000"
            )
            == []
        )


class TestCheckDependencies:
    def test_returns_when_available(self):
        common.check_dependencies()
