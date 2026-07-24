from collections.abc import Callable
from typing import Annotated, Any, ClassVar

from fastlabel.exceptions import FastLabelInvalidException
from fastlabel.lerobot.common import Camera

# Format-agnostic domain types. Deliberately generic (not v3-specific) so this
# module stays decoupled from any dataset version.
Meta = dict[str, Any]
Frame = dict[str, Any]
EpisodeIndex = Annotated[int, "episode_index"]
FrameNum = Annotated[int, "frame_num"]
# A name selector: given the full ordered feature names, return the ones to keep.
NameSelector = Callable[[list[str]], list[str]]


class LeRobotConverter:
    """Customization hooks for ``Client.import_lerobot``.

    Override only the pieces you need; the overall flow (episode loop, task
    creation, upload, cleanup) is owned by ``Client.import_lerobot`` and is not
    overridable. A converter is a pure strategy object: everything is resolved
    once in ``__init__`` from the dataset metadata (``meta/info.json``) and never
    mutated afterwards.

    Instances are constructed by the SDK, not the caller. Pass the class itself
    (or a ``functools.partial`` / factory) to ``import_lerobot``; the SDK calls
    it with the parsed ``info.json`` as ``meta``.

    Selection of ``observation.state`` / ``action`` values is name-based and
    follows a 3-layer structure (requires ``meta/info.json``):

    1. Declare exact names via the ``OBSERVATION_STATE_NAMES`` / ``ACTION_NAMES``
       class variables (``None`` means keep everything).
    2. Or override ``select_observation_state_names`` / ``select_action_names`` to
       pick names dynamically from the given full name list (e.g. by suffix).
    3. ``build_observation_state`` / ``build_action`` return the kept values; the
       SDK resolves the selected names to indices once in ``__init__``.
    """

    # ---- declarative selection (static configuration) ----
    OBSERVATION_STATE_NAMES: ClassVar[tuple[str, ...] | None] = None
    ACTION_NAMES: ClassVar[tuple[str, ...] | None] = None
    CAMERA_KEYS: ClassVar[tuple[str, ...] | None] = None

    def __init__(self, meta: Meta | None = None) -> None:
        self.meta: Meta = meta or {}
        # Selected names -> positions, resolved once and immutable afterwards.
        self._state_index: list[int] = self._names_to_index(
            "observation.state", self.select_observation_state_names
        )
        self._action_index: list[int] = self._names_to_index(
            "action", self.select_action_names
        )

    # ---- value selection (by name; override for dynamic selection) ----
    def select_observation_state_names(self, names: list[str]) -> list[str]:
        """Return the ``observation.state`` feature names to keep (default: all).

        ``names`` is the full ordered name list from ``meta/info.json``. Override
        for dynamic selection, e.g. ``[n for n in names if n.endswith(".pos")]``.
        """
        return (
            list(self.OBSERVATION_STATE_NAMES)
            if self.OBSERVATION_STATE_NAMES
            else names
        )

    def select_action_names(self, names: list[str]) -> list[str]:
        """Return the ``action`` feature names to keep (default: all)."""
        return list(self.ACTION_NAMES) if self.ACTION_NAMES else names

    def _names_to_index(self, key: str, select_names: NameSelector) -> list[int]:
        """Resolve the selected names to their positions in ``key``'s full name
        list (from ``self.meta``). Raises when the metadata lacks the names or a
        selected name is absent.
        """
        try:
            all_names: list[str] = self.meta["features"][key]["names"]
        except (KeyError, TypeError):
            raise FastLabelInvalidException(
                f"'{key}.names' not found in meta/info.json.", 422
            )
        position = {name: i for i, name in enumerate(all_names)}
        index: list[int] = []
        for name in select_names(all_names):
            if name not in position:
                raise FastLabelInvalidException(
                    f"'{name}' not found in '{key}.names'.", 422
                )
            index.append(position[name])
        return index

    # ---- telemetry hooks ----
    def build_observation_state(self, frame: Frame) -> list[Any]:
        values = frame["observation.state"]
        return [values[i] for i in self._state_index]

    def build_action(self, frame: Frame) -> list[Any]:
        values = frame["action"]
        return [values[i] for i in self._action_index]

    def build_telemetry_frame(self, frame: Frame) -> dict[str, Any]:
        """Build one telemetry frame written to the episode JSON.

        Override and call ``super()`` to add extra items (e.g. gripper).
        """
        return {
            "observation.state": self.build_observation_state(frame),
            "action": self.build_action(frame),
            "frame_index": int(frame["frame_index"]),
            "timestamp": float(frame["timestamp"]),
        }

    # ---- video hook ----
    def select_cameras(self, cameras: list[Camera]) -> list[Camera]:
        """Return the cameras to include (default: all — the given list).

        Each ``Camera`` has ``path`` / ``key`` / ``content_name``; ``key`` matches
        ``self.meta["features"][key]`` for looking up resolution etc. When
        ``CAMERA_KEYS`` is set, keep cameras whose key suffix (e.g. ``cam_high``
        of ``observation.images.cam_high``) is in the set.
        """
        if self.CAMERA_KEYS is None:
            return cameras
        return [
            camera
            for camera in cameras
            if camera.key.split(".")[-1] in self.CAMERA_KEYS
        ]

    # ---- task hook ----
    def build_task_kwargs(
        self,
        *,
        episode_index: int,
        episode_name: str,
        frames: list[Frame],
    ) -> dict[str, Any]:
        """Return keyword args forwarded to ``create_robotics_task``.

        Keyword-only:
        episode_index is the episode index.
        episode_name is the task name (e.g. ``episode_000001``).
        frames is the list of raw native frames for the episode.
        Dataset-level metadata is available via ``self.meta``.
        e.g. return ``{"tags": [...], "metadatas": [...]}``.
        """
        return {}

    # ---- episode hooks ----
    def select_episodes(
        self, episode_lengths: dict[EpisodeIndex, FrameNum]
    ) -> list[EpisodeIndex]:
        """Return the episode indices to import (default: all).

        Only called when ``import_lerobot`` is invoked without
        ``episode_indices``. ``episode_lengths`` maps each episode index to its
        frame count, e.g. to skip short episodes. Kept intentionally minimal
        (index -> frame count, no v3 layout details) so this class stays
        decoupled from any dataset version.
        """
        return sorted(episode_lengths)

    def build_episode_name(self, episode_index: int) -> str:
        """Identifier used for the task name, episode JSON name and ZIP name.

        Defaults to the ``episode_000001`` form. Override for custom naming
        (keep it filesystem-safe, as it is also used for artifact filenames).
        """
        return f"episode_{episode_index:06d}"
