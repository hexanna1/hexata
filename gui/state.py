from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from board import Move
from engine import AnalysisMove
from history_tree import MoveTree

EditSnapshot = tuple[MoveTree, tuple[Tuple[int, int], ...]]


@dataclass(slots=True)
class CandidateSelection:
    candidates: set[Tuple[int, int]] = field(default_factory=set)
    root_key: Optional[bytes] = None


class AnalysisModeTag(Enum):
    OFF = "off"
    LIVE = "live"


class BatchKind(Enum):
    RAW_NN = "raw_nn"
    TIMED = "timed"


class TransitionKind(Enum):
    USER_POSITION = "user_position"
    USER_TREE = "user_tree"
    BATCH_START = "batch_start"
    BATCH_STEP = "batch_step"


@dataclass(slots=True)
class BatchRun:
    kind: BatchKind
    first_update_at: Optional[float]
    line: tuple[Move, ...]
    expected_rev: int
    raw_pending: bool = False


AnalysisMode = AnalysisModeTag | BatchRun


@dataclass(slots=True)
class SessionState:
    pending_size: int
    tree: MoveTree = field(default_factory=MoveTree)
    candidate_selection: CandidateSelection = field(default_factory=CandidateSelection)
    analysis_cache: dict[bytes, list[AnalysisMove]] = field(default_factory=dict)
    root_eval_cache: dict[bytes, float] = field(default_factory=dict)
    last_cache_sig: Optional[tuple] = None
    analysis_wide_root_noise: float = 0.04
    analysis_mode: AnalysisMode = AnalysisModeTag.OFF
    edit_undo: list[EditSnapshot] = field(default_factory=list)
    edit_redo: list[EditSnapshot] = field(default_factory=list)

    @property
    def analysis_enabled(self) -> bool:
        return self.analysis_mode != AnalysisModeTag.OFF
