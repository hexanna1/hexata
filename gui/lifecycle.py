from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Optional, Sequence, Tuple

from board import Side
from engine import KataHexEngine, RawNNResult
from gui.state import AnalysisModeTag, BatchKind, BatchRun

if TYPE_CHECKING:
    from gui.core import GuiCore


class EngineLifecycle:
    """Own engine effects; callers install logical state before reconciliation.

    Position changes reconcile immediately on leaving the scope. Callers finish
    installing the tree, board, and selection before any engine commands run.
    """

    def __init__(self, core: GuiCore) -> None:
        self.core = core

    @staticmethod
    def _pause(engine: KataHexEngine) -> None:
        engine.stop_analysis()
        engine.clear_analysis()

    def set_enabled(self, enabled: bool) -> None:
        state = self.core.session.analysis
        if enabled == state.enabled:
            return
        state.mode = AnalysisModeTag.LIVE if enabled else AnalysisModeTag.OFF
        if enabled:
            self.restart()
        else:
            self._pause(self.core.engine)

    def resume_live(self) -> None:
        state = self.core.session.analysis
        if not isinstance(state.mode, BatchRun):
            return
        state.mode = AnalysisModeTag.LIVE
        self.core.engine.cancel_reply_capture()
        self.restart()

    def restart(self) -> None:
        core = self.core
        state = core.session.analysis
        mode = state.mode
        if mode == AnalysisModeTag.OFF:
            return
        candidates: Sequence[Tuple[int, int]] = sorted(state.candidate_selection.candidates)
        if isinstance(mode, BatchRun):
            core.engine.cancel_reply_capture()
            core.engine.clear_analysis()
            if mode.kind == BatchKind.RAW_NN:
                mode.raw_pending = False
                return
            mode.first_update_at = None
            candidates = ()
        else:
            if candidates:
                core._ensure_candidate_root()
                core.maybe_update_analysis_cache()
            core.engine.clear_analysis()
        side = core.map_side_to_engine(core.current_side())
        moves = [core.map_coords_to_engine(*cell) for cell in candidates]
        core.engine.kata_set_param("analysisWideRootNoise", state.wide_root_noise)
        allow_filters = ((side, moves),) if moves else ()
        core.engine.start_analysis(side, core.analyze_interval_cs, allow_filters)

    def candidates_changed(self) -> None:
        if self.core.is_batch_analysis_active():
            self.resume_live()
        else:
            self.restart()

    def clear_caches(self) -> None:
        core = self.core
        core.engine.clear_analysis()
        core.engine.clear_cache()
        core.clear_all_cached_analysis()
        self.restart()

    def poll_raw_nn(self, run: BatchRun) -> Tuple[bool, Optional[RawNNResult]]:
        if not run.raw_pending:
            run.raw_pending = self.core.engine.start_kata_raw_nn(0)
            return False, None
        done, raw = self.core.engine.poll_kata_raw_nn()
        if done:
            run.raw_pending = False
        return done, raw

    def rebuild_position(self) -> None:
        self.core.engine.clear_board()
        for side, col, row in self.core._engine_position_moves():
            self.core.engine.play(side, col, row)

    def _sync_position(
        self, old_moves: Sequence[Tuple[Side, Optional[int], Optional[int]]]
    ) -> None:
        core = self.core
        new_moves = core._engine_position_moves()
        common = 0
        while common < min(len(old_moves), len(new_moves)) and old_moves[common] == new_moves[common]:
            common += 1
        for _ in old_moves[common:]:
            core.engine.undo()
        for side, col, row in new_moves[common:]:
            core.engine.play(side, col, row)

    @contextmanager
    def position_change(self, *, batch: bool = False) -> Iterator[None]:
        """User changes exit batch mode; batch's own changes preserve its run."""
        core = self.core
        state = core.session.analysis
        old_engine = core.engine
        old_size = core.board.n
        old_path = tuple(core.current_path_moves())
        old_moves = core._engine_position_moves()
        old_candidates = frozenset(state.candidate_selection.candidates)
        old_mode = state.mode
        was_running = state.enabled
        yield

        engine_changed = core.engine is not old_engine
        reset = engine_changed or core.board.n != old_size
        path_changed = tuple(core.current_path_moves()) != old_path
        candidates_changed = state.candidate_selection.candidates != old_candidates
        if not batch and isinstance(state.mode, BatchRun):
            state.mode = AnalysisModeTag.LIVE
        mode_changed = state.mode is not old_mode
        position_changed = reset or path_changed
        analysis_changed = position_changed or candidates_changed
        if was_running and analysis_changed:
            self._pause(old_engine)
        if reset:
            if engine_changed:
                old_engine.close()
            else:
                core.engine.set_board_size(core.board.n)
                self.rebuild_position()
                core.engine.clear_analysis()
        elif path_changed:
            self._sync_position(old_moves)
            core.engine.clear_analysis()
        if position_changed:
            core.check_candidate_root()
        core._ensure_candidate_root()

        mode = state.mode
        if isinstance(mode, BatchRun):
            mode.expected_rev = core.board.rev
        if mode_changed and isinstance(old_mode, BatchRun) and not analysis_changed:
            # An unchanged position needs only the new analyze command, which
            # stops the previous search itself. Cancel a pending raw reply too.
            core.engine.cancel_reply_capture()
        # Raw batch steps request their next NN result on the following tick.
        raw_step = (
            batch
            and mode is old_mode
            and isinstance(mode, BatchRun)
            and mode.kind == BatchKind.RAW_NN
        )
        if not raw_step and (analysis_changed or mode_changed):
            self.restart()

    def replace_engine(self, new_engine: KataHexEngine) -> bool:
        core = self.core
        if new_engine is core.engine:
            return False
        if new_engine.game_type != core.board.game_type:
            new_engine.close()
            return False
        was_running = core.session.analysis.enabled
        old_engine = core.engine
        if was_running:
            self._pause(old_engine)
        core.engine = new_engine
        try:
            self.rebuild_position()
        except Exception:
            core.engine = old_engine
            new_engine.close()
            if was_running:
                self.restart()
            return False
        core.clear_all_cached_analysis()
        old_engine.close()
        if isinstance(core.session.analysis.mode, BatchRun):
            core.session.analysis.mode = AnalysisModeTag.LIVE
        self.restart()
        return True
