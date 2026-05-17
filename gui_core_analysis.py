from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from board import HexBoard, Move, MoveKind, Side, coord_to_human
from engine import AnalysisMove

SLOW_BATCH_SECONDS_PER_POS = 3.0


@dataclass(slots=True)
class CandidateRun:
    key: Tuple[int, int]
    target_visits: float
    first_update_at: Optional[float]


@dataclass(slots=True)
class BatchRun:
    kind: "BatchKind"
    first_update_at: Optional[float]
    line: tuple[Move, ...]
    expected_rev: int
    raw_pending: bool = False


@dataclass(slots=True)
class CandidateState:
    candidates: set[Tuple[int, int]]
    results: dict[Tuple[int, int], Tuple[Optional[float], Optional[int]]]
    run: Optional[CandidateRun]
    ratio: float
    root_key: Optional[bytes]


class AnalysisModeTag(Enum):
    OFF = "off"
    LIVE = "live"


class BatchKind(Enum):
    RAW_NN = "raw_nn"
    TIMED = "timed"


AnalysisMode = AnalysisModeTag | BatchRun


@dataclass(slots=True)
class AppState:
    pending_size: int
    candidate_state: CandidateState
    analysis_cache: dict[bytes, list[AnalysisMove]]
    root_eval_cache: dict[bytes, float]
    analysis_cache_generation: int
    last_cache_sig: Optional[tuple]
    analysis_wide_root_noise: float
    analysis_mode: AnalysisMode = AnalysisModeTag.OFF

    @property
    def analysis_enabled(self) -> bool:
        return self.analysis_mode != AnalysisModeTag.OFF


class GuiCoreAnalysisMixin:
    # -------------------- mode and engine mapping --------------------
    def is_batch_analysis_active(self) -> bool:
        return isinstance(self.app.analysis_mode, BatchRun)

    def is_candidate_mode(self) -> bool:
        return self.app.analysis_enabled and not self.is_batch_analysis_active() and bool(
            self.app.candidate_state.candidates
        )

    def current_side(self) -> Side:
        moves = self.current_path_moves()
        if not moves:
            return Side.RED
        last = moves[-1]
        return self.flip_side(last.side)

    def flip_side(self, side: Side) -> Side:
        return Side.BLUE if side == Side.RED else Side.RED

    def swap_active(self) -> bool:
        moves = self.current_path_moves()
        return len(moves) >= 2 and moves[1].kind == MoveKind.SWAP

    def _map_side_to_engine(self, side: Side) -> Side:
        if self.swap_active():
            return self.flip_side(side)
        return side

    def _map_coords_to_engine(self, col: int, row: int) -> Tuple[int, int]:
        if self.swap_active():
            return (row, col)
        return (col, row)

    def _map_coords_from_engine(self, col: int, row: int) -> Tuple[int, int]:
        if self.swap_active():
            return (row, col)
        return (col, row)

    def play_engine_mapped(self, side: Side, col: Optional[int], row: Optional[int]) -> None:
        mapped_side = self._map_side_to_engine(side)
        if col is None or row is None:
            self.engine.play(mapped_side, None, None)
            return
        mapped_col, mapped_row = self._map_coords_to_engine(col, row)
        self.engine.play(mapped_side, mapped_col, mapped_row)

    def get_engine_analysis(self) -> List[AnalysisMove]:
        recs = self.engine.get_analysis()
        if not self.swap_active():
            return recs

        out: List[AnalysisMove] = []
        for r in recs:
            col, row = r.col, r.row
            move = r.move
            if col is not None and row is not None:
                col, row = self._map_coords_from_engine(col, row)
                move = coord_to_human(col, row)

            pv = r.pv
            if pv is not None:
                pv = tuple(self._map_coords_from_engine(c, rr) for c, rr in pv)

            out.append(
                AnalysisMove(
                    move=move,
                    order=r.order,
                    col=col,
                    row=row,
                    winrate=r.winrate,
                    visits=r.visits,
                    prior=r.prior,
                    pv=pv,
                )
            )
        return out

    # -------------------- analysis cache --------------------
    @staticmethod
    def _visits(v: Optional[int]) -> int:
        return 0 if v is None else v

    def _materialized_history_for_moves(self, moves: Sequence[Move]) -> tuple[Move, ...]:
        probe = HexBoard(self.board.n)
        for mv in moves:
            if not probe.apply_move(mv):
                raise AssertionError(f"Illegal move sequence for cache key: {moves!r}")
        return tuple(probe.history)

    @staticmethod
    def _cache_move_token(mv: Move) -> tuple[int, int, int, int]:
        kind = 0
        if mv.kind == MoveKind.PASS:
            kind = 1
        elif mv.kind == MoveKind.SWAP:
            kind = 2
        return (kind, int(mv.side), 0 if mv.col is None else mv.col, 0 if mv.row is None else mv.row)

    def _hash_applied_history(self, moves: Sequence[Move]) -> bytes:
        side = Side.RED if not moves else self.flip_side(moves[-1].side)
        h = hashlib.blake2b(digest_size=16)
        h.update(struct.pack("<IB", self.board.n, int(side)))
        for mv in moves:
            kind, mv_side, col, row = self._cache_move_token(mv)
            h.update(struct.pack("<BBHH", kind, mv_side, col, row))
        return h.digest()

    def cache_key_for_moves(self, moves: Sequence[Move]) -> bytes:
        applied = self._materialized_history_for_moves(moves)
        return self._hash_applied_history(applied)

    def cache_key_for_applied_moves(self, moves: Sequence[Move]) -> bytes:
        return self._hash_applied_history(moves)

    def cache_key(self) -> bytes:
        return self._hash_applied_history(self.applied_history())

    def cache_reset_sig(self) -> None:
        self.app.last_cache_sig = None

    def clear_all_cached_analysis(self) -> None:
        self.app.analysis_cache.clear()
        self.app.root_eval_cache.clear()
        self.app.analysis_cache_generation += 1
        self.cache_reset_sig()

    def clear_analysis_caches(self) -> None:
        was_running = self.app.analysis_enabled
        state = self.app.candidate_state
        had_candidates = bool(state.candidates)
        self._clear_candidate_progress(reset_results=True)

        self.engine.clear_analysis()
        self.engine.clear_cache()
        self.clear_all_cached_analysis()

        if was_running:
            if had_candidates:
                self.start_candidate_search(reset_results=True)
            else:
                self.resume_analysis()

    def _merge_analysis(self, primary: AnalysisMove, secondary: Optional[AnalysisMove]) -> AnalysisMove:
        """Primary supplies order/prior; deeper visits supplies winrate/visits."""
        if secondary is None:
            return primary
        best = primary if self._visits(primary.visits) >= self._visits(secondary.visits) else secondary
        order = primary.order if primary.order is not None else secondary.order
        prior = primary.prior if primary.prior is not None else secondary.prior
        pv = primary.pv if primary.pv is not None else secondary.pv
        return AnalysisMove(
            move=primary.move,
            order=order,
            col=primary.col,
            row=primary.row,
            winrate=best.winrate,
            visits=best.visits,
            prior=prior,
            pv=pv,
        )

    def _merge_analysis_lists(
        self, primary: List[AnalysisMove], secondary: List[AnalysisMove]
    ) -> List[AnalysisMove]:
        def row_key(r: AnalysisMove) -> tuple[str, int | str, int | None]:
            if r.col is not None and r.row is not None:
                return ("coord", r.col, r.row)
            return ("move", r.move, None)

        secondary_by_key: dict[tuple[str, int | str, int | None], AnalysisMove] = {}
        for r in secondary:
            secondary_by_key[row_key(r)] = r

        merged: List[AnalysisMove] = []
        for r in primary:
            merged.append(self._merge_analysis(r, secondary_by_key.pop(row_key(r), None)))

        merged.extend(secondary_by_key.values())
        return merged

    def _merge_live_into_cache(self, live: List[AnalysisMove]) -> None:
        cache_key = self.cache_key()
        existing = self.app.analysis_cache.get(cache_key)
        if existing is None:
            self.app.analysis_cache[cache_key] = list(live)
            self.cache_reset_sig()
            return

        # Primary = live; cache only upgrades winrate/visits.
        merged = self._merge_analysis_lists(live, existing)
        self.app.analysis_cache[cache_key] = merged
        self.cache_reset_sig()

    def _promote_candidate_to_cache(self, key: Tuple[int, int], winrate: float, visits: int) -> None:
        cache_key = self.cache_key()
        base = self.app.analysis_cache.get(cache_key, [])
        incoming = AnalysisMove(
            move=coord_to_human(*key),
            order=None,
            col=key[0],
            row=key[1],
            winrate=winrate,
            visits=visits,
            prior=None,
            pv=None,
        )
        merged = self._merge_analysis_lists(base, [incoming])
        if merged == base:
            return
        self.app.analysis_cache[cache_key] = merged
        self.cache_reset_sig()

    def _cache_root_eval(self, blue_win: float) -> None:
        side_to_play = self.current_side()
        synthetic_wr = (1.0 - blue_win) if side_to_play == Side.RED else blue_win
        cache_key = self.cache_key()
        if self.app.root_eval_cache.get(cache_key) == synthetic_wr:
            return
        self.app.root_eval_cache[cache_key] = synthetic_wr

    def maybe_update_analysis_cache(self) -> None:
        if not self.app.analysis_enabled or self.is_candidate_mode():
            return
        live = self.get_engine_analysis()
        if not live:
            return

        key = self.cache_key()
        sig = (
            key,
            tuple(
                (r.move, r.order, r.col, r.row, r.visits, r.winrate, r.prior, r.pv)
                for r in live
            ),
        )
        if sig == self.app.last_cache_sig:
            return

        self._merge_live_into_cache(live)
        self.app.last_cache_sig = sig

    # -------------------- analysis loop and controls --------------------
    def tick(self, now: float) -> None:
        if isinstance(self.app.analysis_mode, BatchRun):
            self.step_batch_analysis(now)
            self.maybe_update_analysis_cache()
            return
        if self.check_candidate_root():
            self.resume_analysis()
        self.step_candidate_search(now)
        self.maybe_update_analysis_cache()

    def toggle_analysis(self) -> None:
        if self.app.analysis_enabled:
            self._apply_analysis_enabled_transition(False)
        else:
            self._apply_analysis_enabled_transition(True)

    def resume_analysis(self) -> None:
        if self.app.analysis_mode == AnalysisModeTag.OFF:
            return
        self._refresh_analysis()

    def stop_analysis(self) -> None:
        self.stop_candidate_run()
        self.engine.stop_analysis()

    def set_analysis_wide_root_noise(self, value: float) -> None:
        value = max(0.0, min(2.0, float(value)))
        if abs(self.app.analysis_wide_root_noise - value) < 1e-9:
            return
        self.app.analysis_wide_root_noise = value
        if self.app.analysis_enabled and not self.app.candidate_state.candidates:
            self.resume_analysis()

    # -------------------- analysis queries --------------------
    def get_active_analysis(self) -> List[AnalysisMove]:
        # Prefer the most informative analysis (cache/live) while candidates can
        # upgrade cached winrate/visits.
        base = self.app.analysis_cache.get(self.cache_key(), [])
        if base:
            return base
        if self.app.candidate_state.candidates:
            return self.get_candidate_analysis()
        if self.app.analysis_enabled:
            live = self.get_engine_analysis()
            if live:
                return live
        return []

    def get_top_move(self) -> Tuple[Optional[Tuple[int, int]], int]:
        best: Optional[AnalysisMove] = None
        recs = self.app.analysis_cache.get(self.cache_key(), [])
        if not recs and self.app.candidate_state.candidates:
            return None, 0
        if not recs and self.app.analysis_enabled:
            recs = self.get_engine_analysis()
        for r in recs:
            if r.col is None or r.row is None or r.order is None:
                continue
            if not self.board.is_empty(r.col, r.row):
                continue
            if best is None or r.order < best.order:
                best = r
        if best is None:
            return None, 0
        return (best.col, best.row), self._visits(best.visits)

    # -------------------- engine analysis lifecycle --------------------
    def _refresh_analysis(self) -> None:
        if isinstance(self.app.analysis_mode, BatchRun):
            self._resume_batch_engine(self.app.analysis_mode)
            return
        if self.app.candidate_state.candidates:
            self.start_candidate_search(reset_results=False)
            return
        self.stop_candidate_run()
        self._start_analysis(self.current_side(), is_candidate=False)

    def _apply_analysis_enabled_transition(self, enabled: bool) -> None:
        """Centralize analysis/candidate transitions."""
        if not enabled:
            self.app.analysis_mode = AnalysisModeTag.OFF
            self.stop_analysis()
            return
        if self.app.analysis_mode == AnalysisModeTag.OFF:
            self.app.analysis_mode = AnalysisModeTag.LIVE
        self._refresh_analysis()

    def _start_analysis(self, side_to_analyze: Side, *, is_candidate: bool) -> None:
        root_noise = 0.0 if is_candidate else self.app.analysis_wide_root_noise
        self.engine.kata_set_param("analysisWideRootNoise", root_noise)
        self.engine.start_analysis(self._map_side_to_engine(side_to_analyze), self.analyze_interval_cs)

    # -------------------- batch analysis --------------------
    def start_batch_analysis(self, *, fast: bool = False) -> None:
        self.clear_candidates()
        # Freeze the selected line first. Midline batch resumes from the current ply;
        # starting from a leaf keeps the old behavior of rewinding to the root.
        line = tuple(self.visible_line_moves())
        if self.current_ply() >= len(line) and self.current_ply():
            self.go_first(resume_after=False)
        self.app.analysis_mode = BatchRun(
            kind=BatchKind.RAW_NN if fast else BatchKind.TIMED,
            first_update_at=None,
            line=line,
            expected_rev=self.board.rev,
        )
        self._apply_analysis_enabled_transition(True)

    def finish_batch_analysis(self) -> None:
        self._apply_analysis_enabled_transition(False)

    def cancel_batch_analysis(self) -> None:
        if not isinstance(self.app.analysis_mode, BatchRun):
            return
        self.engine.cancel_reply_capture()
        self.engine.clear_analysis()
        self.app.analysis_mode = AnalysisModeTag.OFF
        self._apply_analysis_enabled_transition(True)

    def step_batch_analysis(self, now: float) -> None:
        run = self.app.analysis_mode
        if not isinstance(run, BatchRun):
            return
        if self._should_cancel_batch(run):
            self.cancel_batch_analysis()
            return
        if run.kind == BatchKind.RAW_NN:
            self._step_batch_raw_nn(run)
            return
        self._step_batch_timed(run, now)

    def _should_cancel_batch(self, run: BatchRun) -> bool:
        return (
            (not self.app.analysis_enabled)
            or self.app.candidate_state.candidates
            or (self.board.rev != run.expected_rev)
        )

    def _step_batch_raw_nn(self, run: BatchRun) -> None:
        if not run.raw_pending:
            if not self.engine.start_kata_raw_nn(0):
                return
            run.raw_pending = True
            return
        done, raw = self.engine.poll_kata_raw_nn()
        if not done:
            return
        run.raw_pending = False
        if raw is None or raw.white_win is None:
            return
        if self._map_side_to_engine(Side.BLUE) == Side.BLUE:
            blue_win = raw.white_win
        else:
            blue_win = 1.0 - raw.white_win
        self._cache_root_eval(blue_win)
        self._advance_batch_position(restart_analysis=False)

    def _step_batch_timed(self, run: BatchRun, now: float) -> None:
        live = self.get_engine_analysis()
        if not live:
            return
        if run.first_update_at is None:
            run.first_update_at = now
            return
        if now - run.first_update_at < SLOW_BATCH_SECONDS_PER_POS:
            return
        self._advance_batch_position(restart_analysis=True)

    def _advance_batch_position(self, *, restart_analysis: bool) -> None:
        run = self.app.analysis_mode
        if not isinstance(run, BatchRun):
            return
        next_ply = self.current_ply()
        if next_ply >= len(run.line):
            self.finish_batch_analysis()
            return
        mv = run.line[next_ply]
        did = False

        def mutate() -> None:
            nonlocal did
            did = self._follow_existing_tree_move(mv, promote=False)

        self.with_analysis_keep_engine_synced(mutate, resume_after=False)
        if not did:
            self.cancel_batch_analysis()
            return
        run.expected_rev = self.board.rev
        if restart_analysis:
            # Batch owns the restart timing after stepping to the next position.
            self.resume_analysis()

    def _resume_batch_engine(self, run: BatchRun) -> None:
        self.stop_candidate_run()
        self.engine.cancel_reply_capture()
        self.engine.clear_analysis()
        if run.kind == BatchKind.RAW_NN:
            run.raw_pending = False
            return
        run.first_update_at = None
        self._start_analysis(self.current_side(), is_candidate=False)

    # -------------------- candidate analysis --------------------
    def add_candidate(self, col: int, row: int) -> None:
        self._update_candidate_selection((col, row), selected=True)

    def toggle_candidate(self, col: int, row: int) -> None:
        key = (col, row)
        if key not in self.app.candidate_state.candidates:
            self.add_candidate(col, row)
            return

        self._update_candidate_selection(key, selected=False)

    def remove_candidate(self, col: int, row: int) -> None:
        self._update_candidate_selection((col, row), selected=False)

    def clear_candidates(self) -> None:
        self._clear_candidate_progress(reset_results=True)
        self._clear_candidate_selection()

    def check_candidate_root(self) -> bool:
        return self._invalidate_candidate_root()

    def start_candidate_search(self, *, reset_results: bool) -> None:
        self._clear_candidate_progress(reset_results=reset_results)
        self._ensure_candidate_root()

    def stop_candidate_run(self) -> None:
        self._end_candidate_run()

    def aggregate_child_analysis(self, recs: List[AnalysisMove]) -> Tuple[Optional[float], int]:
        total_visits = sum(self._visits(r.visits) for r in recs)
        best_winrate = recs[0].winrate if recs else None
        if best_winrate is None:
            return None, total_visits
        return 1.0 - best_winrate, total_visits

    def step_candidate_search(self, now: float) -> None:
        state = self.app.candidate_state
        if not self.is_candidate_mode():
            return
        while True:
            if state.run is None:
                next_keys = self.sorted_candidates_by_visits()
                if not next_keys:
                    return
                col, row = next_keys[0]
                if not self.board.is_empty(col, row):
                    return
                self._begin_candidate_run((col, row))
                return

            child = self.get_engine_analysis()
            winrate, visits = self.aggregate_child_analysis(child)
            run = state.run
            key = run.key
            prev_best = self._visits(state.results.get(key, (None, None))[1])
            if visits > prev_best:
                state.results[key] = (winrate, visits)
                if winrate is not None and visits > 0:
                    self._promote_candidate_to_cache(key, winrate, visits)

            if run.first_update_at is None:
                if not child:
                    return
                run.first_update_at = now
                return

            if now - run.first_update_at < 1.0:
                return

            target = run.target_visits
            if visits <= target:
                return

            next_keys = self.sorted_candidates_by_visits()
            if not next_keys or next_keys[0] == key:
                return

            self._end_candidate_run()

    def get_candidate_analysis(self) -> List[AnalysisMove]:
        state = self.app.candidate_state
        rows: List[Tuple[Tuple[int, int], Optional[float], Optional[int]]] = []
        for key in state.candidates:
            wr, visits = state.results.get(key, (None, None))
            rows.append((key, wr, visits))

        def sort_key(
            row: Tuple[Tuple[int, int], Optional[float], Optional[int]]
        ) -> Tuple[int, float, int, int, int]:
            key, wr, visits = row
            col, row_ = key
            if wr is None:
                return (1, 0.0, -self._visits(visits), col, row_)
            return (0, -wr, -self._visits(visits), col, row_)

        rows.sort(key=sort_key)

        out: List[AnalysisMove] = []
        for order, ((col, row), wr, visits) in enumerate(rows):
            out.append(
                AnalysisMove(
                    move=coord_to_human(col, row),
                    order=order,
                    col=col,
                    row=row,
                    winrate=wr,
                    visits=visits,
                    prior=None,
                    pv=None,
                )
            )
        return out

    def sorted_candidates_by_visits(self) -> List[Tuple[int, int]]:
        state = self.app.candidate_state

        def visit_key(key: Tuple[int, int]) -> Tuple[int, int, int]:
            _wr, visits = state.results.get(key, (None, None))
            return (self._visits(visits), key[0], key[1])

        return sorted(state.candidates, key=visit_key)

    def _clear_candidate_selection(self) -> None:
        state = self.app.candidate_state
        state.candidates.clear()
        state.root_key = None

    def _clear_candidate_progress(self, *, reset_results: bool) -> None:
        state = self.app.candidate_state
        self.stop_candidate_run()
        if reset_results:
            state.results.clear()

    def _ensure_candidate_root(self) -> None:
        state = self.app.candidate_state
        if state.candidates and state.root_key is None:
            state.root_key = self.cache_key()

    def _update_candidate_selection(self, key: Tuple[int, int], *, selected: bool) -> bool:
        state = self.app.candidate_state
        if selected:
            if not self.board.is_empty(*key) or key in state.candidates:
                return False
            state.candidates.add(key)
            self._ensure_candidate_root()
            if self.is_batch_analysis_active():
                self.cancel_batch_analysis()
            return True
        if key not in state.candidates:
            return False
        state.candidates.remove(key)
        state.results.pop(key, None)
        if state.run is not None and key == state.run.key:
            self.stop_candidate_run()
        if not state.candidates:
            self._clear_candidate_selection()
            if self.app.analysis_enabled and not self.is_batch_analysis_active():
                self._apply_analysis_enabled_transition(True)
        return True

    def _invalidate_candidate_root(self) -> bool:
        state = self.app.candidate_state
        if not state.candidates or state.root_key is None or self.cache_key() == state.root_key:
            return False
        self.clear_candidates()
        return True

    def _begin_candidate_run(self, key: Tuple[int, int]) -> None:
        state = self.app.candidate_state
        col, row = key
        self.engine.clear_analysis()
        self.play_engine_mapped(self.current_side(), col, row)
        # Candidate search is a pseudo-root search; suppress root noise.
        self._start_analysis(self.flip_side(self.current_side()), is_candidate=True)
        prev_visits = state.results.get(key, (None, None))[1]
        base_visits = self._visits(prev_visits)
        state.run = CandidateRun(
            key=key,
            target_visits=base_visits * state.ratio,
            first_update_at=None,
        )

    def _end_candidate_run(self) -> None:
        state = self.app.candidate_state
        if state.run is not None:
            self.engine.undo()
            self.engine.clear_analysis()
        state.run = None
