from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from board import Move, MoveKind, Side, coord_to_human
from engine import AnalysisMove


@dataclass
class CandidateRun:
    key: Tuple[int, int]
    target_visits: float
    first_update_at: Optional[float]


@dataclass
class BatchRun:
    first_update_at: Optional[float]
    expected_rev: int
    seconds_per_pos: float
    fast: bool = False
    raw_pending: bool = False


@dataclass
class CandidateState:
    candidates: set[Tuple[int, int]]
    results: dict[Tuple[int, int], Tuple[Optional[float], Optional[int]]]
    run: Optional[CandidateRun]
    ratio: float
    root_rev: Optional[int]


class AnalysisModeTag(Enum):
    OFF = "off"
    LIVE = "live"
    CANDIDATE = "candidate"


AnalysisMode = AnalysisModeTag | BatchRun


@dataclass
class AppState:
    pending_size: int
    future_moves: list[Move]
    candidate_state: CandidateState
    analysis_cache: dict[bytes, list[AnalysisMove]]
    last_cache_sig: Optional[tuple]
    analysis_wide_root_noise: float
    analysis_mode: AnalysisMode = AnalysisModeTag.OFF

    @property
    def analysis_running(self) -> bool:
        return self.analysis_mode != AnalysisModeTag.OFF


class GuiCoreAnalysisMixin:
    def is_batch_analysis_active(self) -> bool:
        return isinstance(self.app.analysis_mode, BatchRun)

    # -------------------- analysis cache (GUI-only) --------------------
    def current_side(self) -> Side:
        if not self.board.history:
            return Side.RED
        last = self.board.history[-1]
        return self.flip_side(last.side)

    @staticmethod
    def _cache_move_token(mv: Move) -> tuple[int, int, int, int]:
        kind = 0
        if mv.kind == MoveKind.PASS:
            kind = 1
        elif mv.kind == MoveKind.SWAP:
            kind = 2
        return (kind, int(mv.side), 0 if mv.col is None else mv.col, 0 if mv.row is None else mv.row)

    def cache_key_for_moves(self, moves: Sequence[Move]) -> bytes:
        side = Side.RED if not moves else self.flip_side(moves[-1].side)
        h = hashlib.blake2b(digest_size=16)
        h.update(struct.pack("<IB", self.board.n, int(side)))
        for mv in moves:
            kind, mv_side, col, row = self._cache_move_token(mv)
            h.update(struct.pack("<BBHH", kind, mv_side, col, row))
        return h.digest()

    def cache_key(self) -> bytes:
        return self.cache_key_for_moves(self.board.history)

    def cache_reset_sig(self) -> None:
        self.app.last_cache_sig = None

    def clear_all_cached_analysis(self) -> None:
        self.app.analysis_cache.clear()
        self.cache_reset_sig()

    def clear_analysis_caches(self) -> None:
        was_running = self.app.analysis_running
        state = self.app.candidate_state
        had_candidates = bool(state.candidates)
        if was_running:
            self.stop_candidate_search()

        self.engine.clear_analysis()
        self.engine.clear_cache()
        self.clear_all_cached_analysis()
        state.results.clear()

        if was_running:
            if had_candidates:
                self.start_candidate_search(reset_results=True)
            else:
                self._start_analysis(self.current_side(), is_candidate=False)

    def flip_side(self, side: Side) -> Side:
        return Side.BLUE if side == Side.RED else Side.RED

    def swap_active(self) -> bool:
        return len(self.board.history) >= 2 and self.board.history[1].kind == MoveKind.SWAP

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

    def stop_candidate_search(self) -> None:
        self._end_candidate_run()

    def start_batch_analysis(self, *, fast: bool = False) -> None:
        self.clear_candidates()
        if self.board.history and not self.app.future_moves:
            self.go_first()
        self.app.analysis_mode = BatchRun(
            first_update_at=None,
            expected_rev=self.board.rev,
            seconds_per_pos=0.0 if fast else 3.0,
            fast=fast,
        )
        self._apply_analysis_enabled_transition(True)

    def finish_batch_analysis(self) -> None:
        self._apply_analysis_enabled_transition(False)

    def cancel_batch_analysis(self) -> None:
        if not isinstance(self.app.analysis_mode, BatchRun):
            return
        self.engine.cancel_reply_capture()
        if self.app.candidate_state.candidates:
            self.app.analysis_mode = AnalysisModeTag.CANDIDATE
            self.start_candidate_search(reset_results=False)
            return
        self.app.analysis_mode = AnalysisModeTag.LIVE
        self.stop_candidate_search()
        self._start_analysis(self.current_side(), is_candidate=False)

    def step_batch_analysis(self, now: float) -> None:
        run = self.app.analysis_mode
        if not isinstance(run, BatchRun):
            return
        if (
            (not self.app.analysis_running)
            or self.app.candidate_state.candidates
            or (self.board.rev != run.expected_rev)
        ):
            self.cancel_batch_analysis()
            return
        if run.fast:
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
            self._promote_root_eval_to_cache(blue_win)
            self._advance_batch_position(restart_analysis=False)
            return
        live = self.get_engine_analysis()
        if not live:
            return
        if run.first_update_at is None:
            run.first_update_at = now
            return
        if now - run.first_update_at < run.seconds_per_pos:
            return
        self._advance_batch_position(restart_analysis=True)

    def _apply_analysis_enabled_transition(self, enabled: bool) -> None:
        """Centralize analysis/candidate transitions."""
        if not enabled:
            self.app.analysis_mode = AnalysisModeTag.OFF
            self.stop_analysis()
            return

        if isinstance(self.app.analysis_mode, BatchRun):
            self._resume_batch_engine(self.app.analysis_mode)
            return

        if self.app.candidate_state.candidates:
            self.app.analysis_mode = AnalysisModeTag.CANDIDATE
            self.start_candidate_search(reset_results=False)
            return

        # Live analysis.
        self.app.analysis_mode = AnalysisModeTag.LIVE
        self.stop_candidate_search()
        self._start_analysis(self.current_side(), is_candidate=False)

    def _clear_candidate_state(self) -> None:
        state = self.app.candidate_state
        state.candidates.clear()
        state.results.clear()
        self.stop_candidate_search()
        state.root_rev = None

    def clear_candidates(self) -> None:
        self._clear_candidate_state()
        if self.app.analysis_mode == AnalysisModeTag.CANDIDATE:
            self.app.analysis_mode = AnalysisModeTag.LIVE

    def check_candidate_root(self) -> bool:
        state = self.app.candidate_state
        if not state.candidates or state.root_rev is None:
            return False
        if self.board.rev == state.root_rev:
            return False
        self.clear_candidates()
        return True

    def start_candidate_search(self, *, reset_results: bool) -> None:
        state = self.app.candidate_state
        self.stop_candidate_search()
        if reset_results:
            state.results.clear()
        if state.candidates and state.root_rev is None:
            state.root_rev = self.board.rev

    def set_analysis_wide_root_noise(self, value: float) -> None:
        value = max(0.0, min(2.0, float(value)))
        if abs(self.app.analysis_wide_root_noise - value) < 1e-9:
            return
        self.app.analysis_wide_root_noise = value
        if self.app.analysis_running and not self.app.candidate_state.candidates:
            self._start_analysis(self.current_side(), is_candidate=False)

    def _start_analysis(self, side_to_analyze: Side, *, is_candidate: bool) -> None:
        root_noise = 0.0 if is_candidate else self.app.analysis_wide_root_noise
        self.engine.kata_set_param("analysisWideRootNoise", root_noise)
        self.engine.start_analysis(self._map_side_to_engine(side_to_analyze), self.analyze_interval_cs)

    def aggregate_child_analysis(self, recs: List[AnalysisMove]) -> Tuple[Optional[float], int]:
        total_visits = sum(self._visits(r.visits) for r in recs)
        best_winrate = recs[0].winrate if recs else None
        if best_winrate is None:
            return None, total_visits
        return 1.0 - best_winrate, total_visits

    @staticmethod
    def _visits(v: Optional[int]) -> int:
        return 0 if v is None else v

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
        secondary_by_coord: dict[Tuple[int, int], AnalysisMove] = {}
        secondary_extras: List[AnalysisMove] = []
        for r in secondary:
            if r.col is None or r.row is None:
                secondary_extras.append(r)
                continue
            secondary_by_coord[(r.col, r.row)] = r

        merged: List[AnalysisMove] = []
        for r in primary:
            if r.col is None or r.row is None:
                merged.append(r)
                continue
            merged.append(self._merge_analysis(r, secondary_by_coord.pop((r.col, r.row), None)))

        merged.extend(secondary_extras)
        merged.extend(secondary_by_coord.values())
        return merged

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

    def _promote_root_eval_to_cache(self, blue_win: float) -> None:
        side_to_play = self.current_side()
        synthetic_wr = (1.0 - blue_win) if side_to_play == Side.RED else blue_win
        incoming = AnalysisMove(
            move="__rawnn_root__",
            order=10**9,
            col=None,
            row=None,
            winrate=synthetic_wr,
            visits=0,
            prior=None,
            pv=None,
        )
        cache_key = self.cache_key()
        base = self.app.analysis_cache.get(cache_key, [])
        merged = [r for r in base if not (r.col is None and r.row is None and r.move == "__rawnn_root__")]
        merged.append(incoming)
        if merged == base:
            return
        self.app.analysis_cache[cache_key] = merged
        self.cache_reset_sig()

    def _advance_batch_position(self, *, restart_analysis: bool) -> None:
        if not self.app.future_moves:
            self.finish_batch_analysis()
            return
        if not self.step_forward():
            self.cancel_batch_analysis()
            return
        run = self.app.analysis_mode
        if not isinstance(run, BatchRun):
            return
        if restart_analysis:
            # Reuse normal step/resume semantics so batch behaves like timed manual stepping.
            self.resume_analysis()
            run.first_update_at = None
        run.expected_rev = self.board.rev

    def _resume_batch_engine(self, run: BatchRun) -> None:
        self.stop_candidate_search()
        if run.fast:
            self.engine.clear_analysis()
            return
        self._start_analysis(self.current_side(), is_candidate=False)

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

    def get_candidate_analysis(self) -> List[AnalysisMove]:
        state = self.app.candidate_state
        rows: List[Tuple[Tuple[int, int], Optional[float], int]] = []
        for key in state.candidates:
            wr, visits = state.results.get(key, (None, None))
            rows.append((key, wr, visits))

        def sort_key(
            row: Tuple[Tuple[int, int], Optional[float], Optional[int]]
        ) -> Tuple[int, float, int]:
            _key, wr, visits = row
            if wr is None:
                return (1, 0.0, -self._visits(visits))
            return (0, -wr, -self._visits(visits))

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
        state.run = None

    def step_candidate_search(self, now: float) -> None:
        state = self.app.candidate_state
        if self.app.analysis_mode != AnalysisModeTag.CANDIDATE or not state.candidates:
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
            if len(state.candidates) <= 1:
                return
            if visits <= target:
                return
            self._end_candidate_run()

    def get_active_analysis(self) -> List[AnalysisMove]:
        # Prefer the most informative analysis (cache/live) while candidates can upgrade cached winrate/visits.
        base = self.app.analysis_cache.get(self.cache_key(), [])
        if base:
            return base
        if self.app.candidate_state.candidates:
            return self.get_candidate_analysis()
        if self.app.analysis_running:
            live = self.get_engine_analysis()
            if live:
                return live
        return []

    def maybe_update_analysis_cache(self) -> None:
        if self.app.analysis_mode in (AnalysisModeTag.OFF, AnalysisModeTag.CANDIDATE):
            return
        live = self.get_engine_analysis()
        if not live:
            return

        key = self.cache_key()
        top = live[0]
        sig = (key, len(live), top.move, top.order, top.visits, top.winrate, top.prior)
        if sig == self.app.last_cache_sig:
            return

        self._merge_live_into_cache(live)
        self.app.last_cache_sig = sig

    # -------------------- engine/board coordination --------------------
    def resume_analysis(self) -> None:
        mode = self.app.analysis_mode
        if mode == AnalysisModeTag.OFF:
            return
        if mode == AnalysisModeTag.CANDIDATE:
            self.start_candidate_search(reset_results=False)
            return
        if isinstance(mode, BatchRun):
            self._resume_batch_engine(mode)
            return
        self.stop_candidate_search()
        self._start_analysis(self.current_side(), is_candidate=False)

    def stop_analysis(self) -> None:
        self.stop_candidate_search()
        self.engine.stop_analysis()

    def toggle_analysis(self) -> None:
        if self.app.analysis_running:
            self._apply_analysis_enabled_transition(False)
        else:
            self._apply_analysis_enabled_transition(True)

    def add_candidate(self, col: int, row: int) -> None:
        state = self.app.candidate_state
        if not self.board.is_empty(col, row):
            return
        key = (col, row)
        if key in state.candidates:
            return
        if not state.candidates:
            state.root_rev = self.board.rev
        state.candidates.add(key)
        if self.app.analysis_mode == AnalysisModeTag.LIVE:
            self.app.analysis_mode = AnalysisModeTag.CANDIDATE
        return

    def toggle_candidate(self, col: int, row: int) -> None:
        state = self.app.candidate_state
        if not self.board.is_empty(col, row):
            return
        key = (col, row)
        if key not in state.candidates:
            self.add_candidate(col, row)
            return

        state.candidates.remove(key)
        state.results.pop(key, None)
        if state.run is not None and key == state.run.key:
            self.stop_candidate_search()

        if not state.candidates:
            was_candidate_mode = self.app.analysis_mode == AnalysisModeTag.CANDIDATE
            self._clear_candidate_state()
            if was_candidate_mode:
                self._apply_analysis_enabled_transition(True)
            return

    def remove_candidate(self, col: int, row: int) -> None:
        state = self.app.candidate_state
        key = (col, row)
        if key in state.candidates:
            self.toggle_candidate(col, row)

    def get_top_move(self) -> Tuple[Optional[Tuple[int, int]], int]:
        best: Optional[AnalysisMove] = None
        for r in self.get_active_analysis():
            if r.col is None or r.row is None or r.order is None:
                continue
            if not self.board.is_empty(r.col, r.row):
                continue
            if best is None or r.order < best.order:
                best = r
        if best is None:
            return None, 0
        return (best.col, best.row), self._visits(best.visits)

    def tick(self, now: float) -> None:
        if isinstance(self.app.analysis_mode, BatchRun):
            self.step_batch_analysis(now)
            self.maybe_update_analysis_cache()
            return
        if self.check_candidate_root():
            self.resume_analysis()
        self.step_candidate_search(now)
        self.maybe_update_analysis_cache()
