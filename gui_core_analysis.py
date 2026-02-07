from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from board import Move, MoveKind, Side, coord_to_human
from engine import AnalysisMove


@dataclass
class CandidateRun:
    key: Tuple[int, int]
    target_visits: float
    started_at: float


@dataclass
class BatchRun:
    first_update_at: Optional[float]
    expected_rev: int
    seconds_per_pos: float


@dataclass
class AppState:
    pending_size: int
    analysis_enabled: bool
    future_moves: list[Move]
    candidates: set[Tuple[int, int]]
    candidate_results: dict[Tuple[int, int], Tuple[Optional[float], Optional[int]]]
    candidate_run: Optional[CandidateRun]
    candidate_ratio: float
    candidate_root_rev: Optional[int]
    batch_run: Optional[BatchRun]
    analysis_cache: dict[bytes, list[AnalysisMove]]
    last_cache_sig: Optional[tuple]
    analysis_wide_root_noise: float

    @property
    def analysis_running(self) -> bool:
        return self.analysis_enabled


class GuiCoreAnalysisMixin:
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
        had_candidates = bool(self.app.candidates)
        if was_running:
            self.stop_candidate_search()

        self.engine.clear_analysis()
        self.engine.clear_cache()
        self.clear_all_cached_analysis()
        self.app.candidate_results.clear()

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

    def start_batch_analysis(self) -> None:
        self.clear_candidates()
        self.app.batch_run = BatchRun(
            first_update_at=None,
            expected_rev=self.board.rev,
            seconds_per_pos=3.0,
        )
        self.stop_candidate_search()
        self._set_analysis_enabled(True)

    def finish_batch_analysis(self) -> None:
        self.app.batch_run = None
        self._set_analysis_enabled(False)

    def cancel_batch_analysis(self) -> None:
        self.app.batch_run = None

    def step_batch_analysis(self, now: float) -> None:
        run = self.app.batch_run
        if run is None:
            return
        if not self.app.analysis_running:
            self.cancel_batch_analysis()
            return
        if self.app.candidates:
            self.cancel_batch_analysis()
            return
        if self.board.rev != run.expected_rev:
            self.cancel_batch_analysis()
            return
        live = self.get_engine_analysis()
        if not live:
            return
        if run.first_update_at is None:
            run.first_update_at = now
            return
        if now - run.first_update_at < run.seconds_per_pos:
            return
        if not self.app.future_moves:
            self.finish_batch_analysis()
            return
        if not self.step_forward():
            self.cancel_batch_analysis()
            return
        run = self.app.batch_run
        if run is None:
            return
        run.first_update_at = now if self.get_engine_analysis() else None
        run.expected_rev = self.board.rev

    def _set_analysis_enabled(
        self, enabled: bool, *, reset_candidate_results: bool = False
    ) -> None:
        """Centralize analysis/candidate transitions."""
        if not enabled:
            self.app.analysis_enabled = False
            self.stop_analysis()
            return

        self.app.analysis_enabled = True
        if self.app.candidates:
            self.start_candidate_search(reset_results=reset_candidate_results)
            return

        # Live analysis.
        self.stop_candidate_search()
        self._start_analysis(self.current_side(), is_candidate=False)

    def clear_candidates(self) -> None:
        self.app.candidates.clear()
        self.app.candidate_results.clear()
        self.stop_candidate_search()
        self.app.candidate_root_rev = None

    def check_candidate_root(self) -> bool:
        if not self.app.candidates or self.app.candidate_root_rev is None:
            return False
        if self.board.rev == self.app.candidate_root_rev:
            return False
        self.clear_candidates()
        return True

    def start_candidate_search(self, *, reset_results: bool) -> None:
        self.stop_candidate_search()
        if reset_results:
            self.app.candidate_results.clear()
        if not self.app.candidates:
            return
        if self.app.candidate_root_rev is None:
            self.app.candidate_root_rev = self.board.rev

    def set_analysis_wide_root_noise(self, value: float) -> None:
        value = max(0.0, min(2.0, float(value)))
        if abs(self.app.analysis_wide_root_noise - value) < 1e-9:
            return
        self.app.analysis_wide_root_noise = value
        if self.app.analysis_running and not self.app.candidates:
            self._start_analysis(self.current_side(), is_candidate=False)

    def _start_analysis(self, side_to_analyze: Side, *, is_candidate: bool) -> None:
        root_noise = 0.0 if is_candidate else self.app.analysis_wide_root_noise
        self.engine.kata_set_param("analysisWideRootNoise", root_noise)
        self.engine.start_analysis(self._map_side_to_engine(side_to_analyze), self.analyze_interval_cs)

    def aggregate_child_analysis(self, recs: List[AnalysisMove]) -> Tuple[Optional[float], int]:
        total_visits = 0
        total_wr = 0.0
        for r in recs:
            if r.visits is None or r.winrate is None:
                continue
            total_visits += r.visits
            total_wr += r.winrate * r.visits
        if total_visits <= 0:
            return None, 0
        avg = total_wr / total_visits
        return 1.0 - avg, total_visits

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
        rows: List[Tuple[Tuple[int, int], Optional[float], int]] = []
        for key in self.app.candidates:
            wr, visits = self.app.candidate_results.get(key, (None, None))
            rows.append((key, wr, visits))

        def sort_key(
            row: Tuple[Tuple[int, int], Optional[float], Optional[int]]
        ) -> Tuple[int, float, int]:
            _key, wr, visits = row
            if wr is None:
                return (1, 0.0, -(visits or 0))
            return (0, -wr, -(visits or 0))

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
        def visit_key(key: Tuple[int, int]) -> Tuple[int, int, int]:
            _wr, visits = self.app.candidate_results.get(key, (None, None))
            return (visits or 0, key[0], key[1])

        return sorted(self.app.candidates, key=visit_key)

    def _begin_candidate_run(self, key: Tuple[int, int], now: float) -> None:
        col, row = key
        self.engine.clear_analysis()
        self.play_engine_mapped(self.current_side(), col, row)
        # Candidate search is a pseudo-root search; suppress root noise.
        self._start_analysis(self.flip_side(self.current_side()), is_candidate=True)
        prev_visits = self.app.candidate_results.get(key, (None, None))[1]
        base_visits = prev_visits or 0
        self.app.candidate_run = CandidateRun(
            key=key,
            target_visits=base_visits * self.app.candidate_ratio,
            started_at=now,
        )

    def _end_candidate_run(self) -> None:
        if self.app.candidate_run is not None:
            self.engine.undo()
        self.app.candidate_run = None

    def step_candidate_search(self, now: float) -> None:
        if not self.app.analysis_running or not self.app.candidates:
            return
        while True:
            if self.app.candidate_run is None:
                next_keys = self.sorted_candidates_by_visits()
                if not next_keys:
                    return
                col, row = next_keys[0]
                if not self.board.is_empty(col, row):
                    return
                self._begin_candidate_run((col, row), now)
                return

            child = self.get_engine_analysis()
            winrate, visits = self.aggregate_child_analysis(child)
            run = self.app.candidate_run
            key = run.key
            prev_best = self.app.candidate_results.get(key, (None, None))[1] or 0
            if visits > prev_best:
                self.app.candidate_results[key] = (winrate, visits)
                if winrate is not None and visits > 0:
                    self._promote_candidate_to_cache(key, winrate, visits)

            if now - run.started_at < 1.0:
                return

            target = run.target_visits
            if len(self.app.candidates) <= 1:
                return
            if visits <= target:
                return
            self._end_candidate_run()

    def get_active_analysis(self) -> List[AnalysisMove]:
        # Prefer the most informative analysis (cache/live) while candidates can upgrade cached winrate/visits.
        base = self.app.analysis_cache.get(self.cache_key(), [])
        if base:
            return base
        if self.app.candidates:
            return self.get_candidate_analysis()
        if self.app.analysis_running:
            live = self.get_engine_analysis()
            if live:
                return live
        return []

    def maybe_update_analysis_cache(self) -> None:
        if not self.app.analysis_running or self.app.candidates:
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
        if not self.app.analysis_running:
            return
        if self.app.candidates:
            self.start_candidate_search(reset_results=False)
        else:
            self.stop_candidate_search()
            self._start_analysis(self.current_side(), is_candidate=False)

    def stop_analysis(self) -> None:
        self.stop_candidate_search()
        self.engine.stop_analysis()

    def toggle_analysis(self) -> None:
        if self.app.analysis_running:
            self._set_analysis_enabled(False)
        else:
            self._set_analysis_enabled(True)

    def add_candidate(self, col: int, row: int) -> None:
        if not self.board.is_empty(col, row):
            return
        key = (col, row)
        if key in self.app.candidates:
            return
        if not self.app.candidates:
            self.app.candidate_root_rev = self.board.rev
        self.app.candidates.add(key)
        return

    def toggle_candidate(self, col: int, row: int) -> None:
        if not self.board.is_empty(col, row):
            return
        key = (col, row)
        if key not in self.app.candidates:
            self.add_candidate(col, row)
            return

        self.app.candidates.remove(key)
        self.app.candidate_results.pop(key, None)
        if self.app.candidate_run is not None and key == self.app.candidate_run.key:
            self.stop_candidate_search()

        if not self.app.candidates:
            self.clear_candidates()
            if self.app.analysis_running:
                self._set_analysis_enabled(True)

    def remove_candidate(self, col: int, row: int) -> None:
        key = (col, row)
        if key in self.app.candidates:
            self.toggle_candidate(col, row)

    def get_top_move(self) -> Tuple[Optional[Tuple[int, int]], int]:
        for r in self.get_active_analysis():
            if r.col is None or r.row is None:
                continue
            if self.board.is_empty(r.col, r.row):
                v = 0 if r.visits is None else r.visits
                return (r.col, r.row), max(1, v)
        return None, 1

    def tick(self, now: float) -> None:
        if self.app.batch_run is not None:
            self.step_batch_analysis(now)
            self.maybe_update_analysis_cache()
            return
        if self.check_candidate_root():
            self.resume_analysis()
        self.step_candidate_search(now)
        self.maybe_update_analysis_cache()
