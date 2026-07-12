from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

from board import HexBoard, Move, MoveKind, Side, coord_to_human
from engine import AnalysisMove
from gui.state import (
    AnalysisModeTag,
    BatchKind,
    BatchRun,
    TransitionKind,
)

SLOW_BATCH_SECONDS_PER_POS = 3.0


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    side: Side
    batch_kind: Optional[BatchKind] = None
    allowed_moves: tuple[Tuple[int, int], ...] = ()


class GuiCoreAnalysisMixin:
    # -------------------- mode and engine mapping --------------------
    def is_batch_analysis_active(self) -> bool:
        return isinstance(self.session.analysis.mode, BatchRun)

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

    @staticmethod
    def has_candidate_result(r: AnalysisMove) -> bool:
        return r.winrate is not None and r.visits is not None and r.visits > 0

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
        self.session.analysis.last_cache_sig = None

    def clear_all_cached_analysis(self) -> None:
        self.session.analysis.cache.clear()
        self.session.analysis.root_eval_cache.clear()
        self.cache_reset_sig()

    def clear_analysis_caches(self) -> None:
        was_running = self.session.analysis.enabled

        self.engine.clear_analysis()
        self.engine.clear_cache()
        self.clear_all_cached_analysis()

        if was_running:
            self.restart_analysis()

    def _merge_analysis(self, primary: AnalysisMove, secondary: Optional[AnalysisMove]) -> AnalysisMove:
        """Primary supplies display order; deeper visits supplies eval metadata."""
        if secondary is None:
            return primary
        best = primary if self._visits(primary.visits) >= self._visits(secondary.visits) else secondary
        order = primary.order if primary.order is not None else secondary.order
        # Keep eval metadata from the same source row as the displayed visits/winrate.
        return replace(
            primary,
            order=order,
            winrate=best.winrate,
            visits=best.visits,
            prior=best.prior,
            pv=best.pv,
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
        existing = self.session.analysis.cache.get(cache_key)
        if existing is None:
            self.session.analysis.cache[cache_key] = list(live)
            self.cache_reset_sig()
            return

        merged = self._merge_analysis_lists(live, existing)
        self.session.analysis.cache[cache_key] = merged
        self.cache_reset_sig()

    def _cached_analysis_row(self, key: Tuple[int, int]) -> Optional[AnalysisMove]:
        for r in self.session.analysis.cache.get(self.cache_key(), []):
            if r.col == key[0] and r.row == key[1]:
                return r
        return None

    def _candidate_cache_rows(self, live: List[AnalysisMove]) -> List[AnalysisMove]:
        selected = self.session.analysis.candidate_selection.candidates
        out: List[AnalysisMove] = []
        for r in live:
            if r.col is None or r.row is None or (r.col, r.row) not in selected:
                continue
            if not self.has_candidate_result(r):
                continue
            # Unordered candidate rows can inform display without becoming top moves.
            out.append(replace(r, move=coord_to_human(r.col, r.row), order=None))
        return out

    def candidate_result(self, key: Tuple[int, int]) -> Tuple[Optional[float], Optional[int]]:
        row = self._analysis_row_for_key(key)
        if row is None:
            return None, None
        return row.winrate, row.visits

    def _live_analysis_row(self, key: Tuple[int, int]) -> Optional[AnalysisMove]:
        for r in self.get_engine_analysis():
            if r.col == key[0] and r.row == key[1]:
                return r
        return None

    def _analysis_row_for_key(self, key: Tuple[int, int]) -> Optional[AnalysisMove]:
        live = self._live_analysis_row(key)
        cached = self._cached_analysis_row(key)
        if live is None:
            return cached
        if cached is None:
            return live
        return self._merge_analysis(live, cached)

    def _cache_root_eval(self, blue_win: float) -> None:
        side_to_play = self.current_side()
        synthetic_wr = (1.0 - blue_win) if side_to_play == Side.RED else blue_win
        cache_key = self.cache_key()
        if self.session.analysis.root_eval_cache.get(cache_key) == synthetic_wr:
            return
        self.session.analysis.root_eval_cache[cache_key] = synthetic_wr

    def maybe_update_analysis_cache(self) -> None:
        if not self.session.analysis.enabled:
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
        if sig == self.session.analysis.last_cache_sig:
            return

        if self.session.analysis.candidate_selection.candidates:
            candidate_rows = self._candidate_cache_rows(live)
            if not candidate_rows:
                return
            existing = self.session.analysis.cache.get(key, [])
            merged = self._merge_analysis_lists(existing, candidate_rows)
            if merged != existing:
                self.session.analysis.cache[key] = merged
                self.cache_reset_sig()
        else:
            self._merge_live_into_cache(live)
        self.session.analysis.last_cache_sig = sig

    # -------------------- analysis loop and controls --------------------
    def tick(self, now: float) -> None:
        if isinstance(self.session.analysis.mode, BatchRun):
            self.step_batch_analysis(now)
            self.maybe_update_analysis_cache()
            return
        if self.check_candidate_root():
            self.restart_analysis()
        self.maybe_update_analysis_cache()

    def toggle_analysis(self) -> None:
        self.set_analysis_enabled(not self.session.analysis.enabled)

    def restart_analysis(self) -> None:
        request = self._desired_analysis_request()
        if request is None:
            return
        self._apply_analysis_request(request)

    def pause_engine_analysis(self) -> None:
        self.engine.stop_analysis()
        self.engine.clear_analysis()

    def set_analysis_enabled(self, enabled: bool) -> None:
        if enabled == self.session.analysis.enabled:
            return
        if not enabled:
            self.session.analysis.mode = AnalysisModeTag.OFF
            self.pause_engine_analysis()
            return
        if self.session.analysis.mode == AnalysisModeTag.OFF:
            self.session.analysis.mode = AnalysisModeTag.LIVE
        self.restart_analysis()

    def leave_batch_for_live(self) -> None:
        if not self._exit_batch_mode():
            return
        self.engine.cancel_reply_capture()
        self.restart_analysis()

    def _exit_batch_mode(self) -> bool:
        if not isinstance(self.session.analysis.mode, BatchRun):
            return False
        self.session.analysis.mode = AnalysisModeTag.LIVE
        return True

    def set_analysis_wide_root_noise(self, value: float) -> None:
        value = max(0.0, min(2.0, float(value)))
        if abs(self.session.analysis.wide_root_noise - value) < 1e-9:
            return
        self.session.analysis.wide_root_noise = value
        if self.session.analysis.enabled:
            self.restart_analysis()

    # -------------------- analysis queries --------------------
    def get_active_analysis(self) -> List[AnalysisMove]:
        # Prefer the most informative analysis (cache/live) while candidates can
        # upgrade cached winrate/visits.
        if self.session.analysis.candidate_selection.candidates:
            base = self.session.analysis.cache.get(self.cache_key(), [])
            if not base:
                return self.get_candidate_analysis()
            return self._merge_analysis_lists(base, self.get_candidate_analysis())
        base = self.session.analysis.cache.get(self.cache_key(), [])
        if base:
            return base
        if self.session.analysis.enabled:
            live = self.get_engine_analysis()
            if live:
                return live
        return []

    def get_top_move(self) -> Tuple[Optional[Tuple[int, int]], int]:
        best: Optional[AnalysisMove] = None
        candidate_mode = bool(self.session.analysis.candidate_selection.candidates)
        recs = self.get_candidate_analysis() if candidate_mode else self.get_active_analysis()
        for r in recs:
            if r.col is None or r.row is None or r.order is None:
                continue
            if candidate_mode and not self.has_candidate_result(r):
                continue
            if not self.board.is_empty(r.col, r.row):
                continue
            if best is None or r.order < best.order:
                best = r
        if best is None:
            return None, 0
        return (best.col, best.row), self._visits(best.visits)

    # -------------------- engine analysis lifecycle --------------------
    def _desired_analysis_request(self) -> Optional[AnalysisRequest]:
        mode = self.session.analysis.mode
        if mode == AnalysisModeTag.OFF:
            return None
        side = self.current_side()
        if isinstance(mode, BatchRun):
            return AnalysisRequest(side=side, batch_kind=mode.kind)
        candidates = self.session.analysis.candidate_selection.candidates
        if candidates:
            return AnalysisRequest(
                side=side,
                allowed_moves=tuple(sorted(candidates)),
            )
        return AnalysisRequest(side=side)

    def _apply_analysis_request(self, request: AnalysisRequest) -> None:
        if request.batch_kind is not None:
            run = self.session.analysis.mode
            if not isinstance(run, BatchRun):
                raise AssertionError("Batch request without batch state")
            self.engine.cancel_reply_capture()
            self.engine.clear_analysis()
            if request.batch_kind == BatchKind.RAW_NN:
                run.raw_pending = False
                return
            run.first_update_at = None
        elif request.allowed_moves:
            self._ensure_candidate_root()
            self.maybe_update_analysis_cache()
            self.engine.clear_analysis()
        else:
            self.engine.clear_analysis()
        self._start_analysis(request.side, allowed_moves=request.allowed_moves)

    def _start_analysis(self, side_to_analyze: Side, *, allowed_moves: Sequence[Tuple[int, int]] = ()) -> None:
        mapped_side = self._map_side_to_engine(side_to_analyze)
        mapped_moves = [self._map_coords_to_engine(col, row) for col, row in allowed_moves]
        self.engine.kata_set_param("analysisWideRootNoise", self.session.analysis.wide_root_noise)
        allow_filters = ((mapped_side, mapped_moves),) if mapped_moves else ()
        self.engine.start_analysis(mapped_side, self.analyze_interval_cs, allow_filters)

    # -------------------- batch analysis --------------------
    def start_batch_analysis(self, *, fast: bool = False) -> None:
        self._clear_candidate_selection()
        # Freeze the selected line first. Midline batch resumes from the current ply;
        # starting from a leaf keeps the old behavior of rewinding to the root.
        line = tuple(self.visible_line_moves())
        if self.current_ply() >= len(line) and self.current_ply():
            target = self._cursor_after_steps(self.current_ply(), forward=False)
            if target is None:
                raise AssertionError("Failed to rewind batch line")
            self._commit_cursor(target, kind=TransitionKind.BATCH_START)
        self.session.analysis.mode = BatchRun(
            kind=BatchKind.RAW_NN if fast else BatchKind.TIMED,
            first_update_at=None,
            line=line,
            expected_rev=self.board.rev,
        )
        self.restart_analysis()

    def finish_batch_analysis(self) -> None:
        self.set_analysis_enabled(False)

    def cancel_batch_analysis(self) -> None:
        self.leave_batch_for_live()

    def step_batch_analysis(self, now: float) -> None:
        run = self.session.analysis.mode
        if not isinstance(run, BatchRun):
            return
        if self.board.rev != run.expected_rev:
            self._rebuild_board_from_moves(self.current_path_moves())
            self.cancel_batch_analysis()
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
            (not self.session.analysis.enabled)
            or self.session.analysis.candidate_selection.candidates
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
        self.maybe_update_analysis_cache()
        self._advance_batch_position(restart_analysis=True)

    def _advance_batch_position(self, *, restart_analysis: bool) -> None:
        run = self.session.analysis.mode
        if not isinstance(run, BatchRun):
            return
        next_ply = self.current_ply()
        if next_ply >= len(run.line):
            self.finish_batch_analysis()
            return
        mv = run.line[next_ply]
        target = self.session.tree.find_child(self.session.tree.cursor, mv)
        if target is None:
            self.cancel_batch_analysis()
            return
        self._commit_cursor(target, kind=TransitionKind.BATCH_STEP)
        run.expected_rev = self.board.rev
        if restart_analysis:
            # Batch owns the restart timing after stepping to the next position.
            self.restart_analysis()

    # -------------------- candidate analysis --------------------
    def add_candidate(self, col: int, row: int) -> None:
        self._update_candidate_selection((col, row), selected=True)

    def toggle_candidate(self, col: int, row: int) -> None:
        key = (col, row)
        if key not in self.session.analysis.candidate_selection.candidates:
            self.add_candidate(col, row)
            return

        self._update_candidate_selection(key, selected=False)

    def remove_candidate(self, col: int, row: int) -> None:
        self._update_candidate_selection((col, row), selected=False)

    def clear_candidates(self) -> None:
        had_candidates = bool(self.session.analysis.candidate_selection.candidates)
        self._clear_candidate_selection()
        if had_candidates and self.session.analysis.enabled and not self.is_batch_analysis_active():
            self.restart_analysis()

    def check_candidate_root(self) -> bool:
        return self._invalidate_candidate_root()

    def get_candidate_analysis(self) -> List[AnalysisMove]:
        selection = self.session.analysis.candidate_selection
        rows: List[Tuple[Tuple[int, int], AnalysisMove]] = []
        for key in selection.candidates:
            source = self._analysis_row_for_key(key)
            if source is None:
                col, row = key
                source = AnalysisMove(
                    move=coord_to_human(col, row),
                    order=None,
                    col=col,
                    row=row,
                    winrate=None,
                    visits=None,
                    prior=None,
                    pv=None,
                )
            rows.append((key, source))

        def sort_key(item: Tuple[Tuple[int, int], AnalysisMove]) -> Tuple[int, float, int, int, int]:
            key, source = item
            col, row_ = key
            if source.winrate is None:
                return (1, 0.0, -self._visits(source.visits), col, row_)
            return (0, -source.winrate, -self._visits(source.visits), col, row_)

        rows.sort(key=sort_key)

        out: List[AnalysisMove] = []
        for order, ((col, row), source) in enumerate(rows):
            out.append(replace(source, move=coord_to_human(col, row), order=order, col=col, row=row))
        return out

    def _clear_candidate_selection(self) -> None:
        selection = self.session.analysis.candidate_selection
        selection.candidates.clear()
        selection.root_key = None

    def _ensure_candidate_root(self) -> None:
        selection = self.session.analysis.candidate_selection
        if selection.candidates and selection.root_key is None:
            selection.root_key = self.cache_key()

    def _update_candidate_selection(self, key: Tuple[int, int], *, selected: bool) -> bool:
        selection = self.session.analysis.candidate_selection
        if self.session.analysis.enabled and not self.is_batch_analysis_active():
            self.maybe_update_analysis_cache()
        if selected:
            if not self.board.is_empty(*key) or key in selection.candidates:
                return False
            selection.candidates.add(key)
            self._ensure_candidate_root()
            if self.is_batch_analysis_active():
                self.cancel_batch_analysis()
            elif self.session.analysis.enabled:
                self.restart_analysis()
            return True
        if key not in selection.candidates:
            return False
        selection.candidates.remove(key)
        if not selection.candidates:
            self._clear_candidate_selection()
            if self.session.analysis.enabled and not self.is_batch_analysis_active():
                self.restart_analysis()
        elif self.session.analysis.enabled and not self.is_batch_analysis_active():
            self.restart_analysis()
        return True

    def _invalidate_candidate_root(self) -> bool:
        selection = self.session.analysis.candidate_selection
        if (
            not selection.candidates
            or selection.root_key is None
            or self.cache_key() == selection.root_key
        ):
            return False
        self._clear_candidate_selection()
        return True
