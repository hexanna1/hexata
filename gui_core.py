from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, NoReturn, Optional, Tuple

from board import HexBoard, Move, MoveKind, Side, coord_to_human
from engine import KataHexEngine, AnalysisMove
from hexworld import parse_hexworld_position


@dataclass
class CandidateRun:
    key: Tuple[int, int]
    base_visits: int
    target_visits: float
    started_at: float


class AnalysisMode(Enum):
    OFF = "off"
    LIVE = "live"
    CANDIDATE = "candidate"


@dataclass
class AppState:
    pending_size: int
    to_play: Side
    analysis_mode: AnalysisMode
    future_moves: list[Move]
    candidates: set[Tuple[int, int]]
    candidate_results: dict[Tuple[int, int], Tuple[Optional[float], Optional[int]]]
    candidate_analysis: List[AnalysisMove]
    candidate_run: Optional[CandidateRun]
    candidate_ratio: float
    candidate_root_rev: Optional[int]
    analysis_cache: dict[tuple[int, int], list[AnalysisMove]]
    last_cache_sig: Optional[tuple]
    analysis_wide_root_noise: float

    @property
    def analysis_running(self) -> bool:
        return self.analysis_mode is not AnalysisMode.OFF


class GuiCore:
    def __init__(self, board: HexBoard, engine: KataHexEngine, *, analyze_interval_cs: int = 15) -> None:
        self.board = board
        self.engine = engine
        self.analyze_interval_cs = analyze_interval_cs

        self.app = AppState(
            pending_size=board.n,
            to_play=Side.RED,
            analysis_mode=AnalysisMode.OFF,
            future_moves=[],
            candidates=set(),
            candidate_results={},
            candidate_analysis=[],
            candidate_run=None,
            candidate_ratio=1.6,
            candidate_root_rev=None,
            analysis_cache={},
            last_cache_sig=None,
            analysis_wide_root_noise=0.04,
        )

    # -------------------- analysis cache (GUI-only) --------------------
    def cache_key(self) -> tuple[int, int]:
        return (len(self.board.history), int(self.app.to_play))

    def cache_reset_sig(self) -> None:
        self.app.last_cache_sig = None

    def clear_all_cached_analysis(self) -> None:
        self.app.analysis_cache.clear()
        self.cache_reset_sig()

    def clear_cached_analysis_from_ply(self, cutoff_len: int, *, keep_current: bool) -> None:
        if cutoff_len <= 0:
            self.clear_all_cached_analysis()
            return
        if keep_current:
            keep_len = cutoff_len
            pruned = {
                key: val
                for key, val in self.app.analysis_cache.items()
                if key[0] <= keep_len
            }
        else:
            pruned = {
                key: val
                for key, val in self.app.analysis_cache.items()
                if key[0] < cutoff_len
            }
        if pruned == self.app.analysis_cache:
            return
        self.app.analysis_cache = pruned
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
        self.app.candidate_analysis.clear()

        if was_running:
            if had_candidates:
                self.start_candidate_search(reset_results=True)
            else:
                self._start_analysis(self.app.to_play, is_candidate=False)

    def flip_side(self, side: Side) -> Side:
        return Side.BLUE if side == Side.RED else Side.RED

    def stop_candidate_search(self) -> None:
        self._end_candidate_run()

    def _set_analysis_mode(
        self, mode: AnalysisMode, *, reset_candidate_results: bool = False
    ) -> None:
        """Centralize analysis/candidate transitions."""
        if mode is AnalysisMode.OFF:
            self.app.analysis_mode = mode
            self.stop_analysis()
            return

        self.app.analysis_mode = mode
        if mode is AnalysisMode.CANDIDATE:
            self.start_candidate_search(reset_results=reset_candidate_results)
            return

        # Live analysis.
        self.stop_candidate_search()
        self._start_analysis(self.app.to_play, is_candidate=False)

    def clear_candidates(self) -> None:
        self.app.candidates.clear()
        self.app.candidate_results.clear()
        self.app.candidate_analysis.clear()
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
            self.app.candidate_analysis.clear()
            self.rebuild_candidate_analysis()
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
            self._start_analysis(self.app.to_play, is_candidate=False)

    def _start_analysis(self, side_to_analyze: Side, *, is_candidate: bool) -> None:
        root_noise = 0.0 if is_candidate else self.app.analysis_wide_root_noise
        self.engine.kata_set_param("analysisWideRootNoise", root_noise)
        self.engine.start_analysis(side_to_analyze, self.analyze_interval_cs)

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
        return AnalysisMove(
            move=primary.move,
            order=order,
            col=primary.col,
            row=primary.row,
            winrate=best.winrate,
            visits=best.visits,
            prior=prior,
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

    def rebuild_candidate_analysis(self) -> None:
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
                )
            )
        self.app.candidate_analysis = out

    def sorted_candidates_by_visits(self) -> List[Tuple[int, int]]:
        def visit_key(key: Tuple[int, int]) -> Tuple[int, int, int]:
            _wr, visits = self.app.candidate_results.get(key, (None, None))
            return (visits or 0, key[0], key[1])

        return sorted(self.app.candidates, key=visit_key)

    def _begin_candidate_run(self, key: Tuple[int, int], now: float) -> None:
        col, row = key
        self.engine.clear_analysis()
        self.engine.play(self.app.to_play, col, row)
        # Candidate search is a pseudo-root search; suppress root noise.
        self._start_analysis(self.flip_side(self.app.to_play), is_candidate=True)
        prev_visits = self.app.candidate_results.get(key, (None, None))[1]
        base_visits = prev_visits or 0
        self.app.candidate_run = CandidateRun(
            key=key,
            base_visits=base_visits,
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

            child = self.engine.get_analysis()
            winrate, visits = self.aggregate_child_analysis(child)
            run = self.app.candidate_run
            key = run.key
            prev_best = self.app.candidate_results.get(key, (None, None))[1] or 0
            if visits > prev_best:
                self.app.candidate_results[key] = (winrate, visits)
                if winrate is not None and visits > 0:
                    self._promote_candidate_to_cache(key, winrate, visits)
                self.rebuild_candidate_analysis()

            if now - run.started_at < 1.0:
                return

            target = run.target_visits
            if len(self.app.candidates) <= 1:
                return
            if visits <= target:
                return
            self.app.candidate_results[key] = (winrate, visits)
            if winrate is not None and visits > 0:
                self._promote_candidate_to_cache(key, winrate, visits)
            self._end_candidate_run()
            self.rebuild_candidate_analysis()

    def get_active_analysis(self) -> List[AnalysisMove]:
        # Prefer the most informative analysis (cache/live) while candidates can upgrade cached winrate/visits.
        base = self.app.analysis_cache.get(self.cache_key(), [])
        if base:
            return base
        if self.app.candidates:
            return self.app.candidate_analysis
        if self.app.analysis_running:
            live = self.engine.get_analysis()
            if live:
                return live
        return []

    def maybe_update_analysis_cache(self) -> None:
        if not self.app.analysis_running or self.app.candidates:
            return
        live = self.engine.get_analysis()
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
            self._set_analysis_mode(AnalysisMode.CANDIDATE)
        else:
            self._set_analysis_mode(AnalysisMode.LIVE)

    def stop_analysis(self) -> None:
        self.stop_candidate_search()
        self.engine.stop_analysis()

    def apply_move_to_state(self, col: int, row: int) -> bool:
        if not self.board.place(self.app.to_play, col, row):
            return False
        self.engine.clear_analysis()
        self.engine.play(self.app.to_play, col, row)
        self.app.to_play = self.flip_side(self.app.to_play)
        return True

    def apply_pass_to_state(self) -> bool:
        if not self.board.pass_move(self.app.to_play):
            return False
        self.engine.clear_analysis()
        self.engine.play(self.app.to_play, None, None)
        self.app.to_play = self.flip_side(self.app.to_play)
        return True

    def _assert_never(self, value: MoveKind) -> NoReturn:
        raise AssertionError(f"Unhandled move kind: {value}")

    def move_coords(self, mv: Move) -> Optional[Tuple[int, int]]:
        match mv.kind:
            case MoveKind.PLACE:
                return (mv.col, mv.row)
            case MoveKind.PASS:
                return None
        return self._assert_never(mv.kind)

    def apply_move_to_state_from_move(self, mv: Move) -> bool:
        match mv.kind:
            case MoveKind.PLACE:
                return self.apply_move_to_state(mv.col, mv.row)
            case MoveKind.PASS:
                return self.apply_pass_to_state()
        return self._assert_never(mv.kind)

    def play_move_on_engine(self, mv: Move) -> None:
        coords = self.move_coords(mv)
        if coords is None:
            self.engine.play(mv.side, None, None)
            return
        col, row = coords
        self.engine.play(mv.side, col, row)

    def rebuild_engine_from_history(self) -> None:
        self.engine.clear_board()
        for mv in self.board.history:
            self.play_move_on_engine(mv)

    def find_history_index(self, col: int, row: int) -> Optional[int]:
        for idx, mv in enumerate(self.board.history):
            coords = self.move_coords(mv)
            if coords == (col, row):
                return idx
        return None

    def truncate_future_moves_on_conflict(self) -> None:
        if not self.app.future_moves:
            return
        occupied = {self.move_coords(mv) for mv in self.board.history}
        occupied.discard(None)
        redo = list(reversed(self.app.future_moves))
        cut = len(redo)
        for i, mv in enumerate(redo):
            coords = self.move_coords(mv)
            if coords is None:
                continue
            key = coords
            if key in occupied:
                cut = i
                break
            occupied.add(key)
        if cut == len(redo):
            return
        self.app.future_moves = list(reversed(redo[:cut]))

    def with_analysis_paused(
        self, fn, *, clear_analysis: bool = False, stop_engine: bool = True
    ) -> None:
        was_running = self.app.analysis_running
        if was_running and stop_engine:
            self.stop_analysis()
        if was_running and (not stop_engine):
            self.stop_candidate_search()
            self.engine.clear_analysis()
        fn()
        if clear_analysis:
            self.engine.clear_analysis()
        if was_running:
            self.check_candidate_root()
            self.resume_analysis()

    def toggle_analysis(self) -> None:
        if self.app.analysis_running:
            self._set_analysis_mode(AnalysisMode.OFF)
        else:
            if self.app.candidates:
                self._set_analysis_mode(AnalysisMode.CANDIDATE)
            else:
                self._set_analysis_mode(AnalysisMode.LIVE)

    def load_hexworld_text(self, text: str) -> bool:
        try:
            size, past_moves, future_moves_parsed, next_side = parse_hexworld_position(text)
        except Exception as exc:
            print(f"HexWorld parse failed: {exc}")
            return False

        if size < 4 or size > 42:
            print(f"HexWorld size {size} out of range (4-42).")
            return False

        seen = set()
        for mv in past_moves + future_moves_parsed:
            coords = self.move_coords(mv)
            if coords is None:
                continue
            key = coords
            if key in seen:
                print(f"HexWorld duplicate move: {coord_to_human(*coords)}")
                return False
            seen.add(key)

        def mutate() -> None:
            self.engine.set_board_size(size)
            self.engine.clear_board()
            self.board.set_size(size)
            self.app.future_moves.clear()
            self.app.pending_size = size
            self.clear_all_cached_analysis()

            for mv in past_moves:
                match mv.kind:
                    case MoveKind.PLACE:
                        self.board.place(mv.side, mv.col, mv.row)
                    case MoveKind.PASS:
                        self.board.pass_move(mv.side)
                    case _:
                        self._assert_never(mv.kind)
                self.play_move_on_engine(mv)

            self.app.future_moves.extend(reversed(future_moves_parsed))
            self.app.to_play = next_side

        self.with_analysis_paused(mutate, clear_analysis=self.app.analysis_running)
        return True

    def move_to_label(self, mv: Move) -> str:
        match mv.kind:
            case MoveKind.PLACE:
                return coord_to_human(mv.col, mv.row)
            case MoveKind.PASS:
                return "pass"
        return self._assert_never(mv.kind)

    def _move_to_hexworld(self, mv: Move) -> str:
        match mv.kind:
            case MoveKind.PLACE:
                return coord_to_human(mv.col, mv.row)
            case MoveKind.PASS:
                return ":p"
        return self._assert_never(mv.kind)

    def build_hexworld_url(self) -> str:
        prefix = f"{self.board.n}c1"
        past = "".join(self._move_to_hexworld(mv) for mv in self.board.history)
        if not self.app.future_moves:
            url = f"https://hexworld.org/board/#{prefix}"
            if past:
                url = f"{url},{past}"
            return url

        future = "".join(self._move_to_hexworld(mv) for mv in reversed(self.app.future_moves))
        return f"https://hexworld.org/board/#{prefix},{past},{future}"

    def add_candidate(self, col: int, row: int) -> None:
        if not self.board.is_empty(col, row):
            return
        key = (col, row)
        if key in self.app.candidates:
            return
        if not self.app.candidates:
            self.app.candidate_root_rev = self.board.rev
        self.app.candidates.add(key)
        self.rebuild_candidate_analysis()
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
                self._set_analysis_mode(AnalysisMode.LIVE)
        else:
            self.rebuild_candidate_analysis()

    def remove_candidate(self, col: int, row: int) -> None:
        key = (col, row)
        if key in self.app.candidates:
            self.toggle_candidate(col, row)

    def new_game(self) -> None:
        def mutate() -> None:
            self.engine.clear_board()
            self.board.clear()
            self.app.future_moves.clear()
            self.app.to_play = Side.RED
            self.clear_all_cached_analysis()

        self.with_analysis_paused(mutate, stop_engine=False)

    def undo_one(self) -> bool:
        if self.board.undo():
            self.engine.undo()
            self.app.to_play = self.flip_side(self.app.to_play)
            return True
        return False

    def step_back(self) -> bool:
        if not self.board.history:
            return False
        last = self.board.history[-1]
        did = False

        def mutate() -> None:
            nonlocal did
            did = self.undo_one()

        self.with_analysis_paused(
            mutate, clear_analysis=self.app.analysis_running, stop_engine=False
        )
        if did:
            self.app.future_moves.append(last)
        return did

    def step_forward(self) -> bool:
        if not self.app.future_moves:
            return False

        mv = self.app.future_moves[-1]
        did = False

        def mutate() -> None:
            nonlocal did
            if not self.apply_move_to_state_from_move(mv):
                return
            did = True

        self.with_analysis_paused(mutate, stop_engine=False)
        if did:
            self.app.future_moves.pop()
        return did

    def go_first(self) -> bool:
        if not self.board.history:
            return False
        did = False

        def mutate() -> None:
            nonlocal did
            while self.board.history:
                last = self.board.history[-1]
                if not self.undo_one():
                    break
                self.app.future_moves.append(last)
                did = True

        self.with_analysis_paused(
            mutate, clear_analysis=self.app.analysis_running, stop_engine=False
        )
        return did

    def go_last(self) -> bool:
        if not self.app.future_moves:
            return False
        did = False

        def mutate() -> None:
            nonlocal did
            while self.app.future_moves:
                mv = self.app.future_moves[-1]
                if not self.apply_move_to_state_from_move(mv):
                    break
                self.app.future_moves.pop()
                did = True

        self.with_analysis_paused(mutate, stop_engine=False)
        return did

    def delete_tail(self) -> bool:
        if self.app.future_moves:
            self.app.future_moves.clear()
            return True
        if not self.board.history:
            return False

        did = False

        def mutate() -> None:
            nonlocal did
            did = self.undo_one()

        self.with_analysis_paused(
            mutate, clear_analysis=self.app.analysis_running, stop_engine=False
        )
        if did:
            self.clear_cached_analysis_from_ply(len(self.board.history), keep_current=True)
        return did

    def try_play_move(self, col: int, row: int) -> bool:
        if self.app.future_moves:
            mv = self.app.future_moves[-1]
            coords = self.move_coords(mv)
            if coords == (col, row):
                return self.step_forward()

        did = False

        def mutate() -> None:
            nonlocal did
            if not self.apply_move_to_state(col, row):
                return

            if self.app.future_moves:
                self.clear_cached_analysis_from_ply(len(self.board.history), keep_current=False)
                self.app.future_moves.clear()

            did = True

        self.with_analysis_paused(mutate, stop_engine=False)
        return did

    def try_pass_move(self) -> bool:
        if self.app.future_moves:
            mv = self.app.future_moves[-1]
            if self.move_coords(mv) is None:
                return self.step_forward()

        did = False

        def mutate() -> None:
            nonlocal did
            if not self.apply_pass_to_state():
                return
            if self.app.future_moves:
                self.clear_cached_analysis_from_ply(len(self.board.history), keep_current=False)
                self.app.future_moves.clear()
            did = True

        self.with_analysis_paused(mutate, stop_engine=False)
        return did

    def try_drag_move(self, idx: int, src: Tuple[int, int], col: int, row: int) -> bool:
        did = False

        def mutate() -> None:
            nonlocal did
            if idx < 0 or idx >= len(self.board.history):
                return
            mv = self.board.history[idx]
            coords = self.move_coords(mv)
            if coords != src:
                return
            if not self.board.move_in_history(idx, col, row):
                return
            self.clear_all_cached_analysis()
            self.truncate_future_moves_on_conflict()
            self.rebuild_engine_from_history()
            did = True

        self.with_analysis_paused(
            mutate, clear_analysis=self.app.analysis_running, stop_engine=False
        )
        return did

    def get_top_move(self) -> Tuple[Optional[Tuple[int, int]], int]:
        for r in self.get_active_analysis():
            if r.col is None or r.row is None:
                continue
            if self.board.is_empty(r.col, r.row):
                v = 0 if r.visits is None else r.visits
                return (r.col, r.row), max(1, v)
        return None, 1

    def apply_pending_size(self) -> bool:
        if self.app.pending_size == self.board.n:
            return False

        def mutate() -> None:
            self.engine.set_board_size(self.app.pending_size)
            self.engine.clear_board()
            self.board.set_size(self.app.pending_size)
            self.app.future_moves.clear()
            self.app.to_play = Side.RED
            self.clear_all_cached_analysis()

        self.with_analysis_paused(mutate, clear_analysis=self.app.analysis_running)
        return True

    def tick(self, now: float) -> None:
        if self.check_candidate_root():
            self.resume_analysis()
        self.step_candidate_search(now)
        self.maybe_update_analysis_cache()
