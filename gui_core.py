from __future__ import annotations

import logging
from typing import List, NoReturn, Optional, Sequence, Tuple

from board import HexBoard, Move, MoveKind, Side, coord_to_human
from engine import KataHexEngine
from gui_core_analysis import AppState, GuiCoreAnalysisMixin
from hexworld import parse_hexworld_position

logger = logging.getLogger(__name__)


class GuiCore(GuiCoreAnalysisMixin):
    def __init__(self, board: HexBoard, engine: KataHexEngine, *, analyze_interval_cs: int = 15) -> None:
        self.board = board
        self.engine = engine
        self.analyze_interval_cs = analyze_interval_cs

        self.app = AppState(
            pending_size=board.n,
            analysis_enabled=False,
            future_moves=[],
            candidates=set(),
            candidate_results={},
            candidate_run=None,
            candidate_ratio=1.6,
            candidate_root_rev=None,
            batch_run=None,
            analysis_cache={},
            last_cache_sig=None,
            analysis_wide_root_noise=0.04,
        )

    def apply_move_to_state(self, col: int, row: int) -> bool:
        side = self.current_side()
        if not self.board.place(side, col, row):
            return False
        self.engine.clear_analysis()
        self.play_engine_mapped(side, col, row)
        return True

    def apply_pass_to_state(self) -> bool:
        side = self.current_side()
        if not self.board.pass_move(side):
            return False
        self.engine.clear_analysis()
        self.play_engine_mapped(side, None, None)
        return True

    def apply_swap_to_state(self) -> bool:
        side = self.current_side()
        if not self.board.swap_move(side):
            return False
        self.engine.clear_analysis()
        return True

    def _assert_never(self, value: MoveKind) -> NoReturn:
        raise AssertionError(f"Unhandled move kind: {value}")

    @staticmethod
    def _apply_move_to_board_model(board: HexBoard, mv: Move) -> bool:
        match mv.kind:
            case MoveKind.PLACE:
                return board.place(mv.side, mv.col, mv.row)
            case MoveKind.PASS:
                return board.pass_move(mv.side)
            case MoveKind.SWAP:
                return board.swap_move(mv.side)
        raise AssertionError(f"Unhandled move kind: {mv.kind}")

    @staticmethod
    def _first_illegal_move_index(board: HexBoard, moves: Sequence[Move]) -> Optional[int]:
        for i, mv in enumerate(moves):
            if not GuiCore._apply_move_to_board_model(board, mv):
                return i
        return None

    @staticmethod
    def _copy_board_state(board: HexBoard) -> HexBoard:
        out = HexBoard(board.n)
        out.rev = board.rev
        out.occ = list(board.occ)
        out.history = list(board.history)
        return out

    @staticmethod
    def _opening_token_coords(moves: Sequence[Move]) -> Optional[Tuple[int, int]]:
        if len(moves) < 2:
            return None
        first, second = moves[0], moves[1]
        if first.kind != MoveKind.PLACE or second.kind != MoveKind.SWAP:
            return None
        if second.col is None or second.row is None:
            return None
        return (second.col, second.row)

    def move_coords(self, mv: Move) -> Optional[Tuple[int, int]]:
        match mv.kind:
            case MoveKind.PLACE:
                return (mv.col, mv.row)
            case MoveKind.PASS:
                return None
            case MoveKind.SWAP:
                return None
        return self._assert_never(mv.kind)

    def apply_move_to_state_from_move(self, mv: Move) -> bool:
        match mv.kind:
            case MoveKind.PLACE:
                return self.apply_move_to_state(mv.col, mv.row)
            case MoveKind.PASS:
                return self.apply_pass_to_state()
            case MoveKind.SWAP:
                return self.apply_swap_to_state()
        return self._assert_never(mv.kind)

    def play_move_on_engine(self, mv: Move) -> None:
        match mv.kind:
            case MoveKind.PLACE:
                col, row = mv.col, mv.row
                self.play_engine_mapped(mv.side, col, row)
                return
            case MoveKind.PASS:
                self.play_engine_mapped(mv.side, None, None)
                return
            case MoveKind.SWAP:
                return
        self._assert_never(mv.kind)

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

        redo = list(reversed(self.app.future_moves))
        probe = self._copy_board_state(self.board)
        cut = self._first_illegal_move_index(probe, redo)
        if cut is None:
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

    def load_hexworld_text(self, text: str) -> bool:
        try:
            size, past_moves, future_moves_parsed, _next_side = parse_hexworld_position(text)
        except Exception as exc:
            logger.info("HexWorld parse failed: %s", exc)
            return False

        if size < 4 or size > 42:
            logger.info("HexWorld size %s out of range (4-42).", size)
            return False

        all_moves = past_moves + future_moves_parsed
        probe = HexBoard(size)
        illegal_index = self._first_illegal_move_index(probe, all_moves)
        if illegal_index is not None:
            mv = all_moves[illegal_index]
            if mv.kind == MoveKind.PLACE:
                logger.info("HexWorld illegal/duplicate move: %s", coord_to_human(mv.col, mv.row))
            elif mv.kind == MoveKind.SWAP:
                logger.info("HexWorld illegal swap token placement")
            return False

        def mutate() -> None:
            self.engine.set_board_size(size)
            self.engine.clear_board()
            self.board.set_size(size)
            self.app.future_moves.clear()
            self.app.pending_size = size
            self.clear_all_cached_analysis()

            for mv in past_moves:
                if not self._apply_move_to_board_model(self.board, mv):
                    raise AssertionError(f"Illegal move while loading: {mv}")
                self.play_move_on_engine(mv)

            self.app.future_moves.extend(reversed(future_moves_parsed))

        self.with_analysis_paused(mutate, clear_analysis=self.app.analysis_running)
        return True

    def move_to_label(self, mv: Move) -> str:
        match mv.kind:
            case MoveKind.PLACE:
                return coord_to_human(mv.col, mv.row)
            case MoveKind.PASS:
                return "pass"
            case MoveKind.SWAP:
                return "swap"
        return self._assert_never(mv.kind)

    def move_to_label_in_sequence(self, moves: Sequence[Move], index: int) -> str:
        mv = moves[index]
        opening = self._opening_token_coords(moves)
        if index == 0 and opening is not None:
            return coord_to_human(*opening)
        return self.move_to_label(mv)

    def is_swapped_stone_index(self, idx: int) -> bool:
        return (
            self.swap_active()
            and idx == 0
            and len(self.board.history) >= 2
            and self.board.history[1].kind == MoveKind.SWAP
        )

    def _moves_to_hexworld_stream(self, moves: Sequence[Move]) -> str:
        out: List[str] = []
        opening = self._opening_token_coords(moves)
        for idx, mv in enumerate(moves):
            if idx == 0 and opening is not None:
                out.append(coord_to_human(*opening))
                continue
            out.append(self._move_to_hexworld(mv))
        return "".join(out)

    def _move_to_hexworld(self, mv: Move) -> str:
        match mv.kind:
            case MoveKind.PLACE:
                return coord_to_human(mv.col, mv.row)
            case MoveKind.PASS:
                return ":p"
            case MoveKind.SWAP:
                return ":s"
        return self._assert_never(mv.kind)

    def build_hexworld_url(self) -> str:
        prefix = f"{self.board.n}c1"
        past = self._moves_to_hexworld_stream(self.board.history)
        if not self.app.future_moves:
            url = f"https://hexworld.org/board/#{prefix}"
            if past:
                url = f"{url},{past}"
            return url

        future_moves = list(reversed(self.app.future_moves))
        future = self._moves_to_hexworld_stream(future_moves)
        return f"https://hexworld.org/board/#{prefix},{past},{future}"

    def new_game(self) -> None:
        def mutate() -> None:
            self.engine.clear_board()
            self.board.clear()
            self.app.future_moves.clear()
            self.clear_all_cached_analysis()

        self.with_analysis_paused(mutate, stop_engine=False)

    def undo_one(self) -> bool:
        if not self.board.history:
            return False
        last = self.board.history[-1]
        if self.board.undo():
            if last.kind != MoveKind.SWAP:
                self.engine.undo()
            return True
        return False

    def step_back(self) -> bool:
        return self.step_back_n(1)

    def step_forward(self) -> bool:
        return self.step_forward_n(1)

    def step_back_n(self, count: int) -> bool:
        if count <= 0 or not self.board.history:
            return False
        did = False

        def mutate() -> None:
            nonlocal did
            for _ in range(count):
                if not self.board.history:
                    break
                last = self.board.history[-1]
                if not self.undo_one():
                    break
                self.app.future_moves.append(last)
                did = True

        self.with_analysis_paused(
            mutate, clear_analysis=self.app.analysis_running, stop_engine=False
        )
        return did

    def step_forward_n(self, count: int) -> bool:
        if count <= 0 or not self.app.future_moves:
            return False
        did = False

        def mutate() -> None:
            nonlocal did
            for _ in range(count):
                if not self.app.future_moves:
                    break
                mv = self.app.future_moves[-1]
                if not self.apply_move_to_state_from_move(mv):
                    break
                self.app.future_moves.pop()
                did = True

        self.with_analysis_paused(mutate, stop_engine=False)
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
        return did

    def try_play_moves(self, moves: Sequence[Tuple[int, int]]) -> bool:
        if not moves:
            return False
        did = False

        def mutate() -> None:
            nonlocal did
            remaining = list(moves)
            while remaining and self.app.future_moves:
                mv = self.app.future_moves[-1]
                coords = self.move_coords(mv)
                if coords != remaining[0]:
                    break
                if not self.apply_move_to_state_from_move(mv):
                    return
                self.app.future_moves.pop()
                remaining.pop(0)
                did = True

            if not remaining:
                return

            if self.app.future_moves:
                self.app.future_moves.clear()

            for col, row in remaining:
                if not self.apply_move_to_state(col, row):
                    return
                did = True

        self.with_analysis_paused(mutate, stop_engine=False)
        return did

    def try_play_move(self, col: int, row: int) -> bool:
        return self.try_play_moves([(col, row)])

    def can_swap_move(self) -> bool:
        if len(self.board.history) != 1:
            return False
        first = self.board.history[0]
        if first.kind != MoveKind.PLACE:
            return False
        if first.side != Side.RED:
            return False
        return self.current_side() == Side.BLUE

    def try_pass_move(self) -> bool:
        if self.app.future_moves:
            mv = self.app.future_moves[-1]
            if mv.kind == MoveKind.PASS:
                return self.step_forward()

        did = False

        def mutate() -> None:
            nonlocal did
            if not self.apply_pass_to_state():
                return
            if self.app.future_moves:
                self.app.future_moves.clear()
            did = True

        self.with_analysis_paused(mutate, stop_engine=False)
        return did

    def try_swap_move(self) -> bool:
        if self.app.future_moves:
            mv = self.app.future_moves[-1]
            if mv.kind == MoveKind.SWAP:
                return self.step_forward()

        did = False

        def mutate() -> None:
            nonlocal did
            if not self.apply_swap_to_state():
                return
            if self.app.future_moves:
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
            if idx == 0 and self.app.future_moves and self.app.future_moves[-1].kind == MoveKind.SWAP:
                swap_mv = self.app.future_moves[-1]
                self.app.future_moves[-1] = Move.swap(side=swap_mv.side, col=col, row=row)
            self.clear_all_cached_analysis()
            self.truncate_future_moves_on_conflict()
            self.rebuild_engine_from_history()
            did = True

        self.with_analysis_paused(
            mutate, clear_analysis=self.app.analysis_running, stop_engine=False
        )
        return did

    def apply_pending_size(self) -> bool:
        if self.app.pending_size == self.board.n:
            return False

        def mutate() -> None:
            self.engine.set_board_size(self.app.pending_size)
            self.engine.clear_board()
            self.board.set_size(self.app.pending_size)
            self.app.future_moves.clear()
            self.clear_all_cached_analysis()

        self.with_analysis_paused(mutate, clear_analysis=self.app.analysis_running)
        return True
