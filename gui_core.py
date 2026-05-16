from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NoReturn, Optional, Sequence, Tuple

from board import MAX_BOARD_SIZE, MIN_BOARD_SIZE, HexBoard, Move, MoveKind, Side, coord_to_human
from engine import KataHexEngine
from gui_core_analysis import AnalysisModeTag, AppState, CandidateState, GuiCoreAnalysisMixin
from history_tree import HistoryNode, MoveTree
import hexata_format
import hexworld
DEFAULT_ANALYZE_INTERVAL_CS = 15

EditSnapshot = tuple[
    MoveTree,
    tuple[Tuple[int, int], ...],
    dict[Tuple[int, int], tuple[Optional[float], Optional[int]]],
]


@dataclass(slots=True)
class MovelistCell:
    # Character-cell column in the monospace movelist font.
    column: int
    label: str
    side: Side
    played: bool


@dataclass(slots=True)
class MovelistRow:
    ply: int
    cells: tuple[MovelistCell, ...]


@dataclass(slots=True)
class MovelistView:
    rows: tuple[MovelistRow, ...]
    focus_row: int


@dataclass(slots=True)
class _MovelistPlacement:
    row: int
    cell: MovelistCell

    @property
    def end_column(self) -> int:
        return self.cell.column + 1


@dataclass(slots=True)
class _MovelistSubtree:
    placements: tuple[_MovelistPlacement, ...]


@dataclass(slots=True)
class EvalGraphData:
    moves: tuple[Move, ...]
    prefix_keys: tuple[bytes, ...]


class GuiCore(GuiCoreAnalysisMixin):
    def __init__(
        self,
        board: HexBoard,
        engine: KataHexEngine,
        *,
        analyze_interval_cs: int = DEFAULT_ANALYZE_INTERVAL_CS,
    ) -> None:
        self.board = board
        self.engine = engine
        self.analyze_interval_cs = analyze_interval_cs
        self.tree = MoveTree()

        self.app = AppState(
            pending_size=board.n,
            candidate_state=CandidateState(
                candidates=set(),
                results={},
                run=None,
                ratio=1.6,
                root_key=None,
            ),
            analysis_cache={},
            root_eval_cache={},
            last_cache_sig=None,
            analysis_wide_root_noise=0.04,
        )
        self.edit_undo: list[EditSnapshot] = []
        self.edit_redo: list[EditSnapshot] = []

    # -------------------- tree accessors --------------------
    # The move tree is the canonical logical history. `board.history` is only the
    # currently applied/materialized path, which differs after swap because move 1
    # is transposed on the board.
    def applied_history(self) -> Sequence[Move]:
        return self.board.history

    def current_path_moves(self) -> list[Move]:
        return self.tree.current_path_moves()

    def mainline_tail_moves(self) -> list[Move]:
        return self.tree.mainline_tail_moves()

    def visible_line_moves(self) -> list[Move]:
        return self.tree.visible_line_moves()

    def current_ply(self) -> int:
        return self.tree.current_ply()

    def next_mainline_move(self) -> Optional[Move]:
        return self.tree.next_mainline_move()

    def next_variation_moves(self) -> list[Move]:
        return self.tree.variation_moves()

    # -------------------- edit history --------------------
    def _capture_edit_state(self) -> EditSnapshot:
        state = self.app.candidate_state
        return (
            self.tree.clone(),
            tuple(sorted(state.candidates)),
            dict(state.results),
        )

    def _clear_edit_history(self) -> None:
        self.edit_undo.clear()
        self.edit_redo.clear()

    def _edit_state_sig(self) -> tuple[tuple, tuple[Tuple[int, int], ...]]:
        return (self.tree.signature(), tuple(sorted(self.app.candidate_state.candidates)))

    def _run_tracked_edit(
        self,
        mutate: Callable[[], None],
        *,
        resume_after: bool = True,
    ) -> bool:
        before = self._capture_edit_state()
        before_sig = self._edit_state_sig()
        self.with_analysis_keep_engine_synced(mutate, resume_after=resume_after)
        if self._edit_state_sig() == before_sig:
            return False
        self.edit_undo.append(before)
        self.edit_redo.clear()
        return True

    def _rebuild_board_from_moves(self, moves: Sequence[Move]) -> None:
        self.board.clear()
        for mv in moves:
            if not self.board.apply_move(mv):
                raise AssertionError(f"Illegal move while rebuilding board: {mv}")

    def _rebuild_position_from_tree(self) -> None:
        self._rebuild_board_from_moves(self.current_path_moves())
        self.rebuild_engine_from_applied_history()

    def _restore_edit_state(self, snap: EditSnapshot) -> None:
        tree, candidates, results = snap

        def mutate() -> None:
            self.tree = tree.clone()
            state = self.app.candidate_state
            state.candidates.clear()
            state.candidates.update(candidates)
            state.results.clear()
            state.results.update(results)
            state.run = None
            self._rebuild_position_from_tree()
            state.root_key = self.cache_key() if state.candidates else None
            if self.app.analysis_mode != AnalysisModeTag.OFF:
                self.app.analysis_mode = AnalysisModeTag.LIVE

        self.with_analysis_stopped(mutate)

    def undo_edit(self) -> bool:
        if not self.edit_undo:
            return False
        target = self.edit_undo.pop()
        current = self._capture_edit_state()
        self._restore_edit_state(target)
        self.edit_redo.append(current)
        return True

    def redo_edit(self) -> bool:
        if not self.edit_redo:
            return False
        target = self.edit_redo.pop()
        current = self._capture_edit_state()
        self._restore_edit_state(target)
        self.edit_undo.append(current)
        return True

    # -------------------- board and engine sync --------------------
    def apply_move_to_state(self, col: int, row: int) -> bool:
        side = self.current_side()
        if not self.board.place(side, col, row):
            return False
        self.play_engine_mapped(side, col, row)
        return True

    def apply_pass_to_state(self) -> bool:
        side = self.current_side()
        if not self.board.pass_move(side):
            return False
        self.play_engine_mapped(side, None, None)
        return True

    def apply_swap_to_state(self) -> bool:
        side = self.current_side()
        if not self.board.swap_move(side):
            return False
        return True

    def _assert_never(self, value: MoveKind) -> NoReturn:
        raise AssertionError(f"Unhandled move kind: {value}")

    @staticmethod
    def _first_illegal_move_index(board: HexBoard, moves: Sequence[Move]) -> Optional[int]:
        for i, mv in enumerate(moves):
            if not board.apply_move(mv):
                return i
        return None

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

    def _follow_existing_tree_move(self, mv: Move, *, promote: bool) -> bool:
        if not self.apply_move_to_state_from_move(mv):
            return False
        if not self.tree.follow_child(mv, promote=promote):
            raise AssertionError("Tree follow_child failed for existing move")
        return True

    def play_move_on_engine(self, mv: Move) -> None:
        match mv.kind:
            case MoveKind.PLACE:
                self.play_engine_mapped(mv.side, mv.col, mv.row)
                return
            case MoveKind.PASS:
                self.play_engine_mapped(mv.side, None, None)
                return
            case MoveKind.SWAP:
                return
        self._assert_never(mv.kind)

    def rebuild_engine_from_applied_history(self) -> None:
        self.engine.clear_board()
        for mv in self.applied_history():
            self.play_move_on_engine(mv)

    def find_applied_move_index(self, col: int, row: int) -> Optional[int]:
        for idx, mv in enumerate(self.applied_history()):
            coords = self.move_coords(mv)
            if coords == (col, row):
                return idx
        return None

    def _with_analysis_paused(
        self,
        fn: Callable[[], None],
        *,
        keep_engine_synced: bool,
        resume_after: bool = True,
    ) -> None:
        was_running = self.app.analysis_enabled
        was_batch = self.is_batch_analysis_active()
        rev_before = self.board.rev
        tree_sig_before = self.tree.signature() if was_batch and resume_after else None
        if was_running:
            if keep_engine_synced:
                self.stop_candidate_run()
                self.engine.cancel_reply_capture()
                self.engine.clear_analysis()
            else:
                self.stop_analysis()
        fn()
        self.engine.clear_analysis()
        self.check_candidate_root()
        if was_running and resume_after:
            # User-driven navigation/edits should exit batch immediately once
            # they actually change the selected tree or materialized position.
            if was_batch and (self.board.rev != rev_before or self.tree.signature() != tree_sig_before):
                self.app.analysis_mode = AnalysisModeTag.LIVE
            self.resume_analysis()

    def with_analysis_stopped(self, fn: Callable[[], None], *, resume_after: bool = True) -> None:
        self._with_analysis_paused(fn, keep_engine_synced=False, resume_after=resume_after)

    def with_analysis_keep_engine_synced(
        self, fn: Callable[[], None], *, resume_after: bool = True
    ) -> None:
        self._with_analysis_paused(fn, keep_engine_synced=True, resume_after=resume_after)

    # -------------------- import and export --------------------
    def _import_size_error(self, size: int, *, source: str) -> Optional[str]:
        if MIN_BOARD_SIZE <= size <= MAX_BOARD_SIZE:
            return None
        return f"{source} size {size} out of range ({MIN_BOARD_SIZE}-{MAX_BOARD_SIZE})."

    @staticmethod
    def _parse_hexata_size_prefix(text: str) -> Optional[int]:
        i = 0
        while i < len(text) and text[i].isdigit():
            i += 1
        if i == 0 or i >= len(text) or text[i] != ",":
            return None
        try:
            return int(text[:i])
        except ValueError:
            return None

    def _install_imported_tree(self, size: int, tree: MoveTree) -> None:
        def mutate() -> None:
            self.engine.set_board_size(size)
            self.board.set_size(size)
            self.tree = tree
            self.app.pending_size = size
            self.clear_candidates()
            self.clear_all_cached_analysis()
            self._rebuild_position_from_tree()

        self.with_analysis_stopped(mutate)
        self._clear_edit_history()

    def load_hexworld_text(self, text: str) -> Optional[str]:
        try:
            size, past_moves, future_moves_parsed, _next_side = hexworld.parse_hexworld_position(text)
        except Exception as exc:
            return f"HexWorld parse failed: {exc}"

        size_error = self._import_size_error(size, source="HexWorld")
        if size_error is not None:
            return size_error

        all_moves = past_moves + future_moves_parsed
        probe = HexBoard(size)
        illegal_index = self._first_illegal_move_index(probe, all_moves)
        if illegal_index is not None:
            mv = all_moves[illegal_index]
            if mv.kind == MoveKind.PLACE:
                return f"HexWorld illegal/duplicate move: {coord_to_human(mv.col, mv.row)}"
            if mv.kind == MoveKind.SWAP:
                return "HexWorld illegal swap token placement"
            return "HexWorld illegal move"

        tree = MoveTree()
        tree.rebuild_from_line(past_moves, future_moves_parsed)
        self._install_imported_tree(size, tree)
        return None

    def move_to_label(self, mv: Move) -> str:
        match mv.kind:
            case MoveKind.PLACE:
                return coord_to_human(mv.col, mv.row)
            case MoveKind.PASS:
                return "pass"
            case MoveKind.SWAP:
                return "swap"
        return self._assert_never(mv.kind)

    def is_swapped_stone_index(self, idx: int) -> bool:
        moves = self.current_path_moves()
        return self.swap_active() and idx == 0 and len(moves) >= 2 and moves[1].kind == MoveKind.SWAP

    def build_hexworld_url(self) -> str:
        return hexworld.build_hexworld_url(
            self.board.n,
            self.current_path_moves(),
            self.mainline_tail_moves(),
        )

    def build_hexata_format(self) -> str:
        return hexata_format.build_hexata_format(self.board.n, self.tree)

    def load_hexata_format(self, text: str) -> Optional[str]:
        prefixed_size = self._parse_hexata_size_prefix(text)
        if prefixed_size is not None:
            prefixed_error = self._import_size_error(prefixed_size, source="Hexata format")
            if prefixed_error is not None:
                return prefixed_error
        try:
            size, tree = hexata_format.parse_hexata_format(text)
        except ValueError as exc:
            return f"Hexata format parse failed: {exc}"

        size_error = self._import_size_error(size, source="Hexata format")
        if size_error is not None:
            return size_error

        self._install_imported_tree(size, tree)
        return None

    # -------------------- move navigation and editing --------------------
    def new_game(self) -> None:
        def mutate() -> None:
            self.engine.clear_board()
            self.board.clear()
            self.tree.clear()
            self.clear_candidates()
            self.clear_all_cached_analysis()

        self.with_analysis_keep_engine_synced(mutate)
        self._clear_edit_history()

    def undo_one(self) -> bool:
        moves = self.current_path_moves()
        if not moves:
            return False
        last = moves[-1]
        if not self.board.undo():
            return False
        if not self.tree.step_back():
            raise AssertionError("Tree step_back failed for non-root path")
        if last.kind != MoveKind.SWAP:
            self.engine.undo()
        return True

    def _step_forward_one(self) -> bool:
        mv = self.next_mainline_move()
        if mv is None:
            return False
        if not self.apply_move_to_state_from_move(mv):
            return False
        if not self.tree.step_forward():
            raise AssertionError("Tree step_forward failed with selected mainline child")
        return True

    def step_back(self) -> bool:
        return self.step_back_n(1)

    def step_forward(self) -> bool:
        return self.step_forward_n(1)

    def go_sibling(self, direction: int, *, resume_after: bool = True) -> bool:
        target = self.tree.sibling_cursor(direction)
        if target is None:
            return False
        did = False

        def mutate() -> None:
            nonlocal did
            self.tree.cursor = target
            self._rebuild_position_from_tree()
            did = True

        self.with_analysis_keep_engine_synced(mutate, resume_after=resume_after)
        return did

    def step_back_n(self, count: int, *, resume_after: bool = True) -> bool:
        if count <= 0 or self.current_ply() <= 0:
            return False
        did = False

        def mutate() -> None:
            nonlocal did
            for _ in range(count):
                if not self.undo_one():
                    break
                did = True

        self.with_analysis_keep_engine_synced(mutate, resume_after=resume_after)
        return did

    def step_forward_n(self, count: int, *, resume_after: bool = True) -> bool:
        if count <= 0 or self.next_mainline_move() is None:
            return False
        did = False

        def mutate() -> None:
            nonlocal did
            for _ in range(count):
                if not self._step_forward_one():
                    break
                did = True

        self.with_analysis_keep_engine_synced(mutate, resume_after=resume_after)
        return did

    def go_first(self, *, resume_after: bool = True) -> bool:
        return self.step_back_n(self.current_ply(), resume_after=resume_after)

    def go_last(self) -> bool:
        return self.step_forward_n(len(self.mainline_tail_moves()))

    def _delete_current_leaf(self) -> bool:
        moves = self.current_path_moves()
        if not moves:
            return False
        last = moves[-1]
        if not self.board.undo():
            return False
        if not self.tree.delete_cursor_node():
            return False
        if last.kind != MoveKind.SWAP:
            self.engine.undo()
        return True

    def delete_tail(self) -> bool:
        before = self._capture_edit_state()
        before_sig = self._edit_state_sig()
        if self.tree.delete_selected_tail():
            if self._edit_state_sig() == before_sig:
                return False
            self.edit_undo.append(before)
            self.edit_redo.clear()
            if self.is_batch_analysis_active():
                self.cancel_batch_analysis()
            return True
        if self.current_ply() <= 0:
            return False

        def mutate() -> None:
            self._delete_current_leaf()

        return self._run_tracked_edit(mutate)

    def _play_move_into_tree(self, mv: Move) -> bool:
        next_mv = self.next_mainline_move()
        if next_mv == mv:
            return self._step_forward_one()
        if self.tree.find_child(self.tree.cursor, mv) is not None:
            return self._follow_existing_tree_move(mv, promote=False)
        if not self.apply_move_to_state_from_move(mv):
            return False
        self.tree.add_or_select_child(mv, promote=False)
        return True

    def try_play_moves(self, moves: Sequence[Tuple[int, int]]) -> bool:
        if not moves:
            return False

        def mutate() -> None:
            for col, row in moves:
                side = self.current_side()
                if not self._play_move_into_tree(Move.place(side=side, col=col, row=row)):
                    return

        return self._run_tracked_edit(mutate)

    def try_play_move(self, col: int, row: int) -> bool:
        return self.try_play_moves([(col, row)])

    def can_swap_move(self) -> bool:
        moves = self.current_path_moves()
        if len(moves) != 1:
            return False
        first = moves[0]
        if first.kind != MoveKind.PLACE:
            return False
        if first.side != Side.RED:
            return False
        return self.current_side() == Side.BLUE

    def try_pass_move(self) -> bool:
        def mutate() -> None:
            side = self.current_side()
            self._play_move_into_tree(Move.pass_(side=side))

        return self._run_tracked_edit(mutate)

    def _swap_child_move(self) -> Move:
        first = self.current_path_moves()[0]
        return Move.swap(side=self.current_side(), col=first.col, row=first.row)

    def try_swap_move(self) -> bool:
        if not self.can_swap_move():
            return False

        def mutate() -> None:
            self._play_move_into_tree(self._swap_child_move())

        return self._run_tracked_edit(mutate)

    def _update_swap_children_for_opening(self, node: HistoryNode, col: int, row: int) -> None:
        for child in node.children:
            if child.move.kind != MoveKind.SWAP:
                continue
            child.move = Move.swap(side=child.move.side, col=col, row=row)

    def _prune_invalid_descendants(self, node: HistoryNode, board: HexBoard) -> None:
        stack: list[tuple[HistoryNode, HexBoard]] = [(node, board)]
        while stack:
            parent, parent_board = stack.pop()
            for child in list(parent.children):
                probe = parent_board.copy()
                if not probe.apply_move(child.move):
                    parent.remove_child(child)
                    continue
                stack.append((child, probe))

    def try_drag_move(self, idx: int, src: Tuple[int, int], col: int, row: int) -> bool:
        path_nodes = self.tree.current_path_nodes()
        if idx < 0 or idx >= len(path_nodes):
            return False
        mv = self.applied_history()[idx]
        if self.move_coords(mv) != src:
            return False
        # Only drag onto currently empty cells; future tail conflicts can be
        # pruned after the edited prefix is rebuilt, but applied stones stay fixed.
        if not self.board.is_empty(col, row):
            return False

        edit_node = path_nodes[idx]
        if edit_node.move.kind != MoveKind.PLACE:
            return False

        target_col, target_row = (row, col) if idx == 0 and self.swap_active() else (col, row)
        new_move = Move.place(side=edit_node.move.side, col=target_col, row=target_row)
        parent = edit_node.parent
        existing = None if parent is None else self.tree.find_child(parent, new_move)

        probe = HexBoard(self.board.n)
        prefix_moves = [node.move for node in path_nodes[:idx]] + [new_move]
        if self._first_illegal_move_index(probe, prefix_moves) is not None:
            return False

        def mutate() -> None:
            edit_node.move = new_move
            if idx == 0:
                # Move 1 also anchors swap children, so refresh their stored source
                # coordinate before replaying descendants against the edited prefix.
                self._update_swap_children_for_opening(edit_node, target_col, target_row)
            edited_tail_moves = [node.move for node in path_nodes[idx + 1 :]]
            merge_root = edit_node
            if existing is not None and existing is not edit_node:
                merge_root = self.tree.merge_equivalent_siblings(edit_node, existing)
            self._prune_invalid_descendants(merge_root, probe)

            cursor = merge_root
            for mv in edited_tail_moves:
                next_node = self.tree.find_child(cursor, mv)
                if next_node is None:
                    break
                cursor = next_node
            self.tree.cursor = cursor
            self._rebuild_position_from_tree()

        return self._run_tracked_edit(mutate)

    def apply_pending_size(self) -> bool:
        if self.app.pending_size == self.board.n:
            return False

        def mutate() -> None:
            self.engine.set_board_size(self.app.pending_size)
            self.engine.clear_board()
            self.board.set_size(self.app.pending_size)
            self.tree.clear()
            self.clear_all_cached_analysis()

        self.with_analysis_stopped(mutate)
        self._clear_edit_history()
        return True

    # -------------------- view models --------------------
    def build_eval_graph_data(self) -> EvalGraphData:
        past_moves = list(self.applied_history())
        future_moves = self.mainline_tail_moves()
        moves = past_moves + future_moves
        keys = [self.cache_key_for_applied_moves(past_moves[:i]) for i in range(1, len(past_moves) + 1)]
        if future_moves:
            # Eval-graph prefixes follow applied board history. Already-played moves
            # come from the materialized path, and future moves are replayed from the
            # current cursor so swap-transposed prefixes stay aligned with the
            # positions the engine actually evaluates.
            probe = self.board.copy()
            for mv in future_moves:
                if not probe.apply_move(mv):
                    raise AssertionError(f"Illegal eval-graph future move: {mv}")
                keys.append(self.cache_key_for_applied_moves(probe.history))
        return EvalGraphData(moves=tuple(moves), prefix_keys=tuple(keys))

    def _make_movelist_cell(
        self,
        node: HistoryNode,
        *,
        current_path_ids: set[int],
    ) -> MovelistCell:
        mv = node.move
        if mv is None:
            raise AssertionError("History tree node missing move")
        return MovelistCell(
            column=0,
            label=self.move_to_label(mv),
            side=mv.side,
            played=node.id in current_path_ids,
        )

    @staticmethod
    def _shift_movelist_subtree(
        subtree: _MovelistSubtree,
        *,
        row_delta: int = 0,
        col_delta: int = 0,
    ) -> _MovelistSubtree:
        return _MovelistSubtree(
            placements=tuple(
                _MovelistPlacement(
                    row=placement.row + row_delta,
                    cell=MovelistCell(
                        column=placement.cell.column + col_delta,
                        label=placement.cell.label,
                        side=placement.cell.side,
                        played=placement.cell.played,
                    ),
                )
                for placement in subtree.placements
            ),
        )

    @staticmethod
    def _merge_movelist_subtrees(first: _MovelistSubtree, second: _MovelistSubtree) -> _MovelistSubtree:
        return _MovelistSubtree(
            placements=first.placements + second.placements,
        )

    @staticmethod
    def _movelist_row_right_edge(subtree: _MovelistSubtree, row: int) -> int:
        right = -1
        for placement in subtree.placements:
            if placement.row != row:
                continue
            right = max(right, placement.end_column)
        return right

    @staticmethod
    def _movelist_required_shift(
        existing: _MovelistSubtree,
        incoming: _MovelistSubtree,
        *,
        min_col: int,
    ) -> int:
        # Pack each new sibling subtree as far left as possible while keeping
        # lanes distinct on any overlapping ply row.
        required = min_col
        by_row: dict[int, list[_MovelistPlacement]] = {}
        for placement in existing.placements:
            by_row.setdefault(placement.row, []).append(placement)
        for incoming_placement in incoming.placements:
            for existing_placement in by_row.get(incoming_placement.row, []):
                required = max(
                    required,
                    existing_placement.end_column - incoming_placement.cell.column,
                )
        return required

    def _pack_movelist_subtrees(self, subtrees: Sequence[_MovelistSubtree]) -> _MovelistSubtree:
        packed = _MovelistSubtree(placements=())
        for idx, subtree in enumerate(subtrees):
            if idx == 0:
                packed = self._merge_movelist_subtrees(packed, subtree)
                continue
            shift = self._movelist_required_shift(
                packed,
                subtree,
                min_col=self._movelist_row_right_edge(packed, 0),
            )
            packed = self._merge_movelist_subtrees(
                packed,
                self._shift_movelist_subtree(subtree, col_delta=shift),
            )
        return packed

    def _build_movelist_subtree(
        self,
        node: HistoryNode,
        *,
        current_path_ids: set[int],
    ) -> _MovelistSubtree:
        built: dict[int, _MovelistSubtree] = {}
        stack: list[tuple[HistoryNode, bool]] = [(node, False)]
        while stack:
            current, expanded = stack.pop()
            if not expanded:
                stack.append((current, True))
                for child in reversed(current.children):
                    stack.append((child, False))
                continue
            if current.move is None:
                raise AssertionError("History tree node missing move")
            root = _MovelistSubtree(
                placements=(
                    _MovelistPlacement(
                        row=0,
                        cell=self._make_movelist_cell(current, current_path_ids=current_path_ids),
                    ),
                ),
            )
            if current.children:
                children = self._pack_movelist_subtrees([built[child.id] for child in current.children])
                root = self._merge_movelist_subtrees(
                    root,
                    self._shift_movelist_subtree(children, row_delta=1),
                )
            built[current.id] = root
        return built[node.id]

    def build_movelist_view(self) -> MovelistView:
        current_path_ids = {node.id for node in self.tree.current_path_nodes()}
        if not self.tree.root.children:
            return MovelistView(rows=(), focus_row=0)

        packed = self._pack_movelist_subtrees(
            [
                self._build_movelist_subtree(child, current_path_ids=current_path_ids)
                for child in self.tree.root.children
            ]
        )
        lane_widths: dict[int, int] = {}
        for placement in packed.placements:
            lane_widths[placement.cell.column] = max(
                lane_widths.get(placement.cell.column, 0),
                len(placement.cell.label),
            )
        lane_starts: dict[int, int] = {}
        lane_x = 0
        for lane in range(max(lane_widths, default=-1) + 1):
            lane_starts[lane] = lane_x
            lane_x += lane_widths.get(lane, 0) + 1
        row_cells: dict[int, list[MovelistCell]] = {}
        for placement in packed.placements:
            row_cells.setdefault(placement.row, []).append(
                MovelistCell(
                    column=lane_starts[placement.cell.column],
                    label=placement.cell.label,
                    side=placement.cell.side,
                    played=placement.cell.played,
                )
            )

        rows = tuple(
            MovelistRow(
                ply=row + 1,
                cells=tuple(sorted(row_cells.get(row, ()), key=lambda cell: cell.column)),
            )
            for row in range(max(row_cells) + 1)
        )
        return MovelistView(
            rows=rows,
            focus_row=max(0, self.current_ply() - 1),
        )
