from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NoReturn, Optional, Sequence, Tuple

from board import MAX_BOARD_SIZE, MIN_BOARD_SIZE, HexBoard, Move, MoveKind, Side, coord_to_human
from engine import KataHexEngine
from formats import flexible_moves, hexata, hexworld
from gui.analysis import GuiCoreAnalysisMixin
from gui.state import EditSnapshot, SessionState, TransitionKind
from history_tree import HistoryNode, MoveTree

DEFAULT_ANALYZE_INTERVAL_CS = 15


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

        self.session = SessionState(pending_size=board.n)

    # -------------------- tree accessors --------------------
    # The move tree is the canonical logical history. `board.history` is only the
    # currently applied/materialized path, which differs after swap because move 1
    # is transposed on the board.
    def applied_history(self) -> Sequence[Move]:
        return self.board.history

    def current_path_moves(self) -> list[Move]:
        return self.session.tree.current_path_moves()

    def mainline_tail_moves(self) -> list[Move]:
        return self.session.tree.mainline_tail_moves()

    def visible_line_moves(self) -> list[Move]:
        return self.session.tree.visible_line_moves()

    def current_ply(self) -> int:
        return self.session.tree.current_ply()

    def next_mainline_move(self) -> Optional[Move]:
        return self.session.tree.next_mainline_move()

    def next_variation_moves(self) -> list[Move]:
        return self.session.tree.variation_moves()

    # -------------------- edit history --------------------
    def _take_edit_snapshot(self) -> EditSnapshot:
        selection = self.session.candidate_selection
        # Candidates are position-local and undo/redo restores them with the board.
        # The caller replaces the live tree, transferring this instance to the stack.
        return (
            self.session.tree,
            tuple(sorted(selection.candidates)),
        )

    def _clear_edit_history(self) -> None:
        self.session.edit_undo.clear()
        self.session.edit_redo.clear()

    def _rebuild_board_from_moves(self, moves: Sequence[Move]) -> None:
        self.board.clear()
        for mv in moves:
            if not self.board.apply_move(mv):
                raise AssertionError(f"Illegal move while rebuilding board: {mv}")

    def _rebuild_position_from_tree(self) -> None:
        self._rebuild_board_from_moves(self.current_path_moves())
        self.rebuild_engine_from_applied_history()

    @staticmethod
    def _path_moves_to(tree: MoveTree, cursor: HistoryNode) -> tuple[Move, ...]:
        moves: list[Move] = []
        node = cursor
        while node.parent is not None:
            if node.move is None:
                raise AssertionError("History tree node missing move")
            moves.append(node.move)
            node = node.parent
        if node is not tree.root:
            raise AssertionError("Cursor does not belong to target tree")
        moves.reverse()
        return tuple(moves)

    @staticmethod
    def _materialize_position(moves: Sequence[Move], board_size: int) -> HexBoard:
        probe = HexBoard(board_size)
        for mv in moves:
            if not probe.apply_move(mv):
                raise AssertionError(f"Illegal move while materializing tree: {mv}")
        return probe

    def _commit_transition(
        self,
        tree: MoveTree,
        *,
        kind: TransitionKind = TransitionKind.USER_POSITION,
        cursor: Optional[HistoryNode] = None,
        board_size: Optional[int] = None,
        pending_size: Optional[int] = None,
        candidates: Optional[Sequence[Tuple[int, int]]] = None,
        clear_analysis_cache: bool = False,
        track_edit: bool = False,
    ) -> None:
        target_size = self.board.n if board_size is None else board_size
        target_cursor = tree.cursor if cursor is None else cursor
        new_path = self._path_moves_to(tree, target_cursor)
        target_board = self._materialize_position(new_path, target_size)
        if candidates is not None and any(
            not target_board.is_empty(*candidate) for candidate in candidates
        ):
            raise AssertionError("Candidate is not empty in target position")

        if track_edit and tree is self.session.tree:
            raise AssertionError("Tracked edits must install a detached tree")
        before = self._take_edit_snapshot() if track_edit else None
        old_path = tuple(self.current_path_moves())
        old_engine_moves = self._engine_position_moves()
        old_candidates = frozenset(self.session.candidate_selection.candidates)
        was_running = self.session.analysis_enabled
        was_batch = self.is_batch_analysis_active()
        size_changed = target_size != self.board.n
        path_changed = new_path != old_path
        target_candidates = old_candidates if candidates is None else frozenset(candidates)
        candidates_changed = target_candidates != old_candidates
        if size_changed and kind != TransitionKind.USER_POSITION:
            raise AssertionError("Only user position transitions may resize the session")
        if kind == TransitionKind.USER_TREE and (
            size_changed or path_changed or candidates_changed
        ):
            raise AssertionError("Tree-only transition changed position state")

        if size_changed:
            if was_running:
                self.pause_engine_analysis()
            self.engine.set_board_size(target_size)
            self.board.set_size(target_size)

        self.session.tree = tree
        tree.cursor = target_cursor
        if pending_size is not None:
            self.session.pending_size = pending_size
        if candidates is not None:
            selection = self.session.candidate_selection
            selection.candidates.clear()
            selection.candidates.update(candidates)
            selection.root_key = None
        if clear_analysis_cache:
            self.clear_all_cached_analysis()

        if size_changed:
            self._rebuild_position_from_tree()
            self.engine.clear_analysis()
            self.check_candidate_root()
            self._ensure_candidate_root()
            if was_batch:
                self._exit_batch_mode()
            if was_running:
                self.sync_analysis()
        elif kind == TransitionKind.USER_TREE:
            if was_batch:
                # Tree-only edits are user-driven even when the position is unchanged.
                self.leave_batch_for_live()
        else:
            exits_batch = kind != TransitionKind.BATCH_STEP
            analysis_changed = path_changed or candidates_changed or (was_batch and exits_batch)
            if was_running and analysis_changed:
                self.pause_engine_analysis()
            if path_changed:
                self._rebuild_board_from_moves(new_path)
                self._sync_engine_position(old_engine_moves)
                self.engine.clear_analysis()
                self.check_candidate_root()
            self._ensure_candidate_root()
            if was_batch and exits_batch:
                # User-driven changes leave batch mode; batch stepping opts out.
                self._exit_batch_mode()
            if (
                was_running
                and analysis_changed
                and kind == TransitionKind.USER_POSITION
            ):
                self.sync_analysis()

        if track_edit:
            if before is None:
                raise AssertionError("Tracked state change missing snapshot")
            self.session.edit_undo.append(before)
            self.session.edit_redo.clear()

        selection = self.session.candidate_selection
        if bool(selection.candidates) != (selection.root_key is not None):
            raise AssertionError("Candidate selection is not bound to a position")
        if selection.candidates and selection.root_key != self.cache_key():
            raise AssertionError("Candidate selection is bound to a stale position")

    def _commit_cursor(
        self,
        cursor: HistoryNode,
        *,
        kind: TransitionKind = TransitionKind.USER_POSITION,
    ) -> None:
        self._commit_transition(
            self.session.tree,
            cursor=cursor,
            kind=kind,
        )

    def _edit_tree(
        self,
        edit: Callable[[MoveTree], bool],
        *,
        kind: TransitionKind = TransitionKind.USER_POSITION,
        track_edit: bool = False,
    ) -> bool:
        tree = self.session.tree.clone()
        if not edit(tree):
            return False
        self._commit_transition(tree, kind=kind, track_edit=track_edit)
        return True

    def _restore_edit_state(self, snap: EditSnapshot) -> None:
        tree, candidates = snap
        self._commit_transition(tree, candidates=candidates)

    def undo_edit(self) -> bool:
        if not self.session.edit_undo:
            return False
        target = self.session.edit_undo.pop()
        current = self._take_edit_snapshot()
        self._restore_edit_state(target)
        self.session.edit_redo.append(current)
        return True

    def redo_edit(self) -> bool:
        if not self.session.edit_redo:
            return False
        target = self.session.edit_redo.pop()
        current = self._take_edit_snapshot()
        self._restore_edit_state(target)
        self.session.edit_undo.append(current)
        return True

    # -------------------- board and engine sync --------------------
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

    def _engine_move(self, mv: Move) -> Optional[Tuple[Side, Optional[int], Optional[int]]]:
        match mv.kind:
            case MoveKind.PLACE:
                side = self._map_side_to_engine(mv.side)
                col, row = self._map_coords_to_engine(mv.col, mv.row)
                return side, col, row
            case MoveKind.PASS:
                return self._map_side_to_engine(mv.side), None, None
            case MoveKind.SWAP:
                # Swap changes the GUI's interpretation, not the engine position.
                return None
        self._assert_never(mv.kind)

    def _engine_position_moves(self) -> tuple[Tuple[Side, Optional[int], Optional[int]], ...]:
        return tuple(
            engine_move
            for mv in self.applied_history()
            if (engine_move := self._engine_move(mv)) is not None
        )

    def _sync_engine_position(
        self,
        old_moves: Sequence[Tuple[Side, Optional[int], Optional[int]]],
    ) -> None:
        new_moves = self._engine_position_moves()
        common = 0
        while common < min(len(old_moves), len(new_moves)) and old_moves[common] == new_moves[common]:
            common += 1
        for _ in old_moves[common:]:
            self.engine.undo()
        for side, col, row in new_moves[common:]:
            self.engine.play(side, col, row)

    def rebuild_engine_from_applied_history(self) -> None:
        self.engine.clear_board()
        for side, col, row in self._engine_position_moves():
            self.engine.play(side, col, row)

    def replace_engine(self, new_engine: KataHexEngine) -> bool:
        if new_engine is self.engine:
            return False
        was_running = self.session.analysis_enabled
        if was_running:
            self.pause_engine_analysis()
        old_engine = self.engine
        self.engine = new_engine
        try:
            self.rebuild_engine_from_applied_history()
        except Exception:
            self.engine = old_engine
            new_engine.close()
            if was_running:
                self.sync_analysis()
            return False
        self.clear_all_cached_analysis()
        old_engine.close()
        # Switching engines converts any batch run to live analysis.
        self._exit_batch_mode()
        if was_running:
            self.sync_analysis()
        return True

    def find_applied_move_index(self, col: int, row: int) -> Optional[int]:
        for idx, mv in enumerate(self.applied_history()):
            coords = self.move_coords(mv)
            if coords == (col, row):
                return idx
        return None

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
        self._commit_transition(
            tree,
            board_size=size,
            pending_size=size,
            candidates=(),
            clear_analysis_cache=True,
        )
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
        return hexata.build_hexata_format(self.board.n, self.session.tree)

    def load_hexata_format(self, text: str) -> Optional[str]:
        prefixed_size = self._parse_hexata_size_prefix(text)
        if prefixed_size is not None:
            prefixed_error = self._import_size_error(prefixed_size, source="Hexata format")
            if prefixed_error is not None:
                return prefixed_error
        try:
            size, tree = hexata.parse_hexata_format(text)
        except ValueError as exc:
            return f"Hexata format parse failed: {exc}"

        size_error = self._import_size_error(size, source="Hexata format")
        if size_error is not None:
            return size_error

        self._install_imported_tree(size, tree)
        return None

    def load_flexible_move_format(self, text: str) -> Optional[str]:
        try:
            moves = flexible_moves.parse_flexible_move_format(text, board_size=self.board.n)
        except ValueError as exc:
            return f"Flexible move format parse failed: {exc}"

        tree = MoveTree()
        tree.rebuild_from_line(moves, [])
        self._install_imported_tree(self.board.n, tree)
        return None

    # -------------------- move navigation and editing --------------------
    def new_game(self) -> None:
        self._commit_transition(
            MoveTree(),
            candidates=(),
            clear_analysis_cache=True,
        )
        self._clear_edit_history()

    def step_back(self) -> bool:
        return self.step_back_n(1)

    def step_forward(self) -> bool:
        return self.step_forward_n(1)

    def go_sibling(self, direction: int) -> bool:
        target = self.session.tree.sibling_cursor(direction)
        if target is None:
            return False
        self._commit_cursor(target)
        return True

    def shift_branch(self, direction: int) -> bool:
        return self._edit_tree(
            lambda tree: tree.shift_current_branch(direction),
            kind=TransitionKind.USER_TREE,
            track_edit=True,
        )

    def _cursor_after_steps(self, count: int, *, forward: bool) -> Optional[HistoryNode]:
        node = self.session.tree.cursor
        start = node
        for _ in range(count):
            next_node = node.preferred_child if forward else node.parent
            if next_node is None:
                break
            node = next_node
        return None if node is start else node

    def step_back_n(self, count: int) -> bool:
        if count <= 0 or self.current_ply() <= 0:
            return False
        target = self._cursor_after_steps(count, forward=False)
        if target is None:
            return False
        self._commit_cursor(target)
        return True

    def step_forward_n(self, count: int) -> bool:
        if count <= 0 or self.next_mainline_move() is None:
            return False
        target = self._cursor_after_steps(count, forward=True)
        if target is None:
            return False
        self._commit_cursor(target)
        return True

    def go_first(self) -> bool:
        return self.step_back_n(self.current_ply())

    def go_last(self) -> bool:
        return self.step_forward_n(len(self.mainline_tail_moves()))

    def go_to_ply(self, ply: int) -> bool:
        max_ply = len(self.visible_line_moves())
        target = max(0, min(ply, max_ply))
        current = self.current_ply()
        if target < current:
            return self.step_back_n(current - target)
        if target > current:
            return self.step_forward_n(target - current)
        return False

    def delete_tail(self) -> bool:
        if self.next_mainline_move() is not None:
            return self._edit_tree(
                lambda tree: tree.delete_selected_tail(),
                kind=TransitionKind.USER_TREE,
                track_edit=True,
            )
        if self.current_ply() <= 0:
            return False
        return self._edit_tree(
            lambda tree: tree.delete_cursor_node(),
            track_edit=True,
        )

    @staticmethod
    def _play_move_into_tree(tree: MoveTree, mv: Move) -> bool:
        next_mv = tree.next_mainline_move()
        if next_mv == mv:
            return tree.step_forward()
        if tree.find_child(tree.cursor, mv) is not None:
            return tree.follow_child(mv)
        tree.add_or_select_child(mv)
        return True

    def _current_side_for_tree(self, tree: MoveTree) -> Side:
        moves = tree.current_path_moves()
        return Side.RED if not moves else self.flip_side(moves[-1].side)

    def try_play_moves(self, moves: Sequence[Tuple[int, int]]) -> bool:
        if not moves:
            return False
        if not self.board.is_empty(*moves[0]):
            return False

        tree = self.session.tree.clone()
        probe = self.board.copy()
        did = False
        for col, row in moves:
            side = self._current_side_for_tree(tree)
            mv = Move.place(side=side, col=col, row=row)
            if not probe.apply_move(mv) or not self._play_move_into_tree(tree, mv):
                break
            did = True
        if not did:
            return False
        self._commit_transition(tree, track_edit=True)
        return True

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
        tree = self.session.tree.clone()
        mv = Move.pass_(side=self._current_side_for_tree(tree))
        probe = self.board.copy()
        if not probe.apply_move(mv) or not self._play_move_into_tree(tree, mv):
            return False
        self._commit_transition(tree, track_edit=True)
        return True

    def _swap_child_move(self) -> Move:
        first = self.current_path_moves()[0]
        return Move.swap(side=self.current_side(), col=first.col, row=first.row)

    def try_swap_move(self) -> bool:
        if not self.can_swap_move():
            return False
        tree = self.session.tree.clone()
        mv = self._swap_child_move()
        probe = self.board.copy()
        if not probe.apply_move(mv) or not self._play_move_into_tree(tree, mv):
            return False
        self._commit_transition(tree, track_edit=True)
        return True

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
        path_nodes = self.session.tree.current_path_nodes()
        if idx < 0 or idx >= len(path_nodes):
            return False
        mv = self.applied_history()[idx]
        if self.move_coords(mv) != src:
            return False
        # Only drag onto currently empty cells; future tail conflicts can be
        # pruned after the edited prefix is rebuilt, but applied stones stay fixed.
        if not self.board.is_empty(col, row):
            return False

        if path_nodes[idx].move.kind != MoveKind.PLACE:
            return False

        target_col, target_row = (row, col) if idx == 0 and self.swap_active() else (col, row)
        new_move = Move.place(side=path_nodes[idx].move.side, col=target_col, row=target_row)
        tree = self.session.tree.clone()
        cloned_path = tree.current_path_nodes()
        edit_node = cloned_path[idx]
        parent = edit_node.parent
        existing = None if parent is None else tree.find_child(parent, new_move)

        probe = HexBoard(self.board.n)
        prefix_moves = [node.move for node in cloned_path[:idx]] + [new_move]
        if self._first_illegal_move_index(probe, prefix_moves) is not None:
            return False

        edit_node.move = new_move
        if idx == 0:
            # Move 1 also anchors swap children, so refresh their stored source
            # coordinate before replaying descendants against the edited prefix.
            self._update_swap_children_for_opening(edit_node, target_col, target_row)
        edited_tail_moves = [node.move for node in cloned_path[idx + 1 :]]
        merge_root = edit_node
        if existing is not None and existing is not edit_node:
            merge_root = tree.merge_equivalent_siblings(edit_node, existing)
        self._prune_invalid_descendants(merge_root, probe)

        cursor = merge_root
        for mv in edited_tail_moves:
            next_node = tree.find_child(cursor, mv)
            if next_node is None:
                break
            cursor = next_node
        tree.cursor = cursor
        self._commit_transition(tree, track_edit=True)
        return True

    def apply_pending_size(self) -> bool:
        if self.session.pending_size == self.board.n:
            return False
        self._commit_transition(
            MoveTree(),
            board_size=self.session.pending_size,
            candidates=(),
            clear_analysis_cache=True,
        )
        self._clear_edit_history()
        return True

    def adjust_pending_size(self, delta: int) -> None:
        self.session.pending_size = max(
            MIN_BOARD_SIZE,
            min(MAX_BOARD_SIZE, self.session.pending_size + delta),
        )

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
        current_path_ids = {node.id for node in self.session.tree.current_path_nodes()}
        if not self.session.tree.root.children:
            return MovelistView(rows=(), focus_row=0)

        packed = self._pack_movelist_subtrees(
            [
                self._build_movelist_subtree(child, current_path_ids=current_path_ids)
                for child in self.session.tree.root.children
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
