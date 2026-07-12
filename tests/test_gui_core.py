import unittest
import sys
from unittest import mock

from board import HexBoard, Move, MoveKind, Side
from engine import AnalysisMove
from gui.core import GuiCore
from gui.state import AnalysisModeTag
from formats import hexata


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.analysis = []
        self.params = {}
        self.played = []

    def clear_analysis(self):
        self.calls.append(("clear_analysis",))

    def clear_cache(self):
        self.calls.append(("clear_cache",))

    def start_analysis(self, side, interval_cs, allow_filters=()):
        if allow_filters:
            self.calls.append(("start_analysis", side, interval_cs, allow_filters))
        else:
            self.calls.append(("start_analysis", side, interval_cs))

    def stop_analysis(self):
        self.calls.append(("stop_analysis",))

    def cancel_reply_capture(self):
        return None

    def kata_set_param(self, name, value):
        self.params[name] = value
        self.calls.append(("kata_set_param", name, value))

    def play(self, side, col, row):
        self.calls.append(("play", side, col, row))
        self.played.append((side, col, row))

    def undo(self):
        self.calls.append(("undo",))
        if self.played:
            self.played.pop()

    def close(self):
        self.calls.append(("close",))

    def clear_board(self):
        self.calls.append(("clear_board",))
        self.played = []

    def set_board_size(self, n):
        self.calls.append(("set_board_size", n))

    def get_analysis(self):
        return list(self.analysis)


class RawCaptureBlockingEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.raw_capture_pending = False

    def cancel_reply_capture(self):
        self.calls.append(("cancel_reply_capture",))
        self.raw_capture_pending = False
        return None

    def start_kata_raw_nn(self, symmetry=0):
        self.calls.append(("start_kata_raw_nn", symmetry))
        if self.raw_capture_pending:
            return False
        self.raw_capture_pending = True
        return True

    def poll_kata_raw_nn(self):
        self.calls.append(("poll_kata_raw_nn",))
        return False, None


class GuiCoreTests(unittest.TestCase):
    def _mk_core(self):
        board = HexBoard(5)
        engine = FakeEngine()
        core = GuiCore(board, engine)
        return core, engine

    def _start_candidate_filter(self, core: GuiCore, col: int, row: int):
        core.add_candidate(col, row)
        core.set_analysis_enabled(True)

    def _play_two_moves(self, core: GuiCore) -> None:
        core.try_play_move(1, 1)
        core.try_play_move(2, 1)

    def _new_calls(self, engine: FakeEngine, before: int):
        return engine.calls[before:]

    def _index_of(self, calls, predicate):
        return next(i for i, call in enumerate(calls) if predicate(call))

    def _history_coords(self, core: GuiCore):
        return [core.move_coords(mv) for mv in core.applied_history()]

    def _future_coords(self, core: GuiCore):
        return [core.move_coords(mv) for mv in core.mainline_tail_moves()]

    def _variation_coords(self, core: GuiCore):
        return [core.move_coords(mv) for mv in core.next_variation_moves()]

    def _path_moves(self, core: GuiCore):
        return list(core.current_path_moves())

    def _visible_line(self, core: GuiCore):
        return list(core.visible_line_moves())

    def _movelist_rows(self, core: GuiCore):
        view = core.build_movelist_view()
        rows = [
            (
                row.ply,
                [(cell.column, cell.label, cell.played) for cell in row.cells],
            )
            for row in view.rows
        ]
        return rows, view.focus_row

    def _assert_tree_state(self, core: GuiCore, *, history=None, future=None, variations=None):
        probe = HexBoard(core.board.n)
        for mv in core.current_path_moves():
            self.assertTrue(probe.apply_move(mv))
        self.assertEqual(probe.history, list(core.applied_history()))
        self.assertEqual(probe.occ, core.board.occ)
        self.assertEqual(tuple(core.engine.played), core._engine_position_moves())
        selection = core.session.analysis.candidate_selection
        self.assertEqual(bool(selection.candidates), selection.root_key is not None)
        if selection.candidates:
            self.assertEqual(selection.root_key, core.cache_key())
        if history is not None:
            self.assertEqual(self._history_coords(core), history)
        if future is not None:
            self.assertEqual(self._future_coords(core), future)
        if variations is not None:
            self.assertEqual(self._variation_coords(core), variations)

    def _seed_swap_line(self, core: GuiCore, *, tail=(), rewind: int = 0):
        core.try_play_move(3, 2)
        core.try_swap_move()
        for col, row in tail:
            core.try_play_move(col, row)
        if rewind:
            core.step_back_n(rewind)

    def _assert_failed_load_preserves_state(self, core: GuiCore, engine: FakeEngine, loader, text: str):
        before_history = list(core.board.history)
        before_future = core.mainline_tail_moves()
        before_variations = core.next_variation_moves()
        before_pending = core.session.pending_size
        before_cache = dict(core.session.analysis.cache)
        before_n = core.board.n
        before_rev = core.board.rev
        before_calls = list(engine.calls)

        self.assertIsNotNone(loader(text))
        self.assertEqual(core.board.n, before_n)
        self.assertEqual(core.board.rev, before_rev)
        self.assertEqual(core.board.history, before_history)
        self.assertEqual(core.mainline_tail_moves(), before_future)
        self.assertEqual(core.next_variation_moves(), before_variations)
        self.assertEqual(core.session.pending_size, before_pending)
        self.assertEqual(core.session.analysis.cache, before_cache)
        self.assertEqual(engine.calls, before_calls)

    def _start_slow_batch(self, core: GuiCore, engine: FakeEngine, *, clear_live: bool = False):
        self._play_two_moves(core)
        core.go_first()
        engine.analysis = [
            AnalysisMove("a1", order=1, col=1, row=1, winrate=0.5, visits=1, prior=None, pv=None)
        ]

        if clear_live:
            def clear_analysis_realistic():
                engine.calls.append(("clear_analysis",))
                engine.analysis.clear()

            engine.clear_analysis = clear_analysis_realistic

        core.start_batch_analysis()
        core.tick(0.0)
        run = core.session.analysis.mode
        self.assertIsNotNone(run)
        return run

    def test_current_path_and_mainline_tail_roundtrip(self):
        core, _engine = self._mk_core()
        board = core.board

        core.try_play_move(1, 1)
        core.try_play_move(2, 1)
        core.try_play_move(1, 2)

        self.assertEqual(len(board.history), 3)
        self.assertEqual(core.mainline_tail_moves(), [])

        last = board.history[-1]
        self.assertTrue(core.step_back())
        self.assertEqual(len(board.history), 2)
        self.assertEqual(len(core.mainline_tail_moves()), 1)
        self.assertEqual(core.move_coords(core.mainline_tail_moves()[0]), core.move_coords(last))

        self.assertTrue(core.step_forward())
        self.assertEqual(len(board.history), 3)
        self.assertEqual(core.mainline_tail_moves(), [])
        self.assertEqual(core.move_coords(board.history[-1]), core.move_coords(last))

    def test_step_back_and_forward_n_steps_as_much_as_possible(self):
        core, _engine = self._mk_core()
        self._play_two_moves(core)

        self.assertTrue(core.step_back_n(10))
        self.assertEqual(len(core.board.history), 0)
        self.assertEqual(len(core.mainline_tail_moves()), 2)

        self.assertTrue(core.step_forward_n(10))
        self.assertEqual(len(core.board.history), 2)
        self.assertEqual(core.mainline_tail_moves(), [])

    def test_go_sibling_navigation_cases(self):
        with self.subTest(case="no next frontier node"):
            core, _engine = self._mk_core()
            core.try_play_move(1, 1)
            core.try_play_move(2, 1)
            core.try_play_move(1, 2)
            core.try_play_move(2, 2)
            core.step_back_n(2)
            core.try_play_move(3, 2)
            core.go_first()
            core.step_forward_n(4)

            self._assert_tree_state(core, history=[(1, 1), (2, 1), (1, 2), (2, 2)])
            self.assertFalse(core.go_sibling(1))
            self._assert_tree_state(core, history=[(1, 1), (2, 1), (1, 2), (2, 2)])

        with self.subTest(case="switch sibling without promotion"):
            core, _engine = self._mk_core()
            core.try_play_move(1, 1)
            core.try_play_move(2, 1)
            core.step_back()
            core.try_play_move(3, 1)
            core.go_first()
            core.step_forward_n(2)

            self._assert_tree_state(core, history=[(1, 1), (2, 1)])
            self.assertTrue(core.go_sibling(1))
            self._assert_tree_state(core, history=[(1, 1), (3, 1)])

            self.assertTrue(core.step_back())
            self.assertTrue(core.step_forward())
            self._assert_tree_state(core, history=[(1, 1), (2, 1)])

        with self.subTest(case="left fallback to more-preferred line"):
            core, _engine = self._mk_core()
            core.try_play_move(1, 1)
            core.try_play_move(2, 1)
            core.try_play_move(1, 2)
            core.try_play_move(2, 2)
            core.step_back_n(2)
            core.try_play_move(3, 1)
            core.try_play_move(4, 1)
            core.step_back()
            core.try_play_move(5, 1)
            core.try_play_move(1, 3)

            self._assert_tree_state(core, history=[(1, 1), (2, 1), (3, 1), (5, 1), (1, 3)])
            self.assertTrue(core.go_sibling(-1))
            self._assert_tree_state(core, history=[(1, 1), (2, 1), (3, 1), (4, 1)])

        with self.subTest(case="walk full same-ply frontier left-right"):
            core, _engine = self._mk_core()
            core.try_play_move(1, 1)
            core.try_play_move(2, 1)
            core.step_back()
            core.try_play_move(3, 1)
            core.go_first()
            core.try_play_move(4, 1)
            core.try_play_move(5, 1)
            core.go_first()
            core.step_forward_n(2)

            self._assert_tree_state(core, history=[(1, 1), (2, 1)])
            for direction, expected in (
                (1, [(1, 1), (3, 1)]),
                (1, [(4, 1), (5, 1)]),
                (-1, [(1, 1), (3, 1)]),
                (-1, [(1, 1), (2, 1)]),
            ):
                self.assertTrue(core.go_sibling(direction))
                self._assert_tree_state(core, history=expected)

    def test_build_movelist_view_shows_nested_variations(self):
        core, _engine = self._mk_core()

        core.try_play_move(1, 1)
        core.try_play_move(2, 1)
        core.try_play_move(4, 1)
        core.step_back_n(2)
        core.try_play_move(3, 1)
        core.try_play_move(5, 1)
        core.step_back()
        core.try_play_move(4, 2)
        core.step_back_n(2)
        core.try_play_move(2, 1)
        core.step_forward()

        rows, focus_row = self._movelist_rows(core)
        labels = [[cell[1] for cell in row[1]] for row in rows]
        cols = [[cell[0] for cell in row[1]] for row in rows]
        played = [[cell[2] for cell in row[1]] for row in rows]

        self.assertEqual(labels, [["a1"], ["b1", "c1"], ["d1", "e1", "d2"]])
        self.assertEqual(played, [[True], [True, False], [True, False, False]])
        self.assertEqual(cols[0], [0])
        self.assertEqual(cols[1][0], cols[2][0])
        self.assertEqual(cols[1][1], cols[2][1])
        self.assertGreater(cols[2][1], cols[2][0])
        self.assertGreater(cols[2][2], cols[2][1])
        self.assertEqual(focus_row, 2)

    def test_build_movelist_view_keeps_branch_lane_across_deeper_rows(self):
        core, _engine = self._mk_core()

        core.try_play_move(1, 1)
        core.try_play_move(2, 3)
        core.try_play_move(3, 4)
        core.try_play_move(4, 5)
        core.step_back_n(3)
        core.try_play_move(2, 2)
        core.try_play_move(1, 2)
        core.try_play_move(2, 3)
        core.try_play_move(5, 4)

        rows, focus_row = self._movelist_rows(core)
        labels = [[cell[1] for cell in row[1]] for row in rows]
        cols = [[cell[0] for cell in row[1]] for row in rows]
        played = [[cell[2] for cell in row[1]] for row in rows]

        self.assertEqual(labels, [["a1"], ["b3", "b2"], ["c4", "a2"], ["d5", "b3"], ["e4"]])
        self.assertEqual(played, [[True], [False, True], [False, True], [False, True], [True]])
        branch_col = cols[1][1]
        self.assertEqual(branch_col, cols[2][1])
        self.assertEqual(branch_col, cols[3][1])
        self.assertEqual(branch_col, cols[4][0])
        self.assertLess(cols[1][0], branch_col)
        self.assertEqual(focus_row, 4)

    def test_delete_tail_selected_and_leaf_cases(self):
        with self.subTest(case="clears selected mainline after step back"):
            core, _engine = self._mk_core()
            self._play_two_moves(core)
            core.step_back()
            self._assert_tree_state(core, future=[(2, 1)])
            self.assertTrue(core.delete_tail())
            self._assert_tree_state(core, future=[])

        with self.subTest(case="promotes surviving sibling to mainline"):
            core, _engine = self._mk_core()
            self._play_two_moves(core)
            core.step_back()
            core.try_play_move(3, 1)
            core.step_back()
            self._assert_tree_state(core, future=[(2, 1)])
            self.assertTrue(core.delete_tail())
            self._assert_tree_state(core, future=[(3, 1)])
            self.assertTrue(core.step_forward())
            self._assert_tree_state(core, history=[(1, 1), (3, 1)])

        with self.subTest(case="delete current leaf promotes surviving sibling"):
            core, _engine = self._mk_core()
            self._play_two_moves(core)
            core.step_back()
            core.try_play_move(3, 1)
            self.assertTrue(core.delete_tail())
            self._assert_tree_state(core, history=[(1, 1)], future=[(2, 1)])

    def test_rebuild_engine_from_applied_history(self):
        core, engine = self._mk_core()
        board = core.board

        board.place(Side.RED, 1, 1)
        board.place(Side.BLUE, 2, 1)
        board.pass_move(Side.RED)

        core.rebuild_engine_from_applied_history()

        expected = [
            ("clear_board",),
            ("play", Side.RED, 1, 1),
            ("play", Side.BLUE, 2, 1),
            ("play", Side.RED, None, None),
        ]
        self.assertEqual(engine.calls, expected)

    def test_replace_engine_reconciles_position_and_analysis(self):
        core, old_engine = self._mk_core()
        core.try_play_move(1, 1)
        core.add_candidate(2, 2)
        core.set_analysis_enabled(True)
        core.session.analysis.cache[core.cache_key()] = ["cached"]
        new_engine = FakeEngine()

        self.assertTrue(core.replace_engine(new_engine))

        self.assertIs(core.engine, new_engine)
        self.assertIn(("close",), old_engine.calls)
        self.assertEqual(new_engine.played, [(Side.RED, 1, 1)])
        self.assertEqual(core.session.analysis.candidate_selection.candidates, {(2, 2)})
        self.assertFalse(core.session.analysis.cache)
        self.assertTrue(any(call[0] == "start_analysis" and len(call) == 4 for call in new_engine.calls))

    def test_replace_engine_failure_restores_old_engine(self):
        core, old_engine = self._mk_core()
        new_engine = FakeEngine()
        new_engine.clear_board = mock.Mock(side_effect=RuntimeError("failed"))
        new_engine.close = mock.Mock()

        self.assertFalse(core.replace_engine(new_engine))

        self.assertIs(core.engine, old_engine)
        new_engine.close.assert_called_once_with()

    def test_replace_engine_same_instance_is_noop(self):
        core, engine = self._mk_core()
        before_calls = list(engine.calls)

        self.assertFalse(core.replace_engine(engine))

        self.assertIs(core.engine, engine)
        self.assertEqual(engine.calls, before_calls)

    def test_candidate_root_key_invalidation(self):
        core, _engine = self._mk_core()
        board = core.board

        core.add_candidate(1, 1)
        self.assertTrue(core.session.analysis.candidate_selection.candidates)
        root_key = core.session.analysis.candidate_selection.root_key
        self.assertIsNotNone(root_key)

        board.place(Side.RED, 2, 2)
        self.assertNotEqual(core.cache_key(), root_key)

        cleared = core.check_candidate_root()
        self.assertTrue(cleared)
        self.assertEqual(core.session.analysis.candidate_selection.candidates, set())
        self.assertIsNone(core.session.analysis.candidate_selection.root_key)

    def test_play_move_with_analysis_off_clears_invalid_candidates_immediately(self):
        core, _engine = self._mk_core()

        core.add_candidate(1, 1)
        self.assertEqual(core.session.analysis.candidate_selection.candidates, {(1, 1)})
        self.assertFalse(core.session.analysis.enabled)

        self.assertTrue(core.try_play_move(1, 1))
        self.assertEqual(core.session.analysis.candidate_selection.candidates, set())
        self.assertIsNone(core.session.analysis.candidate_selection.root_key)

    def test_merge_analysis_lists_keeps_order_and_uses_deeper_eval_metadata(self):
        core, _engine = self._mk_core()
        old_pv = ((1, 1), (2, 1))
        new_pv = ((1, 1), (3, 1))

        def row(order, winrate, visits, prior, pv):
            return AnalysisMove("a1", order, 1, 1, winrate, visits, prior, pv)

        primary = [row(1, 0.4, 10, 0.2, old_pv)]
        secondary = [row(9, 0.7, 50, 0.9, new_pv)]

        merged = core._merge_analysis_lists(primary, secondary)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].order, 1)
        self.assertEqual(merged[0].prior, 0.9)
        self.assertEqual(merged[0].pv, new_pv)
        self.assertEqual(merged[0].winrate, 0.7)
        self.assertEqual(merged[0].visits, 50)

    def test_step_back_clears_buffered_engine_analysis(self):
        core, engine = self._mk_core()
        core.try_play_move(1, 1)
        core.try_play_move(2, 1)

        # FakeEngine.clear_analysis only logs by default; make this instance mimic
        # the real engine behavior of dropping buffered live analysis.
        def clear_analysis_realistic():
            engine.calls.append(("clear_analysis",))
            engine.analysis.clear()

        engine.clear_analysis = clear_analysis_realistic
        engine.analysis = [
            AnalysisMove("c1", order=0, col=3, row=1, winrate=0.6, visits=10, prior=0.4, pv=None)
        ]

        self.assertTrue(core.step_back())
        self.assertEqual(engine.analysis, [])

    def test_live_position_change_preserves_engine_command_contract(self):
        core, engine = self._mk_core()
        core.set_analysis_enabled(True)
        engine.calls.clear()

        self.assertTrue(core.try_play_move(1, 1))

        self.assertEqual(
            engine.calls,
            [
                ("stop_analysis",),
                ("clear_analysis",),
                ("play", Side.RED, 1, 1),
                ("clear_analysis",),
                ("clear_analysis",),
                ("kata_set_param", "analysisWideRootNoise", 0.04),
                ("start_analysis", Side.BLUE, core.analyze_interval_cs),
            ],
        )

    def test_delete_tail_keeps_existing_cache_entries(self):
        core, _engine = self._mk_core()

        self._play_two_moves(core)

        key0 = core.cache_key_for_moves([])
        key1 = core.cache_key_for_moves(self._path_moves(core)[:1])
        key2 = core.cache_key_for_moves(self._path_moves(core)[:2])
        core.session.analysis.cache[key0] = ["a"]
        core.session.analysis.cache[key1] = ["b"]
        core.session.analysis.cache[key2] = ["c"]

        core.delete_tail()  # removes last move; history length back to 1

        self.assertIn(key0, core.session.analysis.cache)
        self.assertIn(key1, core.session.analysis.cache)
        self.assertIn(key2, core.session.analysis.cache)

    def test_branching_clears_future_entries_but_keeps_cache(self):
        core, _engine = self._mk_core()

        self._play_two_moves(core)
        core.step_back()  # selected mainline tail has one move

        key0 = core.cache_key_for_moves([])
        key1 = core.cache_key_for_moves(self._path_moves(core)[:1])
        key2 = core.cache_key_for_moves(self._visible_line(core)[:2])
        core.session.analysis.cache[key0] = ["a"]
        core.session.analysis.cache[key1] = ["b"]
        core.session.analysis.cache[key2] = ["c"]

        core.try_play_move(3, 1)  # diverge, should clear future

        self.assertEqual(core.mainline_tail_moves(), [])
        self.assertIn(key0, core.session.analysis.cache)
        self.assertIn(key1, core.session.analysis.cache)
        self.assertIn(key2, core.session.analysis.cache)

    def test_try_play_or_pass_creates_variation_without_promoting_mainline(self):
        cases = (
            ("play creates variation", lambda core: core.try_play_move(3, 1), [(1, 1), (3, 1)], [(3, 1)]),
            ("pass creates variation", lambda core: core.try_pass_move(), [(1, 1), None], [None]),
        )
        for label, action, history, variations in cases:
            with self.subTest(case=label):
                core, _engine = self._mk_core()
                self._play_two_moves(core)
                core.step_back()

                self._assert_tree_state(core, future=[(2, 1)], variations=[])
                self.assertTrue(action(core))
                self._assert_tree_state(core, history=history)

                self.assertTrue(core.step_back())
                self._assert_tree_state(core, future=[(2, 1)], variations=variations)
                self.assertTrue(core.go_last())
                self._assert_tree_state(core, history=[(1, 1), (2, 1)])

    def test_try_play_moves_replays_redo_prefix_then_branches_and_keeps_cache(self):
        core, _engine = self._mk_core()

        core.try_play_move(1, 1)
        core.try_play_move(2, 1)
        core.try_play_move(1, 2)
        core.step_back_n(2)

        key0 = core.cache_key_for_moves([])
        key1 = core.cache_key_for_moves(self._path_moves(core)[:1])
        key2 = core.cache_key_for_moves(self._visible_line(core)[:2])
        key3 = core.cache_key_for_moves(self._visible_line(core)[:3])
        core.session.analysis.cache[key0] = ["a"]
        core.session.analysis.cache[key1] = ["b"]
        core.session.analysis.cache[key2] = ["c"]
        core.session.analysis.cache[key3] = ["d"]

        did = core.try_play_moves([(2, 1), (3, 1)])

        self.assertTrue(did)
        self.assertEqual(self._history_coords(core), [(1, 1), (2, 1), (3, 1)])
        self.assertEqual(core.mainline_tail_moves(), [])
        self.assertIn(key0, core.session.analysis.cache)
        self.assertIn(key1, core.session.analysis.cache)
        self.assertIn(key2, core.session.analysis.cache)
        self.assertIn(key3, core.session.analysis.cache)

    def test_try_play_or_pass_follows_existing_variation_without_promoting_mainline(self):
        cases = (
            (
                "play follows existing variation",
                lambda core: (
                    core.try_play_move(1, 1),
                    core.try_play_move(3, 1),
                    core.step_back(),
                    core.try_play_move(2, 1),
                    core.step_back(),
                ),
                lambda core: core.try_play_move(2, 1),
                [(3, 1)],
                [(2, 1)],
                [(1, 1), (2, 1)],
            ),
            (
                "pass follows existing variation",
                lambda core: (
                    core.try_play_move(1, 1),
                    core.try_play_move(2, 1),
                    core.step_back(),
                    core.try_pass_move(),
                    core.step_back(),
                ),
                lambda core: core.try_pass_move(),
                [(2, 1)],
                [None],
                [(1, 1), None],
            ),
        )
        for label, setup, action, future, variations, history in cases:
            with self.subTest(case=label):
                core, _engine = self._mk_core()
                setup(core)

                self._assert_tree_state(core, future=future, variations=variations)
                self.assertTrue(action(core))
                self._assert_tree_state(core, history=history)

                self.assertTrue(core.step_back())
                self._assert_tree_state(core, future=future, variations=variations)
                self.assertTrue(core.go_last())
                self._assert_tree_state(core, history=[(1, 1)] + future)

    def test_try_pass_move_prefers_redo_pass_over_new_pass(self):
        core, _engine = self._mk_core()

        core.try_play_move(1, 1)
        core.try_play_move(2, 1)
        core.try_pass_move()
        core.try_play_move(3, 1)

        core.step_back_n(2)
        self.assertEqual(self._future_coords(core), [None, (3, 1)])

        did = core.try_pass_move()

        self.assertTrue(did)
        self.assertEqual(self._history_coords(core), [(1, 1), (2, 1), None])
        self.assertEqual(self._future_coords(core), [(3, 1)])

    def test_try_swap_move_sequence_updates_board_and_engine_mapping(self):
        core, engine = self._mk_core()
        self._seed_swap_line(core, tail=[(5, 4)])

        self.assertEqual([mv.kind for mv in core.board.history], [MoveKind.PLACE, MoveKind.SWAP, MoveKind.PLACE])
        self.assertEqual(self._history_coords(core), [(2, 3), None, (5, 4)])  # b3, swap, e4
        self.assertEqual(core.board.history[0].side, Side.BLUE)
        self.assertEqual(core.board.history[2].side, Side.RED)
        self.assertEqual(core.board.get(2, 3), int(Side.BLUE))
        self.assertEqual(core.board.get(5, 4), int(Side.RED))
        self.assertEqual(core.board.get(3, 2), -1)
        self.assertEqual(engine.played, [(Side.RED, 3, 2), (Side.BLUE, 4, 5)])  # c2, d5

    def test_movelist_label_uses_original_opening_after_swap(self):
        core, _engine = self._mk_core()
        self._seed_swap_line(core, tail=[(5, 4)])

        moves = core.visible_line_moves()
        labels = [core.move_to_label(moves[i]) for i in range(len(moves))]
        self.assertEqual(labels, ["c2", "swap", "e4"])

    def test_try_swap_move_prefers_redo_swap_over_new_swap(self):
        core, _engine = self._mk_core()
        self._seed_swap_line(core, tail=[(5, 4)], rewind=2)

        did = core.try_swap_move()

        self.assertTrue(did)
        self.assertEqual([mv.kind for mv in core.board.history], [MoveKind.PLACE, MoveKind.SWAP])
        self.assertEqual(self._history_coords(core), [(2, 3), None])
        self.assertEqual(self._future_coords(core), [(5, 4)])

    def test_undo_redo_restore_handles_swap_history(self):
        core, _engine = self._mk_core()
        self._seed_swap_line(core, tail=[(5, 4)])
        core.delete_tail()  # remove e4; undo target includes swap history

        self.assertTrue(core.undo_edit())
        self.assertEqual([mv.kind for mv in core.board.history], [MoveKind.PLACE, MoveKind.SWAP, MoveKind.PLACE])
        self.assertEqual(self._history_coords(core), [(2, 3), None, (5, 4)])

        self.assertTrue(core.redo_edit())
        self.assertEqual([mv.kind for mv in core.board.history], [MoveKind.PLACE, MoveKind.SWAP])
        self.assertEqual(self._history_coords(core), [(2, 3), None])

    def test_invalid_edit_noop_does_not_restart_analysis(self):
        cases = (
            ("invalid swap", lambda core: core.try_swap_move()),
            ("invalid drag", lambda core: core.try_drag_move(0, (1, 1), 2, 1)),
        )
        for label, action in cases:
            with self.subTest(case=label):
                core, engine = self._mk_core()
                if label == "invalid drag":
                    self._play_two_moves(core)

                core.toggle_analysis()
                before = len(engine.calls)

                self.assertFalse(action(core))
                new_calls = self._new_calls(engine, before)
                self.assertEqual(sum(1 for call in new_calls if call[0] == "start_analysis"), 0)

    def test_drag_first_move_with_future_swap_updates_swap_and_truncates_conflict(self):
        core, _engine = self._mk_core()
        self._seed_swap_line(core, tail=[(2, 4)], rewind=2)

        did = core.try_drag_move(0, (3, 2), 4, 2)  # d2 -> swap target b4

        self.assertTrue(did)
        self.assertEqual(self._history_coords(core), [(4, 2)])
        self.assertEqual([mv.kind for mv in core.mainline_tail_moves()], [MoveKind.SWAP])
        swap_mv = core.mainline_tail_moves()[0]
        self.assertEqual((swap_mv.col, swap_mv.row), (4, 2))
        seq = core.visible_line_moves()
        self.assertEqual([core.move_to_label(seq[i]) for i in range(len(seq))], ["d2", "swap"])

    def test_drag_first_move_prunes_only_illegal_swap_descendant(self):
        core, _engine = self._mk_core()

        core.try_play_move(3, 2)  # c2
        core.try_play_move(5, 4)  # e4 mainline
        core.step_back()
        core.try_swap_move()  # swap variation
        core.try_play_move(2, 4)  # b4 under swap variation
        core.step_back_n(2)  # current path: c2 ; selected tail: e4

        did = core.try_drag_move(0, (3, 2), 4, 2)  # c2 -> d2; swap branch b4 now conflicts

        self.assertTrue(did)
        self.assertEqual(self._history_coords(core), [(4, 2)])
        self.assertEqual(self._future_coords(core), [(5, 4)])

        self.assertTrue(core.step_forward())
        self.assertEqual(self._history_coords(core), [(4, 2), (5, 4)])
        self.assertTrue(core.step_back())
        self.assertTrue(core.try_swap_move())
        self.assertEqual(core.current_path_moves()[1], Move.swap(side=Side.BLUE, col=4, row=2))
        self.assertEqual(core.mainline_tail_moves(), [])

    def test_drag_swapped_stone_truncates_future_on_stone_conflict(self):
        core, _engine = self._mk_core()
        self._seed_swap_line(core, tail=[(5, 4)], rewind=1)

        did = core.try_drag_move(0, (2, 3), 5, 4)  # drag swapped stone onto future e4

        self.assertTrue(did)
        self.assertEqual(core.mainline_tail_moves(), [])
        self.assertEqual([mv.kind for mv in core.board.history], [MoveKind.PLACE, MoveKind.SWAP])
        self.assertEqual((core.board.history[1].col, core.board.history[1].row), (4, 5))

    def test_drag_in_swapped_position_keeps_legal_future_moves(self):
        core, _engine = self._mk_core()
        self._seed_swap_line(core, tail=[(5, 4), (1, 5)], rewind=1)

        did = core.try_drag_move(2, (5, 4), 5, 5)  # e4 -> e5

        self.assertTrue(did)
        self.assertEqual(self._history_coords(core), [(2, 3), None, (5, 5)])
        self.assertEqual(self._future_coords(core), [(1, 5)])

    def test_drag_merges_mainline_branch_onto_existing_variation(self):
        core, _engine = self._mk_core()

        core.try_play_move(1, 1)
        core.try_play_move(2, 1)
        core.try_play_move(1, 2)
        core.try_play_move(2, 2)
        core.step_back_n(3)
        core.try_play_move(3, 1)
        core.try_play_move(1, 2)
        core.try_play_move(3, 2)
        core.step_back_n(3)
        core.step_forward()

        did = core.try_drag_move(1, (2, 1), 3, 1)

        self.assertTrue(did)
        self.assertEqual(self._history_coords(core), [(1, 1), (3, 1)])
        self.assertEqual(self._future_coords(core), [(1, 2), (2, 2)])

        self.assertTrue(core.step_forward_n(2))
        self.assertEqual(self._history_coords(core), [(1, 1), (3, 1), (1, 2), (2, 2)])

        self.assertTrue(core.step_back())
        self.assertEqual(self._history_coords(core), [(1, 1), (3, 1), (1, 2)])
        self.assertEqual(self._future_coords(core), [(2, 2)])
        self.assertEqual(self._variation_coords(core), [(3, 2)])

    def test_drag_merges_root_branch_without_changing_preferred_opening(self):
        core, _engine = self._mk_core()

        core.try_play_move(1, 1)
        core.try_play_move(2, 1)
        core.go_first()
        core.try_play_move(2, 1)
        core.try_play_move(1, 2)

        did = core.try_drag_move(0, (2, 1), 1, 1)

        self.assertTrue(did)
        self.assertEqual(self._history_coords(core), [(1, 1), (1, 2)])

        self.assertTrue(core.step_back())
        self.assertEqual(self._future_coords(core), [(2, 1)])
        self.assertEqual(self._variation_coords(core), [(1, 2)])

    def test_drag_merges_side_variations_without_promoting_branch(self):
        core, _engine = self._mk_core()

        core.try_play_move(1, 1)
        core.try_play_move(2, 1)
        core.try_play_move(1, 2)
        core.try_play_move(2, 2)
        core.step_back_n(3)
        core.try_play_move(3, 1)
        core.try_play_move(1, 2)
        core.try_play_move(3, 2)
        core.step_back_n(3)
        core.try_play_move(4, 1)
        core.try_play_move(1, 2)
        core.try_play_move(4, 2)

        did = core.try_drag_move(1, (4, 1), 3, 1)

        self.assertTrue(did)
        self.assertEqual(self._history_coords(core), [(1, 1), (3, 1), (1, 2), (4, 2)])

        self.assertTrue(core.step_back_n(3))
        self.assertEqual(self._history_coords(core), [(1, 1)])
        self.assertEqual(self._future_coords(core), [(2, 1), (1, 2), (2, 2)])
        self.assertEqual(self._variation_coords(core), [(3, 1)])

        self.assertTrue(core.try_play_move(3, 1))
        self.assertTrue(core.step_forward_n(2))
        self.assertEqual(self._history_coords(core), [(1, 1), (3, 1), (1, 2), (3, 2)])

    def test_drag_first_move_with_future_swap_keeps_legal_reuse(self):
        core, _engine = self._mk_core()
        self._seed_swap_line(core, tail=[(4, 2)], rewind=2)

        did = core.try_drag_move(0, (3, 2), 4, 2)  # c2 -> d2

        self.assertTrue(did)
        self.assertEqual(self._history_coords(core), [(4, 2)])
        self.assertEqual([mv.kind for mv in core.mainline_tail_moves()], [MoveKind.SWAP, MoveKind.PLACE])
        seq = core.visible_line_moves()
        self.assertEqual([core.move_to_label(seq[i]) for i in range(len(seq))], ["d2", "swap", "d2"])

        core.step_forward_n(2)
        self.assertEqual(self._history_coords(core), [(2, 4), None, (4, 2)])
        self.assertEqual(core.board.get(2, 4), int(Side.BLUE))
        self.assertEqual(core.board.get(4, 2), int(Side.RED))

    def test_build_hexworld_url_emits_swap_token(self):
        core, _engine = self._mk_core()
        self._seed_swap_line(core, tail=[(5, 4)])
        self.assertEqual(core.build_hexworld_url(), "https://hexworld.org/board/#5c1,c2:se4")

    def test_hexata_format_roundtrip_with_variations_pass_and_swap(self):
        core, _engine = self._mk_core()

        core.try_play_move(3, 2)  # c2
        core.try_pass_move()
        core.step_back()
        core.try_swap_move()
        core.try_play_move(5, 4)  # e4
        core.go_first()
        core.try_play_move(2, 1)  # b1 root variation

        text = core.build_hexata_format()
        self.assertEqual(text, "5,c2(b1,):p(:se4)")

        loaded = GuiCore(HexBoard(6), FakeEngine())
        error = loaded.load_hexata_format(text)

        self.assertIsNone(error)
        self.assertEqual(loaded.board.n, 5)
        self.assertEqual(loaded.build_hexata_format(), text)
        self.assertEqual(self._history_coords(loaded), [(2, 1)])

        self.assertTrue(loaded.go_first())
        self.assertTrue(loaded.step_forward_n(2))
        self.assertTrue(loaded.go_sibling(1))
        self.assertEqual([mv.kind for mv in loaded.current_path_moves()], [MoveKind.PLACE, MoveKind.SWAP])
        self.assertTrue(loaded.step_forward())
        self.assertEqual(self._history_coords(loaded), [(2, 3), None, (5, 4)])

    def test_hexata_format_canonicalizes_mainline_end_cursor_marker(self):
        core, _engine = self._mk_core()

        core.try_play_move(1, 1)
        core.try_play_move(2, 1)
        core.step_back()
        core.try_play_move(3, 1)
        core.go_first()
        core.step_forward_n(2)

        self.assertEqual(core.build_hexata_format(), "5,a1b1(c1)")
        error = core.load_hexata_format("5,a1b1,(c1)")
        self.assertIsNone(error)
        self.assertEqual(core.build_hexata_format(), "5,a1b1(c1)")
        self.assertEqual(self._history_coords(core), [(1, 1), (2, 1)])

    def test_hexata_format_roundtrips_cursor_inside_variation(self):
        core, _engine = self._mk_core()

        error = core.load_hexata_format("5,a1b1(c1,d1)")

        self.assertIsNone(error)
        self.assertEqual(core.build_hexata_format(), "5,a1b1(c1,d1)")
        self.assertEqual(self._history_coords(core), [(1, 1), (3, 1)])

    def test_load_hexata_format_rejects_spaces_uppercase_and_bad_cursor_markers(self):
        core, engine = self._mk_core()
        before_history = list(core.board.history)
        before_calls = list(engine.calls)

        for text in ("5,a1 a2", "5,A1", "5,a1(,b1)", "5,a1(b1),", "5,a1,b1,"):
            with self.subTest(text=text):
                self.assertIsNotNone(core.load_hexata_format(text))
                self.assertEqual(core.board.history, before_history)
                self.assertEqual(engine.calls, before_calls)

    def test_parse_hexata_format_maps_recursion_error_to_value_error(self):
        with mock.patch("formats.hexata._Parser.parse", side_effect=RecursionError("boom")):
            with self.assertRaises(ValueError):
                hexata.parse_hexata_format("5,a1")

    def test_load_hexata_format_out_of_range_rejected_before_full_parse(self):
        core, _engine = self._mk_core()
        with mock.patch(
            "gui.core.hexata.parse_hexata_format",
            side_effect=AssertionError("parse should not run"),
        ):
            self.assertIsNotNone(core.load_hexata_format("2000,a1"))

    def test_deep_line_operations_avoid_recursion(self):
        old_limit = sys.getrecursionlimit()
        depth = 50
        recursion_limit = depth - 5
        sys.setrecursionlimit(recursion_limit)
        try:
            board_size = 8
            core = GuiCore(HexBoard(board_size), FakeEngine())
            total = 0
            for row in range(1, board_size + 1):
                for col in range(1, board_size + 1):
                    if total >= depth:
                        break
                    self.assertTrue(core.try_play_move(col, row))
                    total += 1
                if total >= depth:
                    break

            self.assertEqual(core.current_ply(), depth)
            self.assertTrue(core.build_hexata_format().startswith(f"{board_size},"))
            self.assertEqual(len(core.build_movelist_view().rows), depth)

            self.assertTrue(core.step_back_n(8))
            src = core.move_coords(core.applied_history()[0])
            self.assertIsNotNone(src)
            target = None
            for row in range(1, board_size + 1):
                for col in range(1, board_size + 1):
                    if core.board.is_empty(col, row):
                        target = (col, row)
                        break
                if target is not None:
                    break
            self.assertIsNotNone(target)
            self.assertTrue(core.try_drag_move(0, src, target[0], target[1]))
        finally:
            sys.setrecursionlimit(old_limit)

    def test_eval_graph_prefix_keys_keep_applied_swap_prefix_after_swap(self):
        core, _engine = self._mk_core()
        self._seed_swap_line(core, tail=[(5, 4)])

        graph_data = core.build_eval_graph_data()
        keys = graph_data.prefix_keys

        self.assertEqual(len(keys), 3)
        self.assertEqual(
            list(graph_data.moves),
            list(core.applied_history()),
        )
        self.assertEqual(keys[0], core.cache_key_for_applied_moves(core.applied_history()[:1]))
        self.assertNotEqual(keys[0], core.cache_key_for_moves(core.visible_line_moves()[:1]))

    def test_eval_graph_prefix_keys_replay_future_swap_from_pre_swap_cursor(self):
        core, _engine = self._mk_core()
        self._seed_swap_line(core, tail=[(5, 4)], rewind=2)

        graph_data = core.build_eval_graph_data()
        keys = graph_data.prefix_keys

        self.assertEqual(len(keys), 3)
        self.assertEqual(
            list(graph_data.moves),
            list(core.applied_history()) + core.mainline_tail_moves(),
        )
        self.assertEqual(keys[0], core.cache_key_for_moves(core.current_path_moves()))
        self.assertEqual(
            keys[1],
            core.cache_key_for_moves(core.current_path_moves() + core.mainline_tail_moves()[:1]),
        )
        self.assertEqual(keys[2], core.cache_key_for_moves(core.visible_line_moves()))

    def test_load_hexworld_text_with_swap(self):
        core, engine = self._mk_core()

        error = core.load_hexworld_text("https://hexworld.org/board/#5c1,c2:se4")

        self.assertIsNone(error)
        self.assertEqual([mv.kind for mv in core.board.history], [MoveKind.PLACE, MoveKind.SWAP, MoveKind.PLACE])
        self.assertEqual(self._history_coords(core), [(2, 3), None, (5, 4)])
        self.assertEqual([mv.side for mv in core.board.history], [Side.BLUE, Side.BLUE, Side.RED])
        self.assertEqual(core.current_side(), Side.BLUE)
        self.assertEqual(core.board.get(2, 3), int(Side.BLUE))
        self.assertEqual(core.board.get(5, 4), int(Side.RED))
        self.assertEqual(core.board.get(3, 2), -1)
        self.assertEqual(engine.played, [(Side.RED, 3, 2), (Side.BLUE, 4, 5)])

    def test_load_hexworld_text_allows_post_swap_reuse_of_opening_coord(self):
        core, _engine = self._mk_core()

        error = core.load_hexworld_text("https://hexworld.org/board/#5c1,c2:sc2")

        self.assertIsNone(error)
        self.assertEqual([mv.kind for mv in core.board.history], [MoveKind.PLACE, MoveKind.SWAP, MoveKind.PLACE])
        self.assertEqual(core.board.get(2, 3), int(Side.BLUE))  # swapped opening stone
        self.assertEqual(core.board.get(3, 2), int(Side.RED))  # legal reuse of c2 after swap

    def test_load_flexible_move_format_accepts_coord_lists_only_at_current_size(self):
        for text in ("1. c2 d5\n3. e1", "1 c2 2 D5 3 e1", "c2d5e1"):
            with self.subTest(text=text):
                core, _engine = self._mk_core()

                error = core.load_flexible_move_format(text)

                self.assertIsNone(error)
                self.assertEqual(self._history_coords(core), [(3, 2), (4, 5), (5, 1)])
                self.assertEqual([mv.side for mv in core.board.history], [Side.RED, Side.BLUE, Side.RED])

        core, _engine = self._mk_core()
        error = core.load_flexible_move_format("d5SWAPd5PASS")
        self.assertIsNone(error)
        self.assertEqual([mv.kind for mv in core.board.history], [MoveKind.PLACE, MoveKind.SWAP, MoveKind.PLACE, MoveKind.PASS])
        self.assertEqual(self._history_coords(core), [(5, 4), None, (4, 5), None])

        core, _engine = self._mk_core()
        error = core.load_flexible_move_format("1.c2 2.resign")
        self.assertIsNone(error)
        self.assertEqual(self._history_coords(core), [(3, 2)])
        self.assertEqual([mv.side for mv in core.board.history], [Side.RED])

        core, engine = self._mk_core()
        self._assert_failed_load_preserves_state(core, engine, core.load_flexible_move_format, "c22.d5")
        self._assert_failed_load_preserves_state(core, engine, core.load_flexible_move_format, "1 c2 2")
        self._assert_failed_load_preserves_state(core, engine, core.load_flexible_move_format, "1 2 c2")
        self._assert_failed_load_preserves_state(core, engine, core.load_flexible_move_format, "1 c2 2d5")

    def test_swap_branch_uses_distinct_cache_key(self):
        core, _engine = self._mk_core()

        core.try_play_move(3, 2)  # c2
        core.try_play_move(4, 2)  # d2
        stale_key = core.cache_key()
        core.session.analysis.cache[stale_key] = [
            AnalysisMove("a1", order=1, col=1, row=1, winrate=0.5, visits=100, prior=0.1, pv=None)
        ]

        core.step_back()  # history: c2 ; future: d2
        did = core.try_swap_move()  # branch to swap at ply 2

        self.assertTrue(did)
        self.assertNotEqual(core.cache_key(), stale_key)
        self.assertIn(stale_key, core.session.analysis.cache)
        self.assertEqual(core.get_active_analysis(), [])

    def test_undo_redo_does_not_clear_analysis_cache(self):
        core, _engine = self._mk_core()

        core.try_play_move(1, 1)
        key_before = core.cache_key()
        core.session.analysis.cache[key_before] = ["a"]

        core.try_play_move(2, 1)
        key_after = core.cache_key()
        core.session.analysis.cache[key_after] = ["b"]

        self.assertTrue(core.undo_edit())
        self.assertEqual(core.cache_key(), key_before)
        self.assertEqual(core.session.analysis.cache[key_before], ["a"])
        self.assertEqual(core.session.analysis.cache[key_after], ["b"])

        self.assertTrue(core.redo_edit())
        self.assertEqual(core.cache_key(), key_after)
        self.assertEqual(core.session.analysis.cache[key_before], ["a"])
        self.assertEqual(core.session.analysis.cache[key_after], ["b"])

    def test_undo_restores_candidates_and_candidate_analysis(self):
        core, engine = self._mk_core()

        def clear_analysis_realistic():
            engine.calls.append(("clear_analysis",))
            engine.analysis.clear()

        engine.clear_analysis = clear_analysis_realistic

        core.toggle_analysis()
        core.add_candidate(1, 1)
        engine.analysis = [
            AnalysisMove("a1", order=0, col=1, row=1, winrate=0.4, visits=5, prior=0.2, pv=None)
        ]
        core.tick(0.0)

        core.try_play_move(2, 1)
        self.assertEqual(core.session.analysis.candidate_selection.candidates, set())
        self.assertEqual(core.session.analysis.mode, AnalysisModeTag.LIVE)

        self.assertTrue(core.undo_edit())
        self.assertEqual(core.session.analysis.candidate_selection.candidates, {(1, 1)})
        self.assertEqual(core.candidate_result((1, 1)), (0.4, 5))
        self.assertEqual(core.session.analysis.candidate_selection.root_key, core.cache_key())
        self.assertEqual(core.session.analysis.mode, AnalysisModeTag.LIVE)

    def test_undo_after_cache_clear_restores_candidates_without_stale_analysis(self):
        core, _engine = self._mk_core()

        core.toggle_analysis()
        core.add_candidate(1, 1)
        core.session.analysis.cache[core.cache_key()] = [
            AnalysisMove("a1", order=0, col=1, row=1, winrate=0.4, visits=5, prior=0.2, pv=None)
        ]

        self.assertTrue(core.try_play_move(2, 1))
        core.clear_analysis_caches()

        self.assertTrue(core.undo_edit())
        self.assertEqual(core.session.analysis.candidate_selection.candidates, {(1, 1)})
        self.assertEqual(core.candidate_result((1, 1)), (None, None))
        self.assertEqual(core.session.analysis.candidate_selection.root_key, core.cache_key())

    def test_undo_during_batch_exits_batch_mode(self):
        core, _engine = self._mk_core()

        core.try_play_move(1, 1)
        core.try_play_move(2, 1)
        core.start_batch_analysis(fast=True)

        self.assertTrue(core.is_batch_analysis_active())
        self.assertTrue(core.undo_edit())
        self.assertFalse(core.is_batch_analysis_active())
        self.assertEqual(core.session.analysis.mode, AnalysisModeTag.LIVE)

    def test_clear_analysis_caches_while_candidate_filter_keeps_candidates(self):
        core, _engine = self._mk_core()

        core.add_candidate(1, 1)
        core.add_candidate(2, 2)
        core.set_analysis_enabled(True)

        core.session.analysis.cache[core.cache_key_for_moves([])] = ["x"]

        core.clear_analysis_caches()

        self.assertTrue(core.session.analysis.enabled)
        self.assertEqual(core.session.analysis.candidate_selection.candidates, {(1, 1), (2, 2)})
        self.assertEqual(core.session.analysis.cache, {})

    def test_clear_analysis_caches_during_fast_batch_restarts_raw_nn_immediately(self):
        board = HexBoard(5)
        engine = RawCaptureBlockingEngine()
        core = GuiCore(board, engine)

        core.try_play_move(1, 1)
        core.start_batch_analysis(fast=True)
        core.tick(0.0)
        run = core.session.analysis.mode

        self.assertTrue(core.is_batch_analysis_active())
        self.assertTrue(engine.raw_capture_pending)
        self.assertTrue(run.raw_pending)

        core.clear_analysis_caches()
        self.assertTrue(core.is_batch_analysis_active())
        self.assertFalse(engine.raw_capture_pending)
        self.assertFalse(run.raw_pending)

        engine.calls.clear()
        core.tick(1.0)

        self.assertIn(("start_kata_raw_nn", 0), engine.calls)
        self.assertTrue(run.raw_pending)

    def test_live_cache_updates_when_only_non_top_rows_change(self):
        core, engine = self._mk_core()

        core.toggle_analysis()
        engine.analysis = [
            AnalysisMove("a1", order=1, col=1, row=1, winrate=0.6, visits=10, prior=0.3, pv=None),
            AnalysisMove("b1", order=2, col=2, row=1, winrate=0.4, visits=5, prior=0.2, pv=None),
        ]
        core.tick(0.0)

        key = core.cache_key()
        cached = core.session.analysis.cache[key]
        self.assertEqual((cached[1].winrate, cached[1].visits), (0.4, 5))

        engine.analysis = [
            AnalysisMove("a1", order=1, col=1, row=1, winrate=0.6, visits=10, prior=0.3, pv=None),
            AnalysisMove("b1", order=2, col=2, row=1, winrate=0.45, visits=50, prior=0.2, pv=None),
        ]
        core.tick(0.1)

        cached = core.session.analysis.cache[key]
        self.assertEqual((cached[1].winrate, cached[1].visits), (0.45, 50))

    def test_fast_batch_actions_do_not_start_live_analysis(self):
        for label, action in (
            ("clear_analysis_caches", lambda core: core.clear_analysis_caches()),
            ("set_awrn", lambda core: core.set_analysis_wide_root_noise(0.10)),
        ):
            with self.subTest(case=label):
                core, engine = self._mk_core()
                core.start_batch_analysis(fast=True)
                before = len(engine.calls)

                action(core)
                new_calls = self._new_calls(engine, before)

                self.assertTrue(core.is_batch_analysis_active())
                self.assertEqual(sum(1 for call in new_calls if call[0] == "start_analysis"), 0)

    def test_slow_batch_actions_reset_position_timer(self):
        for label, action in (
            ("set_awrn", lambda core: core.set_analysis_wide_root_noise(0.10)),
            ("clear_analysis_caches", lambda core: core.clear_analysis_caches()),
        ):
            with self.subTest(case=label):
                core, engine = self._mk_core()
                run = self._start_slow_batch(core, engine)
                self.assertEqual(run.first_update_at, 0.0)

                action(core)
                self.assertIsNone(run.first_update_at)

    def test_failed_loads_preserve_state(self):
        cases = (
            ("hexata duplicate", "load_hexata_format", "5,a1(b1)(b1)"),
            ("hexworld duplicate", "load_hexworld_text", "https://hexworld.org/board/#5c1,a1a1"),
        )
        for label, method, text in cases:
            with self.subTest(case=label):
                core, engine = self._mk_core()
                self._play_two_moves(core)
                core.step_back()
                core.try_play_move(3, 1)
                core.step_back()
                core.session.pending_size = 7
                core.session.analysis.cache[core.cache_key_for_moves([])] = ["cached"]
                self._assert_tree_state(core, history=[(1, 1)], future=[(2, 1)], variations=[(3, 1)])
                self._assert_failed_load_preserves_state(core, engine, getattr(core, method), text)

    def test_apply_pending_size_clears_engine_board_after_nonempty_position(self):
        core, engine = self._mk_core()

        core.try_play_move(1, 1)
        core.try_play_move(2, 1)

        engine.calls.clear()
        core.session.pending_size = 6

        self.assertTrue(core.apply_pending_size())
        self.assertEqual(core.board.n, 6)
        self.assertEqual(core.board.history, [])
        self.assertEqual(core.current_path_moves(), [])
        self.assertEqual(engine.calls, [("set_board_size", 6), ("clear_board",), ("clear_analysis",)])

    def test_get_active_analysis_preference(self):
        core, engine = self._mk_core()

        cache_key = core.cache_key()
        cached = [AnalysisMove("a1", order=1, col=1, row=1, winrate=0.5, visits=10, prior=0.2, pv=None)]
        core.session.analysis.cache[cache_key] = cached
        self.assertEqual(core.get_active_analysis(), cached)

        core.clear_all_cached_analysis()
        core.add_candidate(1, 1)
        engine.analysis = [
            AnalysisMove("a1", order=0, col=1, row=1, winrate=0.4, visits=5, prior=0.2, pv=None),
            AnalysisMove("b2", order=1, col=2, row=2, winrate=0.6, visits=8, prior=0.3, pv=None),
        ]
        active = core.get_active_analysis()
        self.assertEqual(len(active), 1)
        self.assertEqual((active[0].col, active[0].row), (1, 1))

        core.clear_candidates()
        core.set_analysis_enabled(True)
        engine.analysis = [
            AnalysisMove("b2", order=1, col=2, row=2, winrate=0.6, visits=8, prior=None, pv=None)
        ]
        self.assertEqual(core.get_active_analysis(), engine.analysis)

        core.set_analysis_enabled(False)
        self.assertEqual(core.get_active_analysis(), [])

    def test_start_batch_analysis_clears_candidates_and_starts_live(self):
        core, engine = self._mk_core()

        core.try_play_move(1, 1)
        core.step_back()
        core.add_candidate(2, 2)
        core.set_analysis_enabled(True)

        before = len(engine.calls)
        core.start_batch_analysis()
        new_calls = self._new_calls(engine, before)

        self.assertTrue(core.session.analysis.enabled)
        self.assertEqual(core.session.analysis.candidate_selection.candidates, set())
        self.assertTrue(core.is_batch_analysis_active())
        self.assertTrue(any(call[0] == "start_analysis" for call in new_calls))

    def test_batch_analysis_steps_forward_to_end(self):
        core, engine = self._mk_core()

        core.try_play_move(1, 1)
        core.try_play_move(2, 1)
        core.try_play_move(1, 2)
        self.assertEqual(len(core.board.history), 3)
        self.assertEqual(core.mainline_tail_moves(), [])
        engine.analysis = [
            AnalysisMove("a1", order=1, col=1, row=1, winrate=0.5, visits=1, prior=None, pv=None)
        ]

        core.start_batch_analysis()
        self.assertTrue(core.is_batch_analysis_active())
        self.assertEqual(len(core.board.history), 0)

        checkpoints = [
            (2.9, 0, True),
            (3.0, 0, True),
            (6.0, 1, True),
            (9.0, 1, True),
            (12.0, 2, True),
            (15.0, 2, True),
            (18.0, 3, True),
            (21.0, 3, True),
        ]
        for now, expected_history, expect_running in checkpoints:
            core.tick(now)
            self.assertEqual(len(core.board.history), expected_history)
            self.assertIs(expect_running, core.is_batch_analysis_active())

        core.tick(24.0)
        self.assertEqual(len(core.board.history), 3)
        self.assertEqual(core.mainline_tail_moves(), [])
        self.assertFalse(core.is_batch_analysis_active())
        self.assertFalse(core.session.analysis.enabled)

    def test_batch_analysis_restarts_live_analysis_after_step_forward(self):
        core, engine = self._mk_core()
        self._start_slow_batch(core, engine)
        before = len(engine.calls)
        core.tick(3.0)
        new_calls = self._new_calls(engine, before)
        self.assertEqual(len(core.board.history), 1)
        self.assertEqual(sum(1 for call in new_calls if call[0] == "start_analysis"), 1)

    def test_slow_batch_start_clears_stale_buffer_before_first_timing_update(self):
        core, engine = self._mk_core()
        run = self._start_slow_batch(core, engine, clear_live=True)
        self.assertTrue(core.is_batch_analysis_active())
        self.assertIsNone(run.first_update_at)

    def test_start_batch_analysis_from_end_restarts_analysis_once(self):
        core, engine = self._mk_core()
        core.try_play_move(1, 1)
        core.toggle_analysis()
        before = len(engine.calls)

        core.start_batch_analysis()
        new_calls = self._new_calls(engine, before)

        self.assertTrue(core.is_batch_analysis_active())
        self.assertEqual(len(core.board.history), 0)
        self.assertEqual(sum(1 for call in new_calls if call[0] == "start_analysis"), 1)

    def test_start_batch_analysis_selected_variation_cases(self):
        cases = (
            (
                "leaf rewinds but keeps selected variation line",
                lambda core: (core.try_play_move(1, 1), core.go_first(), core.try_play_move(3, 1), core.try_play_move(4, 1)),
                [(3, 1), (4, 1)],
                None,
                0,
                None,
                [(3, 1)],
            ),
            (
                "variation midline stays on variation",
                lambda core: (
                    core.try_play_move(1, 1),
                    core.go_first(),
                    core.try_play_move(3, 1),
                    core.try_play_move(4, 1),
                    core.step_back(),
                ),
                [(3, 1)],
                [(4, 1)],
                1,
                [(3, 1)],
                [(3, 1), (4, 1)],
            ),
        )
        for label, setup, history, future, ply, start_history, end_history in cases:
            with self.subTest(case=label):
                core, engine = self._mk_core()
                setup(core)
                engine.analysis = [
                    AnalysisMove("a1", order=1, col=1, row=1, winrate=0.5, visits=1, prior=None, pv=None)
                ]
                self._assert_tree_state(core, history=history, future=future)
                before = len(engine.calls)

                core.start_batch_analysis()
                new_calls = self._new_calls(engine, before)

                self.assertTrue(core.is_batch_analysis_active())
                self.assertEqual(core.current_ply(), ply)
                self.assertEqual(sum(1 for call in new_calls if call[0] == "start_analysis"), 1)

                if start_history is not None:
                    self._assert_tree_state(core, history=start_history)

                core.tick(0.0)
                core.tick(3.0)
                self._assert_tree_state(core, history=end_history)

    def test_batch_analysis_cancels_on_board_rev_change(self):
        core, _engine = self._mk_core()
        self._play_two_moves(core)
        core.go_first()
        core.start_batch_analysis()

        self.assertTrue(core.is_batch_analysis_active())
        core.board.place(Side.RED, 5, 5)
        core.tick(0.1)

        self.assertFalse(core.is_batch_analysis_active())

    def test_tree_only_edit_during_batch_cancels_immediately(self):
        core, _engine = self._mk_core()

        core.try_play_move(1, 1)
        core.go_first()
        core.try_play_move(3, 1)
        core.go_first()
        core.start_batch_analysis(fast=True)
        before_rev = core.board.rev

        self.assertTrue(core.is_batch_analysis_active())
        self.assertEqual(core.board.rev, before_rev)
        self.assertEqual(self._future_coords(core), [(1, 1)])

        core.delete_tail()

        self.assertFalse(core.is_batch_analysis_active())
        self.assertTrue(core.session.analysis.enabled)
        self.assertEqual(core.board.rev, before_rev)
        self.assertEqual(self._future_coords(core), [(3, 1)])
        self.assertEqual(self._history_coords(core), [])

    def test_manual_step_during_batch_cancels_and_restarts_live_immediately(self):
        core, engine = self._mk_core()
        self._play_two_moves(core)
        core.go_first()
        core.start_batch_analysis()

        self.assertTrue(core.is_batch_analysis_active())
        before = len(engine.calls)

        self.assertTrue(core.step_forward())
        new_calls = self._new_calls(engine, before)

        self.assertFalse(core.is_batch_analysis_active())
        self.assertTrue(core.session.analysis.enabled)
        self.assertEqual(sum(1 for call in new_calls if call[0] == "start_analysis"), 1)

        before = len(engine.calls)
        core.tick(0.1)
        tick_calls = self._new_calls(engine, before)
        self.assertEqual(sum(1 for call in tick_calls if call[0] == "start_analysis"), 0)

    def test_apply_pending_size_during_batch_cancels_and_restarts_live_immediately(self):
        core, engine = self._mk_core()
        self._play_two_moves(core)
        core.go_first()
        core.start_batch_analysis()

        self.assertTrue(core.is_batch_analysis_active())
        core.session.pending_size = 6
        before = len(engine.calls)

        self.assertTrue(core.apply_pending_size())
        new_calls = self._new_calls(engine, before)

        self.assertFalse(core.is_batch_analysis_active())
        self.assertTrue(core.session.analysis.enabled)
        self.assertEqual(sum(1 for call in new_calls if call[0] == "start_analysis"), 1)

        before = len(engine.calls)
        core.tick(0.1)
        tick_calls = self._new_calls(engine, before)
        self.assertEqual(sum(1 for call in tick_calls if call[0] == "start_analysis"), 0)

    def test_batch_cancel_clears_stale_engine_analysis_before_live_resume(self):
        core, engine = self._mk_core()

        core.start_batch_analysis()
        self.assertTrue(core.is_batch_analysis_active())

        engine.analysis = [
            AnalysisMove("a1", order=1, col=1, row=1, winrate=0.5, visits=1, prior=None, pv=None)
        ]

        def clear_analysis_realistic():
            engine.calls.append(("clear_analysis",))
            engine.analysis.clear()

        engine.clear_analysis = clear_analysis_realistic

        core.board.place(Side.RED, 1, 1)
        core.tick(0.1)

        self.assertFalse(core.is_batch_analysis_active())
        self.assertEqual(engine.analysis, [])
        self.assertNotIn(core.cache_key(), core.session.analysis.cache)

    def test_analysis_enabled_transitions_are_idempotent(self):
        for with_candidates in (False, True):
            with self.subTest(with_candidates=with_candidates):
                core, engine = self._mk_core()
                if with_candidates:
                    core.add_candidate(1, 1)

                core.set_analysis_enabled(False)
                self.assertFalse(engine.calls)
                before = len(engine.calls)
                core.set_analysis_enabled(True)
                new_calls = self._new_calls(engine, before)

                self.assertTrue(core.session.analysis.enabled)
                self.assertFalse(core.is_batch_analysis_active())
                if with_candidates:
                    self.assertEqual(core.session.analysis.candidate_selection.candidates, {(1, 1)})
                    self.assertTrue(any(call[0] == "start_analysis" and len(call) == 4 for call in new_calls))
                else:
                    self.assertEqual(core.session.analysis.candidate_selection.candidates, set())
                    self.assertTrue(any(call[0] == "start_analysis" for call in new_calls))
                before = len(engine.calls)
                core.set_analysis_enabled(True)
                self.assertFalse(self._new_calls(engine, before))

    def test_toggle_analysis_off_exits_batch_mode(self):
        core, engine = self._mk_core()
        self._start_slow_batch(core, engine)
        self.assertTrue(core.is_batch_analysis_active())
        self.assertTrue(core.session.analysis.enabled)
        core.toggle_analysis()
        core.tick(0.1)
        self.assertFalse(core.session.analysis.enabled)
        self.assertFalse(core.is_batch_analysis_active())

    def test_play_move_during_candidate_filter_cases(self):
        with self.subTest(case="candidate root change clears candidates and resumes live"):
            core, engine = self._mk_core()
            core.add_candidate(1, 1)
            core.set_analysis_enabled(True)

            before = len(engine.calls)

            core.try_play_move(2, 2)
            new_calls = self._new_calls(engine, before)

            self.assertTrue(core.session.analysis.enabled)
            self.assertEqual(core.session.analysis.candidate_selection.candidates, set())
            self.assertFalse(core.is_batch_analysis_active())
            self.assertTrue(any(call[0] == "start_analysis" for call in new_calls))

        with self.subTest(case="play move clears candidate filter before playing real move"):
            core, engine = self._mk_core()
            self._start_candidate_filter(core, 1, 1)

            before = len(engine.calls)
            core.try_play_move(2, 2)
            new_calls = self._new_calls(engine, before)
            stop_idx = self._index_of(new_calls, lambda call: call[0] == "stop_analysis")
            play_idx = self._index_of(
                new_calls,
                lambda call: call[0] == "play" and call[1] == Side.RED and call[2:] == (2, 2),
            )
            self.assertLess(stop_idx, play_idx)

    def test_candidates_use_root_allow_filter(self):
        core, engine = self._mk_core()
        core.set_analysis_enabled(True)
        engine.analysis = [
            AnalysisMove("b2", order=1, col=2, row=2, winrate=0.6, visits=20, prior=0.4, pv=None),
            AnalysisMove("c3", order=2, col=3, row=3, winrate=0.4, visits=10, prior=0.2, pv=None),
        ]

        def clear_analysis_realistic():
            engine.calls.append(("clear_analysis",))
            engine.analysis.clear()

        engine.clear_analysis = clear_analysis_realistic
        core.add_candidate(1, 1)
        core.add_candidate(2, 1)

        self.assertNotIn(("play", Side.RED, 1, 1), engine.calls)
        self.assertIn(("start_analysis", Side.RED, 15, ((Side.RED, [(1, 1), (2, 1)]),)), engine.calls)
        self.assertEqual(engine.params["analysisWideRootNoise"], core.session.analysis.wide_root_noise)
        self.assertEqual(core.get_top_move(), (None, 0))

        pv = ((1, 1), (3, 1))
        engine.analysis = [
            AnalysisMove("a1", order=0, col=1, row=1, winrate=0.7, visits=40, prior=0.8, pv=pv),
        ]
        core.maybe_update_analysis_cache()

        self.assertEqual(core.get_top_move(), ((1, 1), 40))
        candidate = core.get_candidate_analysis()[0]
        self.assertEqual((candidate.prior, candidate.pv), (0.8, pv))
        core.clear_candidates()
        self.assertEqual(core.get_top_move(), ((2, 2), 20))

    def test_candidate_analysis_keeps_displayed_rows_during_awrn_restart(self):
        core, engine = self._mk_core()
        self._start_candidate_filter(core, 1, 1)
        engine.analysis = [
            AnalysisMove("a1", order=0, col=1, row=1, winrate=0.7, visits=40, prior=0.8, pv=None),
        ]

        def clear_analysis_realistic():
            engine.calls.append(("clear_analysis",))
            engine.analysis.clear()

        engine.clear_analysis = clear_analysis_realistic

        core.set_analysis_wide_root_noise(0.12)

        self.assertEqual(engine.params["analysisWideRootNoise"], 0.12)
        self.assertEqual(core.candidate_result((1, 1)), (0.7, 40))

    def test_delete_tail_syncs_candidate_position(self):
        core, engine = self._mk_core()

        core.try_play_move(1, 1)
        self._start_candidate_filter(core, 2, 2)

        before = len(engine.calls)
        core.delete_tail()
        new_calls = self._new_calls(engine, before)
        self.assertEqual(new_calls.count(("undo",)), 1)

    def test_removing_last_candidate_resumes_live_analysis(self):
        for remover in (GuiCore.toggle_candidate, GuiCore.remove_candidate):
            with self.subTest(remover=remover.__name__):
                core, engine = self._mk_core()

                self._start_candidate_filter(core, 1, 1)
                engine.analysis = [
                    AnalysisMove("b2", order=0, col=2, row=2, winrate=0.6, visits=10, prior=0.4, pv=None)
                ]

                def clear_analysis_realistic():
                    engine.calls.append(("clear_analysis",))
                    engine.analysis.clear()

                engine.clear_analysis = clear_analysis_realistic

                before = len(engine.calls)
                remover(core, 1, 1)

                new_calls = self._new_calls(engine, before)
                self.assertTrue(any(call[0] == "clear_analysis" for call in new_calls))
                self.assertTrue(any(call[0] == "start_analysis" for call in new_calls))
                self.assertFalse(core.session.analysis.candidate_selection.candidates)
                self.assertEqual(engine.analysis, [])

    def test_delete_tail_behavior(self):
        for at_end in (False, True):
            with self.subTest(at_end=at_end):
                core, engine = self._mk_core()

                self._play_two_moves(core)
                if not at_end:
                    core.step_back()

                history_len = len(core.board.history)
                before = len(engine.calls)

                self.assertTrue(core.delete_tail())

                new_calls = self._new_calls(engine, before)
                if at_end:
                    self.assertEqual(len(core.board.history), history_len - 1)
                    self.assertIn(("undo",), new_calls)
                else:
                    self.assertEqual(len(core.board.history), history_len)
                    self.assertNotIn(("undo",), new_calls)
                self.assertEqual(core.mainline_tail_moves(), [])

    def test_delete_tail_keeps_future_cache_when_truncating(self):
        core, _engine = self._mk_core()

        self._play_two_moves(core)
        key0 = core.cache_key_for_moves([])
        key1 = core.cache_key_for_moves(self._path_moves(core)[:1])
        key2 = core.cache_key_for_moves(self._path_moves(core)[:2])
        core.session.analysis.cache[key0] = ["a"]
        core.session.analysis.cache[key1] = ["b"]
        core.session.analysis.cache[key2] = ["c"]

        core.go_first()
        self.assertTrue(core.mainline_tail_moves())

        core.delete_tail()

        self.assertEqual(core.mainline_tail_moves(), [])
        self.assertIn(key0, core.session.analysis.cache)
        self.assertIn(key1, core.session.analysis.cache)
        self.assertIn(key2, core.session.analysis.cache)


if __name__ == "__main__":
    unittest.main()
