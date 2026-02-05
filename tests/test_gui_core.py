import unittest

from board import HexBoard, Move, Side
from engine import AnalysisMove
from gui_core import GuiCore


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

    def start_analysis(self, side, interval_cs=15):
        self.calls.append(("start_analysis", side, interval_cs))

    def stop_analysis(self):
        self.calls.append(("stop_analysis",))

    def kata_set_param(self, name, value):
        self.params[name] = value
        self.calls.append(("kata_set_param", name, value))

    def play(self, side, col, row):
        self.calls.append(("play", side, col, row))
        self.played.append((side, col, row))

    def undo(self):
        self.calls.append(("undo",))

    def clear_board(self):
        self.calls.append(("clear_board",))
        self.played = []

    def set_board_size(self, n):
        self.calls.append(("set_board_size", n))

    def get_analysis(self):
        return list(self.analysis)


class GuiCoreTests(unittest.TestCase):
    def _mk_core(self):
        board = HexBoard(5)
        engine = FakeEngine()
        core = GuiCore(board, engine)
        return core, engine

    def _start_candidate_run(self, core: GuiCore, col: int, row: int, *, now: float = 0.0):
        core.add_candidate(col, row)
        core._set_analysis_enabled(True)
        core.step_candidate_search(now=now)

    def _play_two_moves(self, core: GuiCore) -> None:
        core.try_play_move(1, 1)
        core.try_play_move(2, 1)

    def _new_calls(self, engine: FakeEngine, before: int):
        return engine.calls[before:]

    def _index_of(self, calls, predicate):
        return next(i for i, call in enumerate(calls) if predicate(call))

    def _assert_undo_before(self, calls, predicate):
        undo_idx = self._index_of(calls, lambda call: call[0] == "undo")
        target_idx = self._index_of(calls, predicate)
        self.assertLess(undo_idx, target_idx)

    def _assert_has_undo(self, calls):
        self.assertTrue(any(call[0] == "undo" for call in calls))

    def _assert_no_undo(self, calls):
        self.assertFalse(any(call[0] == "undo" for call in calls))

    def test_history_future_roundtrip(self):
        core, _engine = self._mk_core()
        board = core.board

        core.try_play_move(1, 1)
        core.try_play_move(2, 1)
        core.try_play_move(1, 2)

        self.assertEqual(len(board.history), 3)
        self.assertEqual(core.app.future_moves, [])

        last = board.history[-1]
        self.assertTrue(core.step_back())
        self.assertEqual(len(board.history), 2)
        self.assertEqual(len(core.app.future_moves), 1)
        self.assertEqual(core.move_coords(core.app.future_moves[-1]), core.move_coords(last))

        self.assertTrue(core.step_forward())
        self.assertEqual(len(board.history), 3)
        self.assertEqual(core.app.future_moves, [])
        self.assertEqual(core.move_coords(board.history[-1]), core.move_coords(last))

    def test_step_back_n_steps_as_much_as_possible(self):
        core, _engine = self._mk_core()

        self._play_two_moves(core)

        self.assertTrue(core.step_back_n(10))
        self.assertEqual(len(core.board.history), 0)
        self.assertEqual(len(core.app.future_moves), 2)

    def test_step_forward_n_steps_as_much_as_possible(self):
        core, _engine = self._mk_core()

        self._play_two_moves(core)
        core.step_back_n(10)

        self.assertTrue(core.step_forward_n(10))
        self.assertEqual(len(core.board.history), 2)
        self.assertEqual(core.app.future_moves, [])

    def test_truncate_future_moves_on_conflict(self):
        core, _engine = self._mk_core()
        board = core.board

        board.place(Side.RED, 1, 1)
        board.place(Side.BLUE, 2, 1)

        core.app.future_moves = [
            Move.place(Side.RED, 3, 1),
            Move.place(Side.BLUE, 2, 1),
        ]

        core.truncate_future_moves_on_conflict()
        self.assertEqual(core.app.future_moves, [])

    def test_rebuild_engine_from_history(self):
        core, engine = self._mk_core()
        board = core.board

        board.place(Side.RED, 1, 1)
        board.place(Side.BLUE, 2, 1)
        board.pass_move(Side.RED)

        core.rebuild_engine_from_history()

        expected = [
            ("clear_board",),
            ("play", Side.RED, 1, 1),
            ("play", Side.BLUE, 2, 1),
            ("play", Side.RED, None, None),
        ]
        self.assertEqual(engine.calls, expected)

    def test_candidate_root_rev_invalidation(self):
        core, _engine = self._mk_core()
        board = core.board

        core.add_candidate(1, 1)
        self.assertTrue(core.app.candidates)
        root_rev = core.app.candidate_root_rev
        self.assertIsNotNone(root_rev)

        board.place(Side.RED, 2, 2)
        self.assertNotEqual(board.rev, root_rev)

        cleared = core.check_candidate_root()
        self.assertTrue(cleared)
        self.assertEqual(core.app.candidates, set())
        self.assertIsNone(core.app.candidate_root_rev)

    def test_merge_analysis_lists_prefers_primary_order_prior(self):
        core, _engine = self._mk_core()

        primary = [
            AnalysisMove("a1", order=1, col=1, row=1, winrate=0.4, visits=10, prior=0.2, pv=None),
        ]
        secondary = [
            AnalysisMove("a1", order=9, col=1, row=1, winrate=0.7, visits=50, prior=0.9, pv=None),
        ]

        merged = core._merge_analysis_lists(primary, secondary)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].order, 1)
        self.assertEqual(merged[0].prior, 0.2)
        self.assertEqual(merged[0].winrate, 0.7)
        self.assertEqual(merged[0].visits, 50)

    def test_with_analysis_paused_stops_candidate_run_when_stop_engine_false(self):
        core, engine = self._mk_core()
        self._start_candidate_run(core, 1, 1)
        self.assertTrue(any(call[0] == "play" for call in engine.calls))

        core.with_analysis_paused(lambda: None, stop_engine=False)

        self.assertIsNone(core.app.candidate_run)
        self.assertIn(("undo",), engine.calls)
        # Analysis should resume (candidates exist), which plays the candidate move.
        self.assertTrue(any(call[0] == "play" for call in engine.calls))

    def test_toggle_analysis_switches_between_live_and_candidate(self):
        core, engine = self._mk_core()

        core.toggle_analysis()
        self.assertTrue(core.app.analysis_running)
        self.assertIn(("start_analysis", Side.RED, core.analyze_interval_cs), engine.calls)

        core.add_candidate(1, 1)
        core.toggle_analysis()  # stop
        core.toggle_analysis()  # start with candidates

        self.assertTrue(core.app.analysis_running)
        core.step_candidate_search(now=0.0)
        self.assertTrue(any(call[0] == "play" for call in engine.calls))
        self.assertTrue(any(call[0] == "start_analysis" for call in engine.calls))

    def test_cache_prune_delete_tail_preserves_current_ply(self):
        core, _engine = self._mk_core()

        self._play_two_moves(core)

        key0 = (0, int(Side.RED))
        key1 = (1, int(Side.BLUE))
        key2 = (2, int(Side.RED))
        core.app.analysis_cache[key0] = ["a"]
        core.app.analysis_cache[key1] = ["b"]
        core.app.analysis_cache[key2] = ["c"]

        core.delete_tail()  # removes last move; history length back to 1

        self.assertIn(key0, core.app.analysis_cache)
        self.assertIn(key1, core.app.analysis_cache)
        self.assertNotIn(key2, core.app.analysis_cache)

    def test_cache_prune_on_branching_clears_future_entries(self):
        core, _engine = self._mk_core()

        self._play_two_moves(core)
        core.step_back()  # future_moves has one

        key0 = (0, int(Side.RED))
        key1 = (1, int(Side.BLUE))
        key2 = (2, int(Side.RED))
        core.app.analysis_cache[key0] = ["a"]
        core.app.analysis_cache[key1] = ["b"]
        core.app.analysis_cache[key2] = ["c"]

        core.try_play_move(3, 1)  # diverge, should clear future + prune caches >= new ply

        self.assertEqual(core.app.future_moves, [])
        self.assertIn(key0, core.app.analysis_cache)
        self.assertIn(key1, core.app.analysis_cache)
        self.assertNotIn(key2, core.app.analysis_cache)

    def test_get_active_analysis_preference(self):
        core, engine = self._mk_core()

        cache_key = core.cache_key()
        cached = [AnalysisMove("a1", order=1, col=1, row=1, winrate=0.5, visits=10, prior=0.2, pv=None)]
        core.app.analysis_cache[cache_key] = cached
        self.assertEqual(core.get_active_analysis(), cached)

        core.clear_all_cached_analysis()
        core.add_candidate(1, 1)
        core.app.candidate_results[(1, 1)] = (0.4, 5)
        active = core.get_active_analysis()
        self.assertEqual(len(active), 1)
        self.assertEqual((active[0].col, active[0].row), (1, 1))

        core.clear_candidates()
        core._set_analysis_enabled(True)
        engine.analysis = [
            AnalysisMove("b2", order=1, col=2, row=2, winrate=0.6, visits=8, prior=None, pv=None)
        ]
        self.assertEqual(core.get_active_analysis(), engine.analysis)

        core._set_analysis_enabled(False)
        self.assertEqual(core.get_active_analysis(), [])

    def test_single_candidate_does_not_rotate_or_undo(self):
        core, engine = self._mk_core()

        self._start_candidate_run(core, 1, 1)

        undo_before = sum(1 for call in engine.calls if call[0] == "undo")
        core.step_candidate_search(now=2.0)
        undo_after = sum(1 for call in engine.calls if call[0] == "undo")

        self.assertEqual(undo_before, undo_after)
        self.assertIsNotNone(core.app.candidate_run)

    def test_play_move_undoes_candidate_run_first(self):
        core, engine = self._mk_core()

        self._start_candidate_run(core, 1, 1)

        # Play a different move than the active candidate.
        core.try_play_move(2, 2)

        self._assert_undo_before(
            engine.calls,
            lambda call: call[0] == "play" and call[1] == Side.RED and call[2:] == (2, 2),
        )

    def test_delete_tail_undoes_candidate_and_history(self):
        core, engine = self._mk_core()

        core.try_play_move(1, 1)
        self._start_candidate_run(core, 2, 2)

        before = len(engine.calls)
        core.delete_tail()
        new_calls = self._new_calls(engine, before)
        undo_calls = [call for call in new_calls if call[0] == "undo"]

        self.assertEqual(len(undo_calls), 2)

    def test_removing_last_candidate_resumes_live_analysis(self):
        core, engine = self._mk_core()

        self._start_candidate_run(core, 1, 1)

        before = len(engine.calls)
        core.toggle_candidate(1, 1)

        new_calls = self._new_calls(engine, before)
        self._assert_undo_before(new_calls, lambda call: call[0] == "start_analysis")
        self.assertFalse(core.app.candidates)

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
                    self._assert_has_undo(new_calls)
                else:
                    self.assertEqual(len(core.board.history), history_len)
                    self._assert_no_undo(new_calls)
                self.assertEqual(core.app.future_moves, [])

    def test_delete_tail_prunes_future_cache_when_truncating(self):
        core, _engine = self._mk_core()

        self._play_two_moves(core)
        core.app.analysis_cache[(0, int(Side.RED))] = ["a"]
        core.app.analysis_cache[(1, int(Side.BLUE))] = ["b"]
        core.app.analysis_cache[(2, int(Side.RED))] = ["c"]

        core.go_first()
        self.assertTrue(core.app.future_moves)

        core.delete_tail()

        self.assertEqual(core.app.future_moves, [])
        self.assertIn((0, int(Side.RED)), core.app.analysis_cache)
        self.assertNotIn((1, int(Side.BLUE)), core.app.analysis_cache)
        self.assertNotIn((2, int(Side.RED)), core.app.analysis_cache)


if __name__ == "__main__":
    unittest.main()
