import unittest

from board import HexBoard, Move, MoveKind, Side
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
        core._apply_analysis_enabled_transition(True)
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

    def _history_coords(self, core: GuiCore):
        return [core.move_coords(mv) for mv in core.board.history]

    def _future_coords(self, core: GuiCore):
        # future_moves is stored as a stack (next redo at the end).
        return [core.move_coords(mv) for mv in reversed(core.app.future_moves)]

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

    def test_step_back_and_forward_n_steps_as_much_as_possible(self):
        core, _engine = self._mk_core()

        self._play_two_moves(core)
        self.assertTrue(core.step_back_n(10))
        self.assertEqual(len(core.board.history), 0)
        self.assertEqual(len(core.app.future_moves), 2)

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
        self.assertTrue(core.app.candidate_state.candidates)
        root_rev = core.app.candidate_state.root_rev
        self.assertIsNotNone(root_rev)

        board.place(Side.RED, 2, 2)
        self.assertNotEqual(board.rev, root_rev)

        cleared = core.check_candidate_root()
        self.assertTrue(cleared)
        self.assertEqual(core.app.candidate_state.candidates, set())
        self.assertIsNone(core.app.candidate_state.root_rev)

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

        self.assertIsNone(core.app.candidate_state.run)
        self.assertIn(("undo",), engine.calls)
        # Analysis should resume (candidates exist), which plays the candidate move.
        self.assertTrue(any(call[0] == "play" for call in engine.calls))

    def test_delete_tail_keeps_existing_cache_entries(self):
        core, _engine = self._mk_core()

        self._play_two_moves(core)

        key0 = core.cache_key_for_moves([])
        key1 = core.cache_key_for_moves(core.board.history[:1])
        key2 = core.cache_key_for_moves(core.board.history[:2])
        core.app.analysis_cache[key0] = ["a"]
        core.app.analysis_cache[key1] = ["b"]
        core.app.analysis_cache[key2] = ["c"]

        core.delete_tail()  # removes last move; history length back to 1

        self.assertIn(key0, core.app.analysis_cache)
        self.assertIn(key1, core.app.analysis_cache)
        self.assertIn(key2, core.app.analysis_cache)

    def test_branching_clears_future_entries_but_keeps_cache(self):
        core, _engine = self._mk_core()

        self._play_two_moves(core)
        core.step_back()  # future_moves has one

        key0 = core.cache_key_for_moves([])
        key1 = core.cache_key_for_moves(core.board.history[:1])
        key2 = core.cache_key_for_moves(core.board.history[:2])
        core.app.analysis_cache[key0] = ["a"]
        core.app.analysis_cache[key1] = ["b"]
        core.app.analysis_cache[key2] = ["c"]

        core.try_play_move(3, 1)  # diverge, should clear future

        self.assertEqual(core.app.future_moves, [])
        self.assertIn(key0, core.app.analysis_cache)
        self.assertIn(key1, core.app.analysis_cache)
        self.assertIn(key2, core.app.analysis_cache)

    def test_try_play_moves_replays_redo_prefix_then_branches_and_keeps_cache(self):
        core, _engine = self._mk_core()

        core.try_play_move(1, 1)
        core.try_play_move(2, 1)
        core.try_play_move(1, 2)
        core.step_back_n(2)

        key0 = core.cache_key_for_moves([])
        key1 = core.cache_key_for_moves(core.board.history[:1])
        key2 = core.cache_key_for_moves(core.board.history[:2])
        key3 = core.cache_key_for_moves(core.board.history[:3])
        core.app.analysis_cache[key0] = ["a"]
        core.app.analysis_cache[key1] = ["b"]
        core.app.analysis_cache[key2] = ["c"]
        core.app.analysis_cache[key3] = ["d"]

        did = core.try_play_moves([(2, 1), (3, 1)])

        self.assertTrue(did)
        self.assertEqual(self._history_coords(core), [(1, 1), (2, 1), (3, 1)])
        self.assertEqual(core.app.future_moves, [])
        self.assertIn(key0, core.app.analysis_cache)
        self.assertIn(key1, core.app.analysis_cache)
        self.assertIn(key2, core.app.analysis_cache)
        self.assertIn(key3, core.app.analysis_cache)

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

        core.try_play_move(3, 2)  # c2
        self.assertTrue(core.try_swap_move())
        core.try_play_move(5, 4)  # e4

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

        core.try_play_move(3, 2)  # c2
        core.try_swap_move()
        core.try_play_move(5, 4)  # e4

        moves = list(core.board.history)
        labels = [core.move_to_label_in_sequence(moves, i) for i in range(len(moves))]
        self.assertEqual(labels, ["c2", "swap", "e4"])

    def test_try_swap_move_prefers_redo_swap_over_new_swap(self):
        core, _engine = self._mk_core()

        core.try_play_move(3, 2)  # c2
        core.try_swap_move()
        core.try_play_move(5, 4)  # e4
        core.step_back_n(2)  # now at c2 with swap/e4 in future

        did = core.try_swap_move()

        self.assertTrue(did)
        self.assertEqual([mv.kind for mv in core.board.history], [MoveKind.PLACE, MoveKind.SWAP])
        self.assertEqual(self._history_coords(core), [(2, 3), None])
        self.assertEqual(self._future_coords(core), [(5, 4)])

    def test_drag_first_move_with_future_swap_updates_swap_and_truncates_conflict(self):
        core, _engine = self._mk_core()

        core.try_play_move(3, 2)  # c2
        core.try_swap_move()
        core.try_play_move(2, 4)  # b4
        core.step_back_n(2)  # history: c2, future: swap, b4

        did = core.try_drag_move(0, (3, 2), 4, 2)  # d2 -> swap target b4

        self.assertTrue(did)
        self.assertEqual(self._history_coords(core), [(4, 2)])
        self.assertEqual([mv.kind for mv in reversed(core.app.future_moves)], [MoveKind.SWAP])
        swap_mv = core.app.future_moves[-1]
        self.assertEqual((swap_mv.col, swap_mv.row), (4, 2))
        seq = list(core.board.history) + list(reversed(core.app.future_moves))
        self.assertEqual([core.move_to_label_in_sequence(seq, i) for i in range(len(seq))], ["d2", "swap"])

    def test_drag_swapped_stone_truncates_future_on_stone_conflict(self):
        core, _engine = self._mk_core()

        core.try_play_move(3, 2)  # c2
        core.try_swap_move()
        core.try_play_move(5, 4)  # e4
        core.step_back()  # history: c2,swap ; future: e4

        did = core.try_drag_move(0, (2, 3), 5, 4)  # drag swapped stone onto future e4

        self.assertTrue(did)
        self.assertEqual(core.app.future_moves, [])
        self.assertEqual([mv.kind for mv in core.board.history], [MoveKind.PLACE, MoveKind.SWAP])
        self.assertEqual((core.board.history[1].col, core.board.history[1].row), (4, 5))

    def test_drag_in_swapped_position_keeps_legal_future_moves(self):
        core, _engine = self._mk_core()

        core.try_play_move(3, 2)  # c2
        core.try_swap_move()
        core.try_play_move(5, 4)  # e4
        core.try_play_move(1, 5)  # a5
        core.step_back()  # history: c2,swap,e4 ; future: a5

        did = core.try_drag_move(2, (5, 4), 5, 5)  # e4 -> e5

        self.assertTrue(did)
        self.assertEqual(self._history_coords(core), [(2, 3), None, (5, 5)])
        self.assertEqual(self._future_coords(core), [(1, 5)])

    def test_drag_first_move_with_future_swap_keeps_legal_reuse(self):
        core, _engine = self._mk_core()

        core.try_play_move(3, 2)  # c2
        core.try_swap_move()
        core.try_play_move(4, 2)  # d2
        core.step_back_n(2)  # history: c2, future: swap, d2

        did = core.try_drag_move(0, (3, 2), 4, 2)  # c2 -> d2

        self.assertTrue(did)
        self.assertEqual(self._history_coords(core), [(4, 2)])
        self.assertEqual([mv.kind for mv in reversed(core.app.future_moves)], [MoveKind.SWAP, MoveKind.PLACE])
        seq = list(core.board.history) + list(reversed(core.app.future_moves))
        self.assertEqual([core.move_to_label_in_sequence(seq, i) for i in range(len(seq))], ["d2", "swap", "d2"])

        core.step_forward_n(2)
        self.assertEqual(self._history_coords(core), [(2, 4), None, (4, 2)])
        self.assertEqual(core.board.get(2, 4), int(Side.BLUE))
        self.assertEqual(core.board.get(4, 2), int(Side.RED))

    def test_build_hexworld_url_emits_swap_token(self):
        core, _engine = self._mk_core()

        core.try_play_move(3, 2)  # c2
        core.try_swap_move()
        core.try_play_move(5, 4)  # e4

        self.assertEqual(core.build_hexworld_url(), "https://hexworld.org/board/#5c1,c2:se4")

    def test_load_hexworld_text_with_swap(self):
        core, engine = self._mk_core()

        ok = core.load_hexworld_text("https://hexworld.org/board/#5c1,c2:se4")

        self.assertTrue(ok)
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

        ok = core.load_hexworld_text("https://hexworld.org/board/#5c1,c2:sc2")

        self.assertTrue(ok)
        self.assertEqual([mv.kind for mv in core.board.history], [MoveKind.PLACE, MoveKind.SWAP, MoveKind.PLACE])
        self.assertEqual(core.board.get(2, 3), int(Side.BLUE))  # swapped opening stone
        self.assertEqual(core.board.get(3, 2), int(Side.RED))  # legal reuse of c2 after swap

    def test_swap_branch_uses_distinct_cache_key(self):
        core, _engine = self._mk_core()

        core.try_play_move(3, 2)  # c2
        core.try_play_move(4, 2)  # d2
        stale_key = core.cache_key()
        core.app.analysis_cache[stale_key] = [
            AnalysisMove("a1", order=1, col=1, row=1, winrate=0.5, visits=100, prior=0.1, pv=None)
        ]

        core.step_back()  # history: c2 ; future: d2
        did = core.try_swap_move()  # branch to swap at ply 2

        self.assertTrue(did)
        self.assertNotEqual(core.cache_key(), stale_key)
        self.assertIn(stale_key, core.app.analysis_cache)
        self.assertEqual(core.get_active_analysis(), [])

    def test_clear_analysis_caches_while_candidate_mode_resets_results_but_keeps_candidates(self):
        core, _engine = self._mk_core()

        core.add_candidate(1, 1)
        core.add_candidate(2, 2)
        core._apply_analysis_enabled_transition(True)
        core.step_candidate_search(now=0.0)
        self.assertIsNotNone(core.app.candidate_state.run)

        core.app.candidate_state.results[(1, 1)] = (0.4, 10)
        core.app.analysis_cache[core.cache_key_for_moves([])] = ["x"]

        core.clear_analysis_caches()

        self.assertTrue(core.app.analysis_running)
        self.assertEqual(core.app.candidate_state.candidates, {(1, 1), (2, 2)})
        self.assertEqual(core.app.candidate_state.results, {})
        self.assertEqual(core.app.analysis_cache, {})
        self.assertIsNone(core.app.candidate_state.run)

    def test_load_hexworld_text_duplicate_rejected_without_mutating_state(self):
        core, engine = self._mk_core()

        core.try_play_move(1, 1)
        core.try_play_move(2, 1)
        core.step_back()
        core.app.pending_size = 7
        core.app.analysis_cache[core.cache_key_for_moves([])] = ["cached"]

        before_history = list(core.board.history)
        before_future = list(core.app.future_moves)
        before_pending = core.app.pending_size
        before_cache = dict(core.app.analysis_cache)
        before_n = core.board.n
        before_rev = core.board.rev
        before_calls = list(engine.calls)

        ok = core.load_hexworld_text("https://hexworld.org/board/#5c1,a1a1")

        self.assertFalse(ok)
        self.assertEqual(core.board.n, before_n)
        self.assertEqual(core.board.rev, before_rev)
        self.assertEqual(core.board.history, before_history)
        self.assertEqual(core.app.future_moves, before_future)
        self.assertEqual(core.app.pending_size, before_pending)
        self.assertEqual(core.app.analysis_cache, before_cache)
        self.assertEqual(engine.calls, before_calls)

    def test_get_active_analysis_preference(self):
        core, engine = self._mk_core()

        cache_key = core.cache_key()
        cached = [AnalysisMove("a1", order=1, col=1, row=1, winrate=0.5, visits=10, prior=0.2, pv=None)]
        core.app.analysis_cache[cache_key] = cached
        self.assertEqual(core.get_active_analysis(), cached)

        core.clear_all_cached_analysis()
        core.add_candidate(1, 1)
        core.app.candidate_state.results[(1, 1)] = (0.4, 5)
        active = core.get_active_analysis()
        self.assertEqual(len(active), 1)
        self.assertEqual((active[0].col, active[0].row), (1, 1))

        core.clear_candidates()
        core._apply_analysis_enabled_transition(True)
        engine.analysis = [
            AnalysisMove("b2", order=1, col=2, row=2, winrate=0.6, visits=8, prior=None, pv=None)
        ]
        self.assertEqual(core.get_active_analysis(), engine.analysis)

        core._apply_analysis_enabled_transition(False)
        self.assertEqual(core.get_active_analysis(), [])

    def test_start_batch_analysis_clears_candidates_and_starts_live(self):
        core, engine = self._mk_core()

        core.try_play_move(1, 1)
        core.step_back()
        core.add_candidate(2, 2)
        core._apply_analysis_enabled_transition(True)
        core.step_candidate_search(now=0.0)
        self.assertIsNotNone(core.app.candidate_state.run)

        before = len(engine.calls)
        core.start_batch_analysis()
        new_calls = self._new_calls(engine, before)

        self.assertTrue(core.app.analysis_running)
        self.assertEqual(core.app.candidate_state.candidates, set())
        self.assertIsNone(core.app.candidate_state.run)
        self.assertTrue(core.is_batch_analysis_active())
        self.assertTrue(any(call[0] == "start_analysis" for call in new_calls))

    def test_batch_analysis_steps_forward_to_end(self):
        core, engine = self._mk_core()

        core.try_play_move(1, 1)
        core.try_play_move(2, 1)
        core.try_play_move(1, 2)
        self.assertEqual(len(core.board.history), 3)
        self.assertEqual(core.app.future_moves, [])
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
        self.assertEqual(core.app.future_moves, [])
        self.assertFalse(core.is_batch_analysis_active())
        self.assertFalse(core.app.analysis_running)

    def test_batch_analysis_restarts_live_analysis_after_step_forward(self):
        core, engine = self._mk_core()

        core.try_play_move(1, 1)
        core.try_play_move(2, 1)
        core.go_first()
        engine.analysis = [
            AnalysisMove("a1", order=1, col=1, row=1, winrate=0.5, visits=1, prior=None, pv=None)
        ]

        core.start_batch_analysis()
        core.tick(0.0)  # set first_update_at
        before = len(engine.calls)
        core.tick(3.0)  # step forward
        new_calls = self._new_calls(engine, before)

        self.assertEqual(len(core.board.history), 1)
        self.assertTrue(any(call[0] == "start_analysis" for call in new_calls))

    def test_batch_analysis_cancels_on_board_rev_change(self):
        core, _engine = self._mk_core()

        core.try_play_move(1, 1)
        core.try_play_move(2, 1)
        core.go_first()
        core.start_batch_analysis()
        self.assertTrue(core.is_batch_analysis_active())

        core.board.place(Side.RED, 5, 5)
        core.tick(0.1)

        self.assertFalse(core.is_batch_analysis_active())

    def test_batch_cancel_restores_live_analysis_after_stop_engine_pause(self):
        core, engine = self._mk_core()

        core.try_play_move(1, 1)
        core.try_play_move(2, 1)
        core.go_first()
        core.start_batch_analysis()
        self.assertTrue(core.is_batch_analysis_active())

        core.app.pending_size = 6
        core.apply_pending_size()
        self.assertTrue(core.is_batch_analysis_active())

        before = len(engine.calls)
        core.tick(0.1)
        new_calls = self._new_calls(engine, before)

        self.assertFalse(core.is_batch_analysis_active())
        self.assertTrue(core.app.analysis_running)
        self.assertTrue(any(call[0] == "start_analysis" for call in new_calls))

    def test_enable_analysis_mode_transitions(self):
        for with_candidates in (False, True):
            with self.subTest(with_candidates=with_candidates):
                core, engine = self._mk_core()
                if with_candidates:
                    core.add_candidate(1, 1)

                before = len(engine.calls)
                core._apply_analysis_enabled_transition(True)
                new_calls = self._new_calls(engine, before)

                self.assertTrue(core.app.analysis_running)
                self.assertFalse(core.is_batch_analysis_active())

                if with_candidates:
                    self.assertEqual(core.app.candidate_state.candidates, {(1, 1)})
                    self.assertIsNone(core.app.candidate_state.run)
                    core.step_candidate_search(now=0.0)
                    self.assertIsNotNone(core.app.candidate_state.run)
                else:
                    self.assertEqual(core.app.candidate_state.candidates, set())
                    self.assertIsNone(core.app.candidate_state.run)
                    self.assertTrue(any(call[0] == "start_analysis" for call in new_calls))

    def test_toggle_analysis_off_exits_batch_mode(self):
        core, engine = self._mk_core()

        core.try_play_move(1, 1)
        core.try_play_move(2, 1)
        core.go_first()
        engine.analysis = [
            AnalysisMove("a1", order=1, col=1, row=1, winrate=0.5, visits=1, prior=None, pv=None)
        ]
        core.start_batch_analysis()
        self.assertTrue(core.is_batch_analysis_active())
        self.assertTrue(core.app.analysis_running)

        core.toggle_analysis()
        core.tick(0.1)

        self.assertFalse(core.app.analysis_running)
        self.assertFalse(core.is_batch_analysis_active())

    def test_candidate_root_change_clears_candidates_and_resumes_live(self):
        core, engine = self._mk_core()

        core.add_candidate(1, 1)
        core._apply_analysis_enabled_transition(True)
        core.step_candidate_search(now=0.0)
        self.assertIsNotNone(core.app.candidate_state.run)

        before = len(engine.calls)
        core.try_play_move(2, 2)
        new_calls = self._new_calls(engine, before)

        self.assertTrue(core.app.analysis_running)
        self.assertEqual(core.app.candidate_state.candidates, set())
        self.assertIsNone(core.app.candidate_state.run)
        self.assertFalse(core.is_batch_analysis_active())
        self.assertTrue(any(call[0] == "start_analysis" for call in new_calls))

    def test_single_candidate_does_not_rotate_or_undo(self):
        core, engine = self._mk_core()

        self._start_candidate_run(core, 1, 1)

        undo_before = sum(1 for call in engine.calls if call[0] == "undo")
        core.step_candidate_search(now=2.0)
        undo_after = sum(1 for call in engine.calls if call[0] == "undo")

        self.assertEqual(undo_before, undo_after)
        self.assertIsNotNone(core.app.candidate_state.run)

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
        self.assertFalse(core.app.candidate_state.candidates)

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

    def test_delete_tail_keeps_future_cache_when_truncating(self):
        core, _engine = self._mk_core()

        self._play_two_moves(core)
        key0 = core.cache_key_for_moves([])
        key1 = core.cache_key_for_moves(core.board.history[:1])
        key2 = core.cache_key_for_moves(core.board.history[:2])
        core.app.analysis_cache[key0] = ["a"]
        core.app.analysis_cache[key1] = ["b"]
        core.app.analysis_cache[key2] = ["c"]

        core.go_first()
        self.assertTrue(core.app.future_moves)

        core.delete_tail()

        self.assertEqual(core.app.future_moves, [])
        self.assertIn(key0, core.app.analysis_cache)
        self.assertIn(key1, core.app.analysis_cache)
        self.assertIn(key2, core.app.analysis_cache)


if __name__ == "__main__":
    unittest.main()
