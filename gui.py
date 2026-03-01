from __future__ import annotations

import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
)

import pygame

from board import MAX_BOARD_SIZE, MIN_BOARD_SIZE, HexBoard
from engine import KataHexEngine
from gui_core import DEFAULT_ANALYZE_INTERVAL_CS, GuiCore
from gui_render import GuiRenderer


@dataclass
class UiState:
    prefs: UiPrefs
    drag_select: bool = False
    drag_added: bool = False
    drag_last_cell: Optional[Tuple[int, int]] = None
    drag_start_candidates: Optional[set[Tuple[int, int]]] = None
    drag_move: bool = False
    drag_move_from: Optional[Tuple[int, int]] = None
    drag_move_idx: Optional[int] = None
    hover_cell: Optional[Tuple[int, int]] = None
    show_help: bool = False
    show_engine_debug: bool = False
    last_cand_display: Optional[Tuple[int, int]] = None
    speed_last_t: Optional[float] = None
    speed_last_total: Optional[int] = None
    speed_vps: Optional[float] = None
    swap_click_candidate: bool = False


@dataclass
class UiPrefs:
    show_move_numbers: bool = False
    show_elo: bool = False


def run_gui(
    board: HexBoard,
    engine: KataHexEngine,
    *,
    analyze_interval_cs: int = DEFAULT_ANALYZE_INTERVAL_CS,
    ui_prefs: UiPrefs,
) -> None:
    os.environ.setdefault("SDL_VIDEO_ALLOW_HIGHDPI", "1")
    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")

    core = GuiCore(board, engine, analyze_interval_cs=analyze_interval_cs)
    app = core.app

    pygame.init()
    pygame.freetype.init()
    pygame.key.set_repeat(400, 33)

    flags = pygame.RESIZABLE
    renderer = GuiRenderer(board, core, flags=flags)
    clock = pygame.time.Clock()

    scrap_ok = False
    try:
        pygame.scrap.init()
        pygame.scrap.set_mode(pygame.SCRAP_CLIPBOARD)
        scrap_ok = True
    except Exception:
        scrap_ok = False

    def _get_clipboard_fallback() -> Optional[str]:
        if sys.platform != "darwin":
            return None
        try:
            out = subprocess.run(["pbpaste"], check=False, capture_output=True, text=True)
        except Exception:
            return None
        text = out.stdout.strip()
        return text if text else None

    def get_clipboard_text() -> Optional[str]:
        if not scrap_ok:
            return _get_clipboard_fallback()
        try:
            raw = pygame.scrap.get(pygame.SCRAP_TEXT)
        except Exception:
            return _get_clipboard_fallback()
        if not raw:
            return _get_clipboard_fallback()
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", errors="replace")
        else:
            text = str(raw)
        text = text.split("\x00", 1)[0]
        return text.strip()

    def set_clipboard_text(text: str) -> bool:
        if scrap_ok:
            try:
                pygame.scrap.put(pygame.SCRAP_TEXT, text.encode("utf-8"))
                return True
            except Exception:
                pass
        if sys.platform == "darwin":
            try:
                subprocess.run(
                    ["pbcopy"],
                    input=text,
                    check=False,
                    text=True,
                    capture_output=True,
                )
                return True
            except Exception:
                return False
        return False

    def save_screenshot() -> Optional[str]:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        shots_dir = os.path.join(base_dir, "screenshots")
        os.makedirs(shots_dir, exist_ok=True)
        # Filenames use second-level timestamps, so repeated saves in the same
        # second intentionally overwrite the previous screenshot.
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(shots_dir, f"hexata-{stamp}.png")
        pygame.image.save(renderer.screen, path)
        return path

    def load_hexworld_text(text: str) -> bool:
        if not core.load_hexworld_text(text):
            return False
        win_w, win_h = pygame.display.get_window_size()
        renderer.apply_window_size(win_w, win_h)
        return True

    def load_from_clipboard() -> None:
        text = get_clipboard_text()
        if not text:
            return
        load_hexworld_text(text)

    def copy_hexworld_url() -> None:
        url = core.build_hexworld_url()
        set_clipboard_text(url)

    running = True

    def handle_keydown(ev: pygame.event.Event, ui: UiState) -> None:
        nonlocal running
        awrn_steps = [0.00, 0.01, 0.02, 0.04, 0.10, 0.20, 0.50, 1.00, 2.00]
        mods = ev.mod
        has_ctrl = bool(mods & (pygame.KMOD_META | pygame.KMOD_GUI | pygame.KMOD_CTRL))

        def step_awrn(direction: int) -> None:
            current = app.analysis_wide_root_noise
            idx = min(range(len(awrn_steps)), key=lambda i: abs(awrn_steps[i] - current))
            if direction < 0:
                idx = max(0, idx - 1)
            elif direction > 0:
                idx = min(len(awrn_steps) - 1, idx + 1)
            core.set_analysis_wide_root_noise(awrn_steps[idx])

        if ev.unicode == "?":
            ui.show_help = not ui.show_help
        elif ev.key == pygame.K_d and not (ev.mod & pygame.KMOD_CTRL):
            ui.show_engine_debug = not ui.show_engine_debug
        elif ev.key == pygame.K_m:
            ui.prefs.show_move_numbers = not ui.prefs.show_move_numbers
        elif ev.key == pygame.K_ESCAPE or (
            ev.key == pygame.K_d and (ev.mod & pygame.KMOD_CTRL)
        ):
            running = False
        elif ev.key == pygame.K_SPACE:
            core.toggle_analysis()
        elif ev.key == pygame.K_b and (mods & pygame.KMOD_SHIFT) and has_ctrl:
            if not core.is_batch_analysis_active():
                core.start_batch_analysis(fast=False)
        elif ev.key == pygame.K_b and (mods & pygame.KMOD_SHIFT) and not has_ctrl:
            if not core.is_batch_analysis_active():
                core.start_batch_analysis(fast=True)
        elif ev.key == pygame.K_e:
            ui.prefs.show_elo = not ui.prefs.show_elo
        elif ev.key == pygame.K_n and (mods & pygame.KMOD_SHIFT) and not has_ctrl:
            core.new_game()
        elif ev.key == pygame.K_v and has_ctrl:
            load_from_clipboard()
        elif ev.key == pygame.K_c and has_ctrl:
            copy_hexworld_url()
        elif ev.key == pygame.K_s and has_ctrl:
            save_screenshot()
        elif ev.key == pygame.K_c and (mods & pygame.KMOD_SHIFT):
            core.clear_analysis_caches()
        elif ev.key == pygame.K_p and (mods & pygame.KMOD_SHIFT) and not has_ctrl:
            core.try_pass_move()
        elif ev.key == pygame.K_s and not has_ctrl:
            core.try_swap_move()
        elif has_ctrl and ev.key in (pygame.K_p, pygame.K_LEFT, pygame.K_UP):
            core.step_back_n(10)
        elif has_ctrl and ev.key in (pygame.K_n, pygame.K_RIGHT, pygame.K_DOWN):
            core.step_forward_n(10)
        elif ev.key in (pygame.K_p, pygame.K_LEFT, pygame.K_UP):
            core.step_back()
        elif ev.key in (pygame.K_n, pygame.K_RIGHT, pygame.K_DOWN):
            core.step_forward()
        elif ev.key == pygame.K_f:
            core.go_first()
        elif ev.key == pygame.K_l:
            core.go_last()
        elif ev.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
            core.delete_tail()
        elif ev.key == pygame.K_x and (ev.mod & pygame.KMOD_SHIFT):
            had = bool(app.candidate_state.candidates)
            core.clear_candidates()
            if had and app.analysis_running:
                core.resume_analysis()
        elif ev.key == pygame.K_COMMA:
            pv = renderer.get_display_pv(ui.hover_cell)
            if renderer.should_show_pv(pv):
                core.try_play_moves(list(pv))
            else:
                top, _ = core.get_top_move()
                if top is not None:
                    col, row = top
                    core.try_play_move(col, row)
        elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS):
            app.pending_size = min(MAX_BOARD_SIZE, app.pending_size + 1)
        elif ev.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
            app.pending_size = max(MIN_BOARD_SIZE, app.pending_size - 1)
        elif ev.key == pygame.K_LEFTBRACKET:
            step_awrn(-1)
        elif ev.key == pygame.K_RIGHTBRACKET:
            step_awrn(1)
        elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if core.apply_pending_size():
                renderer.apply_window_size(*pygame.display.get_window_size())

    def handle_mouse_down(ev: pygame.event.Event, ui: UiState) -> None:
        if ev.button == 1:
            mx, my = renderer.window_to_surface_pos(ev.pos)
            cell = renderer.pixel_to_cell(mx, my)
            ui.swap_click_candidate = False
            if cell is None:
                return
            col, row = cell
            if board.is_empty(col, row):
                core.try_play_move(col, row)
                return
            idx = core.find_history_index(col, row)
            if idx is None:
                return
            if core.can_swap_move():
                first = board.history[0] if board.history else None
                first_coords = core.move_coords(first) if first is not None else None
                ui.swap_click_candidate = first_coords == (col, row)
            ui.drag_move = True
            ui.drag_move_from = cell
            ui.drag_move_idx = idx
        elif ev.button == 3:
            ui.drag_select = True
            ui.drag_added = False
            ui.drag_last_cell = None
            ui.drag_start_candidates = set(app.candidate_state.candidates)

    def handle_mouse_up(ev: pygame.event.Event, ui: UiState) -> None:
        if ev.button == 1:
            if ui.drag_move:
                mx, my = renderer.window_to_surface_pos(ev.pos)
                cell = renderer.pixel_to_cell(mx, my)
                if (
                    cell is not None
                    and ui.drag_move_from is not None
                    and cell != ui.drag_move_from
                    and ui.drag_move_idx is not None
                ):
                    col, row = cell
                    core.try_drag_move(ui.drag_move_idx, ui.drag_move_from, col, row)
                elif (
                    cell is not None
                    and ui.drag_move_from is not None
                    and cell == ui.drag_move_from
                    and ui.swap_click_candidate
                ):
                    core.try_swap_move()
            ui.drag_move = False
            ui.drag_move_from = None
            ui.drag_move_idx = None
            ui.swap_click_candidate = False
        elif ev.button == 3:
            mx, my = renderer.window_to_surface_pos(ev.pos)
            cell = renderer.pixel_to_cell(mx, my)
            if cell is not None and not ui.drag_added:
                col, row = cell
                start = ui.drag_start_candidates or set()
                if (col, row) in start:
                    core.remove_candidate(col, row)
                else:
                    core.add_candidate(col, row)
            ui.drag_select = False
            ui.drag_added = False
            ui.drag_last_cell = None
            ui.drag_start_candidates = None

    def handle_mouse_motion(ev: pygame.event.Event, ui: UiState) -> None:
        mx, my = renderer.window_to_surface_pos(ev.pos)
        ui.hover_cell = renderer.pixel_to_cell(mx, my)
        if not ui.drag_select or not ev.buttons[2]:
            return
        if ui.hover_cell is None or ui.hover_cell == ui.drag_last_cell:
            return
        col, row = ui.hover_cell
        start = ui.drag_start_candidates or set()
        if (col, row) in start:
            core.remove_candidate(col, row)
        else:
            core.add_candidate(col, row)
        ui.drag_added = True
        ui.drag_last_cell = ui.hover_cell

    def handle_events(ui: UiState) -> None:
        nonlocal running
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                handle_keydown(ev, ui)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                handle_mouse_down(ev, ui)
            elif ev.type == pygame.MOUSEBUTTONUP:
                handle_mouse_up(ev, ui)
            elif ev.type == pygame.MOUSEMOTION:
                handle_mouse_motion(ev, ui)
            elif ev.type == pygame.MOUSEWHEEL:
                if ev.y > 0:
                    core.step_back_n(ev.y)
                elif ev.y < 0:
                    core.step_forward_n(-ev.y)
            elif ev.type == pygame.VIDEORESIZE:
                renderer.apply_window_size(ev.w, ev.h)

    def update_frame_state(
        now: float, ui: UiState
    ) -> Tuple[bool, bool, Optional[Tuple[int, int]], int]:
        # Snapshot live analysis for this position so undo/redo can instantly display cached overlays.
        core.tick(now)
        if app.candidate_state.run is not None:
            ui.last_cand_display = app.candidate_state.run.key

        pressed = pygame.key.get_pressed()
        show_prior = bool(pressed[pygame.K_t])
        mods = pygame.key.get_mods()
        show_coords = bool(pressed[pygame.K_c]) and not (
            mods & (pygame.KMOD_CTRL | pygame.KMOD_META | pygame.KMOD_GUI | pygame.KMOD_SHIFT)
        )
        top_cell, top_visits = core.get_top_move()

        if app.analysis_running:
            total_visits = 0
            for r in core.get_engine_analysis():
                if r.visits:
                    total_visits += r.visits
            if total_visits > 0:
                if ui.speed_last_t is None:
                    ui.speed_last_t = now
                    ui.speed_last_total = total_visits
                else:
                    dt = now - ui.speed_last_t
                    if dt >= 1.0:
                        dv = total_visits - (ui.speed_last_total or 0)
                        if dv < 0:
                            dv = 0
                        ui.speed_vps = dv / dt if dt > 0 else 0.0
                        ui.speed_last_t = now
                        ui.speed_last_total = total_visits
            else:
                ui.speed_last_t = None
                ui.speed_last_total = None
                ui.speed_vps = None
        else:
            ui.speed_last_t = None
            ui.speed_last_total = None
            ui.speed_vps = None

        return show_prior, show_coords, top_cell, top_visits

    ui = UiState(prefs=ui_prefs)
    while running:
        handle_events(ui)
        now = time.monotonic()
        show_prior, show_coords, top_cell, top_visits = update_frame_state(now, ui)
        renderer.draw_frame(ui, show_prior, show_coords, top_cell, top_visits)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
