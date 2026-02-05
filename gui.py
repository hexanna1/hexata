from __future__ import annotations

import os
import subprocess
import sys
import warnings
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
)

import pygame

from board import HexBoard, Side, coord_to_human, col_to_human_letters
from engine import KataHexEngine, AnalysisMove
from gui_core import GuiCore


def run_gui(board: HexBoard, engine: KataHexEngine, *, analyze_interval_cs: int = 15) -> None:
    import math
    import pygame.freetype

    os.environ.setdefault("SDL_VIDEO_ALLOW_HIGHDPI", "1")
    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")

    RED = (220, 60, 60)
    BLUE = (40, 100, 220)
    BG = (247, 243, 233)
    OFF_WHITE = (246, 241, 232)
    GRID_EDGE = (182, 182, 182)
    BLACK = (15, 15, 15)
    HOVER_DOT = (190, 190, 190)
    GRAY = (135, 135, 135)
    PANEL_BG = (242, 238, 228)
    PANEL_EDGE = (210, 210, 210)
    WHITE = (250, 250, 250)

    ANALYSIS_LOW = (250, 236, 220)
    ANALYSIS_HIGH = (160, 210, 140)
    ANALYSIS_BEST = (175, 235, 240)
    CANDIDATE_LOW = (244, 232, 250)
    CANDIDATE_HIGH = (170, 125, 210)
    CANDIDATE_UNKNOWN = (228, 214, 245)
    CANDIDATE_ACTIVE = (250, 196, 92)

    HUD_H = 48
    BOARD_PAD = 16
    MIN_R = 4
    BASE_R = 24
    MAX_R = 38
    LABEL_PAD_K = 0.85
    PANEL_W = 230
    DEFAULT_WIN_W = 1456
    DEFAULT_WIN_H = 808

    SQ3 = math.sqrt(3)
    CORNER_DEG = [90, 30, -30, -90, -150, 150]

    core = GuiCore(board, engine, analyze_interval_cs=analyze_interval_cs)
    app = core.app

    @dataclass
    class LayoutState:
        r: int
        wstep: float
        hstep: float
        origin_x: float
        origin_y: float
        board_px_w: int
        board_px_h: int

    layout = LayoutState(
        r=BASE_R,
        wstep=SQ3 * BASE_R,
        hstep=1.5 * BASE_R,
        origin_x=0.0,
        origin_y=0.0,
        board_px_w=DEFAULT_WIN_W - PANEL_W,
        board_px_h=DEFAULT_WIN_H,
    )

    def fmt_visits(v: Optional[int]) -> str:
        if v is None:
            return ""
        if v < 1000:
            return str(v)
        if v < 100_000:
            return f"{v/1000:.1f}k"
        if v < 1_000_000:
            return f"{v//1000}k"
        if v < 10_000_000:
            return f"{v/1_000_000:.2f}m"
        if v < 1_000_000_000:
            return f"{v/1_000_000:.1f}m"
        return f"{v/1_000_000:.0f}m"

    def fmt_wr(w: Optional[float]) -> str:
        if w is None:
            return ""
        return f"{w*100:.1f}"

    def fmt_wr_or_elo(w: Optional[float], show_elo: bool) -> str:
        if not show_elo:
            return fmt_wr(w)
        if w is None:
            return ""
        p = round(w, 3)
        if p <= 0.0:
            return "-inf"
        if p >= 1.0:
            return "+inf"
        elo = int(round(400.0 * math.log10(p / (1.0 - p))))
        if elo == 0:
            return "±0"
        return f"{elo:+d}"

    def fmt_prior(p: Optional[float]) -> str:
        if p is None:
            return ""
        return f"{p*100:.1f}"

    def clamp01(x: float) -> float:
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

    def lerp_rgb(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
        t = clamp01(t)
        return (
            int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t),
        )

    pygame.init()
    pygame.freetype.init()
    pygame.key.set_repeat(400, 33)

    FLAGS = pygame.RESIZABLE

    screen = pygame.display.set_mode((DEFAULT_WIN_W, DEFAULT_WIN_H), FLAGS)
    clock = pygame.time.Clock()

    scrap_ok = False
    try:
        pygame.scrap.init()
        pygame.scrap.set_mode(pygame.SCRAP_CLIPBOARD)
        scrap_ok = True
    except Exception:
        scrap_ok = False

    def make_board_small(r: int) -> pygame.freetype.Font:
        t = clamp01((BASE_R - r) / (BASE_R - 8))
        fill = 0.65 + 0.10 * t
        small_sz = max(4, min(16, int(r * fill)))
        return pygame.freetype.Font(None, small_sz)

    hud_font = pygame.freetype.Font(None, 20)
    hud_small = pygame.freetype.Font(None, 14)
    movelist_font = pygame.freetype.SysFont(["Menlo", "Consolas", "monospace"], 14)
    io_font = pygame.freetype.SysFont(["Menlo", "Consolas", "monospace"], 12)

    @dataclass
    class BaselineText:
        font: pygame.freetype.Font
        baseline_offset: float
        line_ref_y: float

        @classmethod
        def for_font(cls, font: pygame.freetype.Font) -> "BaselineText":
            # Align glyphs to a consistent baseline using a single reference string.
            _surf, rect = font.render("Ag01", fgcolor=BLACK)
            return cls(
                font=font,
                baseline_offset=rect.y - rect.height / 2,
                line_ref_y=rect.y,
            )

        def blit_center(self, text: str, color, cx: float, cy: float) -> None:
            surf, rect = self.font.render(text, fgcolor=color)
            x = cx - surf.get_width() / 2
            y = cy + self.baseline_offset - rect.y
            screen.blit(surf, (x, y))

        def blit_line(self, text: str, color, x: float, y: float) -> float:
            surf, rect = self.font.render(text, fgcolor=color)
            screen.blit(surf, (x, y + self.line_ref_y - rect.y))
            return surf.get_width()

    @dataclass
    class FontState:
        board_small: pygame.freetype.Font
        hud_font: pygame.freetype.Font
        hud_small: pygame.freetype.Font
        movelist_font: pygame.freetype.Font
        io_font: pygame.freetype.Font

    @dataclass
    class TextRenderer:
        board: BaselineText
        movelist: BaselineText
        hud: BaselineText
        hud_small: BaselineText
        io: BaselineText
        line_hud_small: int
        line_io: int

        @classmethod
        def from_fonts(cls, fonts: FontState) -> "TextRenderer":
            return cls(
                board=BaselineText.for_font(fonts.board_small),
                movelist=BaselineText.for_font(fonts.movelist_font),
                hud=BaselineText.for_font(fonts.hud_font),
                hud_small=BaselineText.for_font(fonts.hud_small),
                io=BaselineText.for_font(fonts.io_font),
                line_hud_small=fonts.hud_small.get_sized_height(),
                line_io=fonts.io_font.get_sized_height(),
            )

        def update_board_small(self, board_small: pygame.freetype.Font) -> None:
            self.board = BaselineText.for_font(board_small)

    fonts = FontState(
        board_small=make_board_small(layout.r),
        hud_font=hud_font,
        hud_small=hud_small,
        movelist_font=movelist_font,
        io_font=io_font,
    )
    text = TextRenderer.from_fonts(fonts)

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
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(shots_dir, f"hexata-{stamp}.png")
        pygame.image.save(screen, path)
        return path

    def _get_clipboard_fallback() -> Optional[str]:
        if sys.platform != "darwin":
            return None
        try:
            out = subprocess.run(["pbpaste"], check=False, capture_output=True, text=True)
        except Exception:
            return None
        text = out.stdout.strip()
        return text if text else None

    def min_window_size(n: int) -> Tuple[int, int]:
        span = 1.5 * (n - 1)
        board_w = SQ3 * MIN_R * (span + 1)
        board_h = MIN_R * (span + 2)
        label_pad = LABEL_PAD_K * MIN_R
        win_w = int(board_w + label_pad + PANEL_W + 2 * BOARD_PAD)
        win_h = int(board_h + label_pad + HUD_H + 2 * BOARD_PAD)
        return win_w, win_h

    def apply_window_size(win_w: int, win_h: int) -> Tuple[int, int]:
        nonlocal screen
        min_w, min_h = min_window_size(board.n)
        win_w = max(win_w, min_w)
        win_h = max(win_h, min_h)
        if (win_w, win_h) != pygame.display.get_window_size():
            screen = pygame.display.set_mode((win_w, win_h), FLAGS)
        layout.board_px_w = max(0, win_w - PANEL_W)
        layout.board_px_h = win_h

        usable_w0 = max(0.0, layout.board_px_w - 2 * BOARD_PAD)
        usable_h0 = max(0.0, layout.board_px_h - HUD_H - 2 * BOARD_PAD)

        span = 1.5 * (board.n - 1)
        denom_w = SQ3 * (span + 1)
        denom_h = span + 2
        pad_k = LABEL_PAD_K
        r_w = usable_w0 / (denom_w + pad_k) if denom_w > 0 else MIN_R
        r_h = usable_h0 / (denom_h + pad_k) if denom_h > 0 else MIN_R
        layout.r = int(max(MIN_R, min(r_w, r_h, MAX_R)))

        layout.wstep = SQ3 * layout.r
        layout.hstep = 1.5 * layout.r
        label_pad = pad_k * layout.r
        usable_w = max(0.0, usable_w0 - label_pad)
        usable_h = max(0.0, usable_h0 - label_pad)

        board_w = SQ3 * layout.r * (span + 1)
        board_h = layout.r * (span + 2)
        extra_w = max(0.0, usable_w - board_w)
        extra_h = max(0.0, usable_h - board_h)
        layout.origin_x = BOARD_PAD + label_pad + extra_w / 2 + (SQ3 * layout.r) / 2
        layout.origin_y = HUD_H + BOARD_PAD + label_pad + extra_h / 2 + layout.r

        fonts.board_small = make_board_small(layout.r)
        text.update_board_small(fonts.board_small)
        pygame.display.set_caption(f"Hex {board.n}x{board.n}")
        return win_w, win_h

    apply_window_size(DEFAULT_WIN_W, DEFAULT_WIN_H)

    def window_to_surface_pos(pos: Tuple[int, int]) -> Tuple[int, int]:
        wx, wy = pygame.display.get_window_size()
        sx, sy = pygame.display.get_surface().get_size()
        if wx <= 0 or wy <= 0:
            return pos
        if (wx, wy) == (sx, sy):
            return pos
        scale_x = sx / wx
        scale_y = sy / wy
        return (int(pos[0] * scale_x), int(pos[1] * scale_y))

    def center(ax: int, ay: int) -> Tuple[float, float]:
        cx = layout.origin_x + layout.wstep * (ax + ay / 2)
        cy = layout.origin_y + layout.hstep * ay
        return cx, cy

    def corner(ax: int, ay: int, i: int) -> Tuple[int, int]:
        cx, cy = center(ax, ay)
        a = math.radians(CORNER_DEG[i % 6])
        return round(cx + layout.r * math.cos(a)), round(cy + layout.r * math.sin(a))

    def poly(ax: int, ay: int) -> List[Tuple[int, int]]:
        return [corner(ax, ay, i) for i in range(6)]

    def pixel_to_cell(mx: int, my: int) -> Optional[Tuple[int, int]]:
        if my < HUD_H:
            return None
        if mx >= layout.board_px_w:
            return None
        best: Optional[Tuple[int, int]] = None
        best_d2 = 10**18
        for row in range(1, board.n + 1):
            for col in range(1, board.n + 1):
                ax, ay = col - 1, row - 1
                cx, cy = center(ax, ay)
                dx = mx - cx
                dy = my - cy
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best = (col, row)
        if best is None:
            return None
        if best_d2 <= (layout.r * 0.98) ** 2:
            return best
        return None

    def load_hexworld_text(text: str) -> bool:
        if not core.load_hexworld_text(text):
            return False
        win_w, win_h = pygame.display.get_window_size()
        apply_window_size(win_w, win_h)
        return True

    def load_from_clipboard() -> None:
        text = get_clipboard_text()
        if not text:
            return
        load_hexworld_text(text)

    def copy_hexworld_url() -> None:
        url = core.build_hexworld_url()
        set_clipboard_text(url)

    def draw_grid_and_stones(
        top_cell: Optional[Tuple[int, int]],
        top_visits: int,
        show_prior: bool,
        *,
        skip_cell: Optional[Tuple[int, int]] = None,
    ) -> None:
        visits_map: dict[Tuple[int, int], int] = {}
        winrate_map: dict[Tuple[int, int], float] = {}
        prior_map: dict[Tuple[int, int], float] = {}
        candidate_wr_map: dict[Tuple[int, int], Optional[float]] = {}
        for r in core.get_active_analysis():
            if r.col is None or r.row is None:
                continue
            if not board.is_empty(r.col, r.row):
                continue
            if r.visits is not None:
                visits_map[(r.col, r.row)] = r.visits
            if r.winrate is not None:
                winrate_map[(r.col, r.row)] = r.winrate
            if r.prior is not None:
                prior_map[(r.col, r.row)] = r.prior
        if app.candidates:
            for key in app.candidates:
                candidate_wr_map[key] = app.candidate_results.get(key, (None, None))[0]
        cand_count = len(app.candidates) if app.candidates else 0

        denom = math.log(max(2, top_visits))
        max_prior = max(prior_map.values()) if prior_map else None

        for row in range(1, board.n + 1):
            for col in range(1, board.n + 1):
                ax, ay = col - 1, row - 1
                occ = board.get(col, row)
                if skip_cell is not None and skip_cell == (col, row):
                    occ = -1

                if occ < 0:
                    if (col, row) in app.candidates:
                        cand_wr = candidate_wr_map.get((col, row))
                        if cand_wr is None:
                            fill = CANDIDATE_UNKNOWN
                        else:
                            t = clamp01(cand_wr) ** 0.9
                            fill = lerp_rgb(CANDIDATE_LOW, CANDIDATE_HIGH, t)
                        if (
                            cand_count > 1
                            and app.candidate_run is not None
                            and app.candidate_run.key == (col, row)
                        ):
                            fill = CANDIDATE_ACTIVE
                    elif show_prior:
                        pr = prior_map.get((col, row))
                        if pr is None:
                            fill = OFF_WHITE
                        else:
                            t = clamp01(pr) ** 0.9
                            fill = lerp_rgb(ANALYSIS_LOW, ANALYSIS_HIGH, t)
                            if max_prior is not None and pr >= max_prior - 0.001:
                                fill = ANALYSIS_BEST
                    else:
                        v = visits_map.get((col, row))
                        wr = winrate_map.get((col, row))
                        if v is None and wr is None:
                            fill = OFF_WHITE
                        else:
                            if wr is None:
                                wr = 0.5
                            t = 0.0 if denom <= 0 else (math.log(max(1, v or 1)) / denom)
                            t = clamp01(t) ** 1.1
                            wr_bias = (wr - 0.5) * 2.0
                            t = clamp01(t + 0.35 * wr_bias)
                            fill = lerp_rgb(ANALYSIS_LOW, ANALYSIS_HIGH, t)

                            if top_cell is not None and (col, row) == top_cell:
                                fill = ANALYSIS_BEST
                else:
                    fill = RED if occ == int(Side.RED) else BLUE

                pts = poly(ax, ay)
                pygame.draw.polygon(screen, fill, pts, 0)
                pygame.draw.polygon(screen, GRID_EDGE, pts, 1)

    def draw_next_future_outline() -> None:
        if not app.future_moves:
            return
        mv = app.future_moves[-1]
        coords = core.move_coords(mv)
        if coords is None:
            return
        col, row = coords
        if not board.is_empty(col, row):
            return

        ax, ay = col - 1, row - 1
        pts = poly(ax, ay)
        colr = RED if core.current_side() == Side.RED else BLUE
        thickness = max(3, int(layout.r * 0.12))
        pygame.draw.polygon(screen, colr, pts, thickness)

    def draw_move_numbers(show_all: bool) -> None:
        if not board.history:
            return
        last_idx = len(board.history) - 1
        if show_all:
            number_tint = 0.45
            for idx in range(len(board.history)):
                mv = board.history[idx]
                coords = core.move_coords(mv)
                if coords is None:
                    continue
                ax, ay = coords[0] - 1, coords[1] - 1
                cx, cy = center(ax, ay)
                txt = str(idx + 1)
                if idx == last_idx:
                    colr = OFF_WHITE
                else:
                    base = RED if mv.side == Side.RED else BLUE
                    colr = lerp_rgb(base, OFF_WHITE, number_tint)
                text.board.blit_center(txt, colr, cx, cy)
            return

        mv = board.history[-1]
        coords = core.move_coords(mv)
        if coords is None:
            return
        ax, ay = coords[0] - 1, coords[1] - 1
        cx, cy = center(ax, ay)
        dot_r = max(2, int(layout.r * 0.18))
        pygame.draw.circle(screen, OFF_WHITE, (int(cx), int(cy)), dot_r, 0)

    def draw_borders() -> None:
        thickness = 4
        sides = [
            {"color": RED, "segs": [(2, 3), (3, 4)], "coord": lambda i: (i, 0)},
            {"color": RED, "segs": [(5, 0), (0, 1)], "coord": lambda i: (i, board.n - 1)},
            {"color": BLUE, "segs": [(4, 5), (5, 0)], "coord": lambda i: (0, i)},
            {"color": BLUE, "segs": [(1, 2), (2, 3)], "coord": lambda i: (board.n - 1, i)},
        ]
        for side in sides:
            for i in range(board.n):
                ax, ay = side["coord"](i)
                for c1, c2 in side["segs"]:
                    pygame.draw.line(
                        screen, side["color"], corner(ax, ay, c1), corner(ax, ay, c2), thickness
                    )

    def draw_side_coords() -> None:
        col_color = GRAY
        row_color = GRAY
        for col in range(1, board.n + 1):
            ax, ay = col - 1, -1
            cx, cy = center(ax, ay)
            txt = col_to_human_letters(col)
            text.board.blit_center(txt, col_color, cx, cy)
        for row in range(1, board.n + 1):
            ax, ay = -1, row - 1
            cx, cy = center(ax, ay)
            txt = str(row)
            text.board.blit_center(txt, row_color, cx, cy)

    def draw_ghost_cell(cell: Tuple[int, int], side: Side) -> None:
        ax, ay = cell[0] - 1, cell[1] - 1
        pts = poly(ax, ay)
        base = RED if side == Side.RED else BLUE
        ghost = lerp_rgb(base, OFF_WHITE, 0.5)
        pygame.draw.polygon(screen, ghost, pts, 0)
        pygame.draw.polygon(screen, GRID_EDGE, pts, 1)

    def get_hover_pv(
        hover_cell: Optional[Tuple[int, int]],
    ) -> Optional[Tuple[Tuple[int, int], ...]]:
        if hover_cell is None:
            return None
        if app.candidates:
            return None
        col, row = hover_cell
        if not board.is_empty(col, row):
            return None
        for r in core.get_active_analysis():
            if r.col == col and r.row == row and r.pv:
                return r.pv
        return None

    def should_show_pv(pv: Optional[Tuple[Tuple[int, int], ...]]) -> bool:
        return bool(pv) and len(pv) > 1

    def draw_pv_ghosts(pv: Tuple[Tuple[int, int], ...], start_side: Side) -> None:
        side = start_side
        for cell in pv:
            if board.is_empty(*cell):
                draw_ghost_cell(cell, side)
            side = core.flip_side(side)

    def draw_pv_numbers(pv: Tuple[Tuple[int, int], ...], start_side: Side) -> None:
        side = start_side
        for idx, cell in enumerate(pv):
            if idx == 0:
                side = core.flip_side(side)
                continue
            if not board.is_empty(*cell):
                side = core.flip_side(side)
                continue
            ax, ay = cell[0] - 1, cell[1] - 1
            cx, cy = center(ax, ay)
            text.board.blit_center(str(idx + 1), OFF_WHITE, cx, cy)
            side = core.flip_side(side)

    def draw_analysis_text(
        show_prior: bool,
        show_coords: bool,
        *,
        suppress_cells: Optional[set[Tuple[int, int]]] = None,
    ) -> None:
        if show_coords:
            for row in range(1, board.n + 1):
                for col in range(1, board.n + 1):
                    ax, ay = col - 1, row - 1
                    cx, cy = center(ax, ay)

                    occ = board.get(col, row)
                    colr = WHITE if occ >= 0 else BLACK

                    txt = coord_to_human(col, row)
                    text.board.blit_center(txt, colr, cx, cy)
            return

        for r in core.get_active_analysis():
            if r.col is None or r.row is None:
                continue
            col, row = r.col, r.row
            if suppress_cells and (col, row) in suppress_cells:
                continue
            if not board.is_empty(col, row):
                continue
            ax, ay = col - 1, row - 1
            cx, cy = center(ax, ay)

            if show_prior:
                pr = fmt_prior(r.prior)
                if not pr:
                    continue
                text.board.blit_center(pr, BLACK, cx, cy)
                continue

            wr = fmt_wr_or_elo(r.winrate, ui.show_elo)
            vv = fmt_visits(r.visits)
            if not wr and not vv:
                continue

            gap = 1
            rect1 = text.board.font.get_rect(wr)
            rect2 = text.board.font.get_rect(vv)
            total_h = rect1.height + gap + rect2.height
            top = cy - total_h / 2
            x1 = cx - rect1.width / 2
            y1 = top - text.board.line_ref_y + rect1.y
            text.board.blit_line(wr, BLACK, x1, y1)
            top2 = top + rect1.height + gap
            x2 = cx - rect2.width / 2
            y2 = top2 - text.board.line_ref_y + rect2.y
            text.board.blit_line(vv, BLACK, x2, y2)

    def blit_segments(x: int, y: int, parts: List[Tuple[str, Tuple[int, int, int]]], use_small: bool) -> None:
        blit_line = text.hud_small.blit_line if use_small else text.hud.blit_line
        cx = x
        for txt, col in parts:
            cx += blit_line(txt, col, cx, y)

    GRAPH_MIN_MOVES = 50
    GRAPH_MIN_HEIGHT = 40
    GRAPH_LINE_WIDTH = 2
    GRAPH_DOT_RADIUS = 3
    GRAPH_LABEL_PAD = 6
    GRAPH_EDGE_PAD = 2
    GRAPH_PAD = 0
    GRAPH_ELO_CLAMP = 1000.0

    def draw_eval_graph(
        rect: pygame.Rect, cursor_ply: int, total_moves: int, show_elo: bool
    ) -> None:

        pygame.draw.rect(screen, PANEL_BG, rect)
        pygame.draw.rect(screen, PANEL_EDGE, rect, 1)

        n_moves = total_moves
        if rect.width <= 1 or rect.height <= 1:
            return

        def best_reply_winrate(ply_len: int, side_to_play: Side) -> Optional[float]:
            recs = app.analysis_cache.get((ply_len, int(side_to_play)))
            if not recs:
                return None
            # Only use ordered (live) analysis; candidate cache entries have no order.
            best = None
            for r in recs:
                if r.order is None or r.winrate is None:
                    continue
                if best is None or r.order < best.order:
                    best = r
            if best is None or best.winrate is None:
                return None
            return best.winrate

        def x_for_move(m: int) -> int:
            return int(rect.left + (m - 1) * rect.width / denom)

        def winrate_to_elo(v: float) -> float:
            if v <= 0.0:
                return -GRAPH_ELO_CLAMP
            if v >= 1.0:
                return GRAPH_ELO_CLAMP
            elo = 400.0 * math.log10(v / (1.0 - v))
            if elo > GRAPH_ELO_CLAMP:
                return GRAPH_ELO_CLAMP
            if elo < -GRAPH_ELO_CLAMP:
                return -GRAPH_ELO_CLAMP
            return elo

        def y_for_plot(v: float) -> int:
            if show_elo:
                v = winrate_to_elo(v)
                t = (v + GRAPH_ELO_CLAMP) / (2.0 * GRAPH_ELO_CLAMP)
            else:
                t = 0.0 if v < 0.0 else 1.0 if v > 1.0 else v
            return int(rect.bottom - t * rect.height)

        mid_y = y_for_plot(0.5)
        pygame.draw.line(screen, GRID_EDGE, (rect.left, mid_y), (rect.right, mid_y), 1)

        if n_moves <= 0:
            return

        values: dict[int, float] = {}
        max_analyzed = 0
        for m in range(1, n_moves + 1):
            side_to_play = Side.BLUE if m % 2 == 1 else Side.RED
            best_wr = best_reply_winrate(m, side_to_play)
            if best_wr is None:
                continue
            # Red-perspective winrate for the current position.
            red_wr = best_wr if side_to_play == Side.RED else 1.0 - best_wr
            values[m] = red_wr
            if m > max_analyzed:
                max_analyzed = m

        if max_analyzed <= 0:
            return

        max_moves = max(GRAPH_MIN_MOVES, max_analyzed)
        denom = max(1, max_moves - 1)

        def draw_segment(points: List[Tuple[int, int]]) -> None:
            if len(points) >= 2:
                pygame.draw.lines(screen, RED, False, points, GRAPH_LINE_WIDTH)
                pygame.draw.aalines(screen, RED, False, points)
            elif len(points) == 1:
                pygame.draw.circle(screen, RED, points[0], 2, 0)

        points: List[Tuple[int, int]] = []
        for m in range(1, n_moves + 1):
            val = values.get(m)
            if val is None:
                draw_segment(points)
                points = []
                continue
            points.append((x_for_move(m), y_for_plot(val)))
        draw_segment(points)

        if 1 <= cursor_ply <= n_moves:
            m = cursor_ply
            val = values.get(m)
            if val is not None:
                cx = x_for_move(m)
                cy = y_for_plot(val)
                pygame.draw.circle(screen, RED, (cx, cy), GRAPH_DOT_RADIUS, 0)

                label = fmt_wr_or_elo(val, show_elo)
                label_rect = text.hud_small.font.get_rect(label)
                line_h = text.hud_small.font.get_sized_height()
                lx = cx + GRAPH_LABEL_PAD
                ly = cy - line_h / 2
                if lx + label_rect.width > rect.right - GRAPH_EDGE_PAD:
                    lx = cx - label_rect.width - GRAPH_LABEL_PAD
                if ly < rect.top + GRAPH_EDGE_PAD:
                    ly = rect.top + GRAPH_EDGE_PAD
                if ly + line_h > rect.bottom - GRAPH_EDGE_PAD:
                    ly = rect.bottom - line_h - GRAPH_EDGE_PAD
                text.hud_small.blit_line(label, RED, lx, ly)

    def draw_movelist_panel() -> None:
        x0 = layout.board_px_w
        pygame.draw.rect(screen, PANEL_BG, pygame.Rect(x0, 0, PANEL_W, screen.get_height()))

        pad = 12
        y = 10
        blit_segments(x0 + pad, y, [("Moves", BLACK)], use_small=False)
        y += 26

        moves = list(board.history)
        if app.future_moves:
            moves.extend(reversed(app.future_moves))

        total_moves = len(moves)
        cursor_ply = len(board.history)
        nrows = (total_moves + 1) // 2

        line_h = movelist_font.get_sized_height() + 4
        io_line_h = text.line_io + 2
        io_max_lines = 30
        io_panel_h = (io_line_h * io_max_lines) + 10
        graph_h = 0
        if not ui.show_engine_debug:
            avail = screen.get_height() - y
            graph_h = max(0, min(PANEL_W, avail))
        graph_top = screen.get_height() - graph_h - GRAPH_PAD
        io_top = max(y, screen.get_height() - io_panel_h) if ui.show_engine_debug else graph_top
        max_lines = max(0, (io_top - y - 6) // line_h)

        start = 0
        if max_lines and nrows > max_lines:
            if cursor_ply == total_moves:
                start = nrows - max_lines
            else:
                focus_row = 0 if cursor_ply == 0 else (cursor_ply - 1) // 2
                start = focus_row - (max_lines // 2)
                start = max(0, min(start, nrows - max_lines))

        end = nrows if not max_lines else min(nrows, start + max_lines)

        for row_idx in range(start, end):
            red_i = 2 * row_idx
            blue_i = red_i + 1

            red_mv = core.move_to_label(moves[red_i])
            blue_mv = (
                core.move_to_label(moves[blue_i]) if blue_i < total_moves else None
            )

            odd = 2 * row_idx + 1
            label = f"{odd}."

            red_col = RED if red_i < cursor_ply else GRAY
            parts: List[Tuple[str, Tuple[int, int, int]]] = [
                (f"{label:>3} ", BLACK),
                (red_mv + " ", red_col),
            ]

            if blue_mv:
                blue_col = BLUE if blue_i < cursor_ply else GRAY
                parts.append((blue_mv, blue_col))

            cx = x0 + pad
            for txt, col in parts:
                cx += text.movelist.blit_line(txt, col, cx, y)
            y += line_h

        if ui.show_engine_debug:
            io_rect = pygame.Rect(x0, io_top, PANEL_W, screen.get_height() - io_top)
            pygame.draw.rect(screen, PANEL_BG, io_rect)
            pygame.draw.rect(screen, PANEL_EDGE, io_rect, 1)

            io_y = io_top + 4
            logs = core.engine.get_io_log(io_max_lines)
            start_idx = max(0, len(logs) - io_max_lines)
            for direction, msg, count in logs[start_idx:]:
                prefix = ">>" if direction == "out" else "<<"
                if count > 1:
                    line = f"{prefix} {msg} ({count})"
                else:
                    line = f"{prefix} {msg}"
                text.io.blit_line(line, BLACK, x0 + pad, io_y)
                io_y += io_line_h
        elif graph_h >= GRAPH_MIN_HEIGHT:
            graph_rect = pygame.Rect(
                x0,
                graph_top,
                PANEL_W,
                screen.get_height() - graph_top,
            )
            draw_eval_graph(graph_rect, cursor_ply, total_moves, ui.show_elo)

        pygame.draw.line(screen, PANEL_EDGE, (x0, 0), (x0, screen.get_height()), 1)
    def draw_hud() -> None:
        pygame.draw.rect(screen, BG, pygame.Rect(0, 0, screen.get_width(), HUD_H))

        turn_side = core.current_side()
        turn_color = RED if turn_side == Side.RED else BLUE
        turn_name = "Red" if turn_side == Side.RED else "Blue"
        analysis_txt = "ON" if app.analysis_running else "OFF"
        analysis_color = BLACK if app.analysis_running else GRAY

        parts: List[Tuple[str, Tuple[int, int, int]]] = [
            ("Size: ", BLACK),
            (f"{board.n}", BLACK),
        ]
        if app.pending_size != board.n:
            parts += [
                ("  (pending ", BLACK),
                (f"{app.pending_size}", BLACK),
                (")", BLACK),
            ]
        parts += [
            ("   |   Turn: ", BLACK),
            (turn_name, turn_color),
            ("   |   ", BLACK),
            ("Analysis: ", analysis_color),
            (analysis_txt, analysis_color),
        ]
        if app.candidates:
            cand_key = (
                app.candidate_run.key if app.candidate_run is not None else ui.last_cand_display
            )
            if cand_key is None:
                next_keys = core.sorted_candidates_by_visits()
                cand_key = next_keys[0] if next_keys else None
            parts += [("   |   ", BLACK), ("Cand: ", BLACK)]
            if cand_key is not None:
                parts += [(coord_to_human(*cand_key), turn_color)]
        else:
            display: Optional[AnalysisMove] = None
            for r in core.get_active_analysis():
                if r.col is None or r.row is None:
                    continue
                if not board.is_empty(r.col, r.row):
                    continue
                display = r
                break
            if display is not None:
                parts += [("   |   ", BLACK), ("Best: ", BLACK)]
                coord = coord_to_human(display.col, display.row)
                parts += [(coord, turn_color)]
                wr = fmt_wr_or_elo(display.winrate, ui.show_elo)
                if wr:
                    if ui.show_elo:
                        parts += [(" ", BLACK), (wr, BLACK)]
                    else:
                        parts += [(" ", BLACK), (f"{wr}%", BLACK)]
                vv = fmt_visits(display.visits)
                if vv:
                    parts += [(" ", BLACK), (f"({vv})", BLACK)]
        blit_segments(12, 10, parts, use_small=False)

        help_line = "space:analysis • ,:play best • +/-/enter:size • ?:help"
        text.hud_small.blit_line(help_line, BLACK, 12, 32)

        awrn = f"{app.analysis_wide_root_noise:.2f}".rstrip("0").rstrip(".")
        awrn_text = "AWRN –" if app.candidates else f"AWRN {awrn}"
        awrn_w = fonts.hud_small.get_rect(awrn_text).width
        awrn_x = max(12, layout.board_px_w - awrn_w - 12)
        text.hud_small.blit_line(awrn_text, GRAY, awrn_x, 10)
        vps_suffix = " visits/s"
        if ui.speed_vps is not None and ui.speed_vps > 0:
            vps_text = f"{fmt_visits(int(ui.speed_vps))}{vps_suffix}"
        else:
            vps_text = f"–{vps_suffix}"
        vps_w = fonts.hud_small.get_rect(vps_text).width
        vps_x = max(12, layout.board_px_w - vps_w - 12)
        vps_y = 10 + text.line_hud_small + 2
        text.hud_small.blit_line(vps_text, GRAY, vps_x, vps_y)

    def draw_help_overlay() -> None:
        lines = [
            "Help (? to hide)",
            "space:analysis   ,:play best/PV   esc:quit",
            "p:prev   n:next   f:first   l:last   shift+p:pass",
            "t:priors   c:coords   m:moves   e:elo",
            "ctrl+v:load   ctrl+c:copy   shift+c:clear cache",
            "del:delete tail   shift+n:new",
            "+/-:pending size   enter:apply size",
            "[/]:set analysisWideRootNoise",
            "d:engine debug   ctrl+s:screenshot",
            "left-drag:move stone",
            "right-click:toggle cand   right-drag:toggle cands",
            "shift+x:clear cands",
        ]
        pad = 8
        gap = 2
        surfs = [fonts.hud_small.render(line, fgcolor=BLACK)[0] for line in lines]
        w = max(s.get_width() for s in surfs)
        line_h = text.line_hud_small
        h = line_h * len(lines) + gap * (len(surfs) - 1)
        x = 12
        y = HUD_H + 12
        rect = pygame.Rect(x - pad, y - pad, w + pad * 2, h + pad * 2)
        pygame.draw.rect(screen, PANEL_BG, rect)
        pygame.draw.rect(screen, PANEL_EDGE, rect, 1)
        for line in lines:
            text.hud_small.blit_line(line, BLACK, x, y)
            y += line_h + gap

    @dataclass
    class UiState:
        drag_select: bool = False
        drag_added: bool = False
        drag_last_cell: Optional[Tuple[int, int]] = None
        drag_start_candidates: Optional[set[Tuple[int, int]]] = None
        drag_move: bool = False
        drag_move_from: Optional[Tuple[int, int]] = None
        drag_move_idx: Optional[int] = None
        hover_cell: Optional[Tuple[int, int]] = None
        show_help: bool = False
        show_move_numbers: bool = False
        show_elo: bool = False
        show_engine_debug: bool = False
        last_cand_display: Optional[Tuple[int, int]] = None
        speed_last_t: Optional[float] = None
        speed_last_total: Optional[int] = None
        speed_vps: Optional[float] = None

    def handle_keydown(ev: pygame.event.Event, ui: UiState) -> None:
        nonlocal running
        awrn_steps = [0.00, 0.01, 0.02, 0.04, 0.10, 0.20, 0.50, 1.00, 2.00]

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
            ui.show_move_numbers = not ui.show_move_numbers
        elif ev.key == pygame.K_ESCAPE or (
            ev.key == pygame.K_d and (ev.mod & pygame.KMOD_CTRL)
        ):
            running = False
        elif ev.key == pygame.K_SPACE:
            core.toggle_analysis()
        elif ev.key == pygame.K_e:
            ui.show_elo = not ui.show_elo
        elif ev.key == pygame.K_n and (ev.mod & pygame.KMOD_SHIFT):
            core.new_game()
        elif ev.key == pygame.K_v and (
            ev.mod & (pygame.KMOD_META | pygame.KMOD_GUI | pygame.KMOD_CTRL)
        ):
            load_from_clipboard()
        elif ev.key == pygame.K_c and (
            ev.mod & (pygame.KMOD_META | pygame.KMOD_GUI | pygame.KMOD_CTRL)
        ):
            copy_hexworld_url()
        elif ev.key == pygame.K_s and (
            ev.mod & (pygame.KMOD_META | pygame.KMOD_GUI | pygame.KMOD_CTRL)
        ):
            save_screenshot()
        elif ev.key == pygame.K_c and (ev.mod & pygame.KMOD_SHIFT):
            core.clear_analysis_caches()
        elif ev.key == pygame.K_p and (ev.mod & pygame.KMOD_SHIFT):
            core.try_pass_move()
        elif ev.key in (pygame.K_UP, pygame.K_p, pygame.K_LEFT):
            core.step_back()
        elif ev.key in (pygame.K_DOWN, pygame.K_n, pygame.K_RIGHT):
            core.step_forward()
        elif ev.key == pygame.K_f:
            core.go_first()
        elif ev.key == pygame.K_l:
            core.go_last()
        elif ev.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
            core.delete_tail()
        elif ev.key == pygame.K_x and (ev.mod & pygame.KMOD_SHIFT):
            had = bool(app.candidates)
            core.clear_candidates()
            if had and app.analysis_running:
                core.resume_analysis()
        elif ev.key == pygame.K_COMMA:
            pv = get_hover_pv(ui.hover_cell)
            if should_show_pv(pv):
                core.try_play_moves(list(pv))
            else:
                top, _top_visits = core.get_top_move()
                if top is not None:
                    col, row = top
                    core.try_play_move(col, row)
        elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS):
            app.pending_size = min(42, app.pending_size + 1)
        elif ev.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
            app.pending_size = max(4, app.pending_size - 1)
        elif ev.key == pygame.K_LEFTBRACKET:
            step_awrn(-1)
        elif ev.key == pygame.K_RIGHTBRACKET:
            step_awrn(1)
        elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if core.apply_pending_size():
                apply_window_size(*pygame.display.get_window_size())

    def handle_mouse_down(ev: pygame.event.Event, ui: UiState) -> None:
        if ev.button == 1:
            mx, my = window_to_surface_pos(ev.pos)
            cell = pixel_to_cell(mx, my)
            if cell is None:
                return
            col, row = cell
            if board.is_empty(col, row):
                core.try_play_move(col, row)
                return
            idx = core.find_history_index(col, row)
            if idx is None:
                return
            ui.drag_move = True
            ui.drag_move_from = cell
            ui.drag_move_idx = idx
        elif ev.button == 3:
            ui.drag_select = True
            ui.drag_added = False
            ui.drag_last_cell = None
            ui.drag_start_candidates = set(app.candidates)

    def handle_mouse_up(ev: pygame.event.Event, ui: UiState) -> None:
        if ev.button == 1:
            if ui.drag_move:
                mx, my = window_to_surface_pos(ev.pos)
                cell = pixel_to_cell(mx, my)
                if (
                    cell is not None
                    and ui.drag_move_from is not None
                    and cell != ui.drag_move_from
                    and ui.drag_move_idx is not None
                ):
                    col, row = cell
                    core.try_drag_move(ui.drag_move_idx, ui.drag_move_from, col, row)
            ui.drag_move = False
            ui.drag_move_from = None
            ui.drag_move_idx = None
        elif ev.button == 3:
            mx, my = window_to_surface_pos(ev.pos)
            cell = pixel_to_cell(mx, my)
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
        mx, my = window_to_surface_pos(ev.pos)
        ui.hover_cell = pixel_to_cell(mx, my)
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
            elif ev.type == pygame.VIDEORESIZE:
                apply_window_size(ev.w, ev.h)

    def update_frame_state(
        now: float, ui: UiState
    ) -> Tuple[bool, bool, Optional[Tuple[int, int]], int]:
        # Snapshot live analysis for this position so undo/redo can instantly display cached overlays.
        core.tick(now)
        if app.candidate_run is not None:
            ui.last_cand_display = app.candidate_run.key

        pressed = pygame.key.get_pressed()
        show_prior = bool(pressed[pygame.K_t])
        mods = pygame.key.get_mods()
        show_coords = bool(pressed[pygame.K_c]) and not (
            mods & (pygame.KMOD_CTRL | pygame.KMOD_META | pygame.KMOD_GUI | pygame.KMOD_SHIFT)
        )
        top_cell, top_visits = core.get_top_move()

        if app.analysis_running:
            total_visits = 0
            for r in core.engine.get_analysis():
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

        return show_prior, show_coords, top_cell, top_visits

    def draw_frame(
        show_prior: bool,
        show_coords: bool,
        top_cell: Optional[Tuple[int, int]],
        top_visits: int,
        ui: UiState,
    ) -> None:
        screen.fill(BG)
        draw_hud()
        pv = get_hover_pv(ui.hover_cell)
        show_pv = should_show_pv(pv)
        pv_cells = set(pv[1:]) if show_pv else None
        drag_target = None
        drag_side = None
        drag_source = None
        if ui.drag_move and ui.drag_move_from is not None and ui.drag_move_idx is not None:
            if 0 <= ui.drag_move_idx < len(board.history):
                drag_side = board.history[ui.drag_move_idx].side
                drag_source = ui.drag_move_from
                if (
                    ui.hover_cell is not None
                    and ui.hover_cell != ui.drag_move_from
                    and board.is_empty(*ui.hover_cell)
                ):
                    drag_target = ui.hover_cell
        draw_grid_and_stones(top_cell, top_visits, show_prior, skip_cell=None)
        if show_pv:
            draw_pv_ghosts(pv, core.current_side())
        if drag_side is not None:
            if drag_source is not None:
                draw_ghost_cell(drag_source, drag_side)
            if drag_target is not None:
                draw_ghost_cell(drag_target, drag_side)
        draw_next_future_outline()
        if drag_target is None and ui.hover_cell is not None and board.is_empty(*ui.hover_cell) and not show_pv:
            ax, ay = ui.hover_cell[0] - 1, ui.hover_cell[1] - 1
            cx, cy = center(ax, ay)
            dot_r = max(2, int(layout.r * 0.12))
            pygame.draw.circle(screen, HOVER_DOT, (int(cx), int(cy)), dot_r, 0)
        draw_borders()
        draw_side_coords()
        draw_analysis_text(show_prior, show_coords, suppress_cells=pv_cells)
        if not show_coords and show_pv:
            draw_pv_numbers(pv, core.current_side())
        if not show_coords:
            draw_move_numbers(ui.show_move_numbers)
        draw_movelist_panel()
        if ui.show_help:
            draw_help_overlay()

    ui = UiState()
    running = True
    while running:
        handle_events(ui)
        now = time.monotonic()
        show_prior, show_coords, top_cell, top_visits = update_frame_state(now, ui)
        draw_frame(show_prior, show_coords, top_cell, top_visits, ui)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
