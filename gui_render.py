from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

import pygame
import pygame.freetype

from board import HexBoard, MoveKind, Side, col_to_human_letters, coord_to_human
from engine import AnalysisMove
from gui_core import GuiCore

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

GRAPH_MIN_MOVES = 50
GRAPH_MIN_HEIGHT = 40
GRAPH_LINE_WIDTH = 2
GRAPH_DOT_RADIUS = 3
GRAPH_LABEL_PAD = 6
GRAPH_EDGE_PAD = 2
GRAPH_PAD = 0
GRAPH_ELO_CLAMP = 1000.0


class UiStateLike(Protocol):
    drag_move: bool
    drag_move_from: Optional[Tuple[int, int]]
    drag_move_idx: Optional[int]
    hover_cell: Optional[Tuple[int, int]]
    show_help: bool
    show_move_numbers: bool
    show_elo: bool
    show_engine_debug: bool
    last_cand_display: Optional[Tuple[int, int]]
    speed_vps: Optional[float]


@dataclass
class LayoutState:
    r: int
    wstep: float
    hstep: float
    origin_x: float
    origin_y: float
    board_px_w: int
    board_px_h: int


@dataclass
class BaselineText:
    font: pygame.freetype.Font
    baseline_offset: float
    line_ref_y: float

    @classmethod
    def for_font(cls, font: pygame.freetype.Font) -> "BaselineText":
        _surf, rect = font.render("0", fgcolor=BLACK)
        return cls(
            font=font,
            baseline_offset=rect.y - rect.height / 2,
            line_ref_y=rect.y,
        )

    def blit_center(self, text: str, color, cx: float, cy: float) -> None:
        screen = pygame.display.get_surface()
        surf, rect = self.font.render(text, fgcolor=color)
        x = cx - surf.get_width() / 2
        y = cy + self.baseline_offset - rect.y
        screen.blit(surf, (x, y))

    def blit_line(self, text: str, color, x: float, y: float) -> float:
        screen = pygame.display.get_surface()
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


class GuiRenderer:
    def __init__(self, board: HexBoard, core: GuiCore, *, flags: int) -> None:
        self.board = board
        self.core = core
        self.app = core.app
        self.flags = flags
        self.screen = pygame.display.set_mode((DEFAULT_WIN_W, DEFAULT_WIN_H), flags)
        self.layout = LayoutState(
            r=BASE_R,
            wstep=SQ3 * BASE_R,
            hstep=1.5 * BASE_R,
            origin_x=0.0,
            origin_y=0.0,
            board_px_w=DEFAULT_WIN_W - PANEL_W,
            board_px_h=DEFAULT_WIN_H,
        )
        self.fonts = FontState(
            board_small=self.make_board_small(self.layout.r),
            hud_font=pygame.freetype.Font(None, 20),
            hud_small=pygame.freetype.Font(None, 14),
            movelist_font=pygame.freetype.SysFont(["Menlo", "Consolas", "monospace"], 14),
            io_font=pygame.freetype.SysFont(["Menlo", "Consolas", "monospace"], 12),
        )
        self.text = TextRenderer.from_fonts(self.fonts)
        self.apply_window_size(DEFAULT_WIN_W, DEFAULT_WIN_H)

    def make_board_small(self, r: int) -> pygame.freetype.Font:
        t = clamp01((BASE_R - r) / (BASE_R - 8))
        fill = 0.65 + 0.10 * t
        small_sz = max(4, min(16, int(r * fill)))
        return pygame.freetype.Font(None, small_sz)

    def min_window_size(self, n: int) -> Tuple[int, int]:
        span = 1.5 * (n - 1)
        board_w = SQ3 * MIN_R * (span + 1)
        board_h = MIN_R * (span + 2)
        label_pad = LABEL_PAD_K * MIN_R
        win_w = int(board_w + label_pad + PANEL_W + 2 * BOARD_PAD)
        win_h = int(board_h + label_pad + HUD_H + 2 * BOARD_PAD)
        return win_w, win_h

    def apply_window_size(self, win_w: int, win_h: int) -> Tuple[int, int]:
        min_w, min_h = self.min_window_size(self.board.n)
        win_w = max(win_w, min_w)
        win_h = max(win_h, min_h)
        if (win_w, win_h) != pygame.display.get_window_size():
            self.screen = pygame.display.set_mode((win_w, win_h), self.flags)
        self.layout.board_px_w = max(0, win_w - PANEL_W)
        self.layout.board_px_h = win_h

        usable_w0 = max(0.0, self.layout.board_px_w - 2 * BOARD_PAD)
        usable_h0 = max(0.0, self.layout.board_px_h - HUD_H - 2 * BOARD_PAD)

        span = 1.5 * (self.board.n - 1)
        denom_w = SQ3 * (span + 1)
        denom_h = span + 2
        pad_k = LABEL_PAD_K
        r_w = usable_w0 / (denom_w + pad_k) if denom_w > 0 else MIN_R
        r_h = usable_h0 / (denom_h + pad_k) if denom_h > 0 else MIN_R
        self.layout.r = int(max(MIN_R, min(r_w, r_h, MAX_R)))

        self.layout.wstep = SQ3 * self.layout.r
        self.layout.hstep = 1.5 * self.layout.r
        label_pad = pad_k * self.layout.r
        usable_w = max(0.0, usable_w0 - label_pad)
        usable_h = max(0.0, usable_h0 - label_pad)

        board_w = SQ3 * self.layout.r * (span + 1)
        board_h = self.layout.r * (span + 2)
        extra_w = max(0.0, usable_w - board_w)
        extra_h = max(0.0, usable_h - board_h)
        self.layout.origin_x = BOARD_PAD + label_pad + extra_w / 2 + (SQ3 * self.layout.r) / 2
        self.layout.origin_y = HUD_H + BOARD_PAD + label_pad + extra_h / 2 + self.layout.r

        self.fonts.board_small = self.make_board_small(self.layout.r)
        self.text.update_board_small(self.fonts.board_small)
        pygame.display.set_caption(f"Hex {self.board.n}x{self.board.n}")
        return win_w, win_h

    def window_to_surface_pos(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        wx, wy = pygame.display.get_window_size()
        surf = pygame.display.get_surface()
        if surf is None:
            return pos
        sx, sy = surf.get_size()
        if wx <= 0 or wy <= 0:
            return pos
        if (wx, wy) == (sx, sy):
            return pos
        scale_x = sx / wx
        scale_y = sy / wy
        return (int(pos[0] * scale_x), int(pos[1] * scale_y))

    def center(self, ax: int, ay: int) -> Tuple[float, float]:
        cx = self.layout.origin_x + self.layout.wstep * (ax + ay / 2)
        cy = self.layout.origin_y + self.layout.hstep * ay
        return cx, cy

    def corner(self, ax: int, ay: int, i: int) -> Tuple[int, int]:
        cx, cy = self.center(ax, ay)
        a = math.radians(CORNER_DEG[i % 6])
        return round(cx + self.layout.r * math.cos(a)), round(cy + self.layout.r * math.sin(a))

    def poly(self, ax: int, ay: int) -> List[Tuple[int, int]]:
        return [self.corner(ax, ay, i) for i in range(6)]

    def pixel_to_cell(self, mx: int, my: int) -> Optional[Tuple[int, int]]:
        if my < HUD_H:
            return None
        if mx >= self.layout.board_px_w:
            return None
        best: Optional[Tuple[int, int]] = None
        best_d2 = 10**18
        for row in range(1, self.board.n + 1):
            for col in range(1, self.board.n + 1):
                ax, ay = col - 1, row - 1
                cx, cy = self.center(ax, ay)
                dx = mx - cx
                dy = my - cy
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best = (col, row)
        if best is None:
            return None
        if best_d2 <= (self.layout.r * 0.98) ** 2:
            return best
        return None

    def get_hover_pv(
        self,
        hover_cell: Optional[Tuple[int, int]],
    ) -> Optional[Tuple[Tuple[int, int], ...]]:
        if hover_cell is None:
            return None
        if self.app.candidates:
            return None
        col, row = hover_cell
        if not self.board.is_empty(col, row):
            return None
        for r in self.core.get_active_analysis():
            if r.col == col and r.row == row and r.pv:
                return r.pv
        return None

    @staticmethod
    def should_show_pv(pv: Optional[Tuple[Tuple[int, int], ...]]) -> bool:
        return bool(pv) and len(pv) > 1

    def draw_grid_and_stones(
        self,
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
        for r in self.core.get_active_analysis():
            if r.col is None or r.row is None:
                continue
            if not self.board.is_empty(r.col, r.row):
                continue
            if r.visits is not None:
                visits_map[(r.col, r.row)] = r.visits
            if r.winrate is not None:
                winrate_map[(r.col, r.row)] = r.winrate
            if r.prior is not None:
                prior_map[(r.col, r.row)] = r.prior
        if self.app.candidates:
            for key in self.app.candidates:
                candidate_wr_map[key] = self.app.candidate_results.get(key, (None, None))[0]
        cand_count = len(self.app.candidates) if self.app.candidates else 0

        denom = math.log(max(2, top_visits))
        max_prior = max(prior_map.values()) if prior_map else None

        for row in range(1, self.board.n + 1):
            for col in range(1, self.board.n + 1):
                ax, ay = col - 1, row - 1
                occ = self.board.get(col, row)
                if skip_cell is not None and skip_cell == (col, row):
                    occ = -1

                if occ < 0:
                    if (col, row) in self.app.candidates:
                        cand_wr = candidate_wr_map.get((col, row))
                        if cand_wr is None:
                            fill = CANDIDATE_UNKNOWN
                        else:
                            t = clamp01(cand_wr) ** 0.9
                            fill = lerp_rgb(CANDIDATE_LOW, CANDIDATE_HIGH, t)
                        if (
                            cand_count > 1
                            and self.app.candidate_run is not None
                            and self.app.candidate_run.key == (col, row)
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

                pts = self.poly(ax, ay)
                pygame.draw.polygon(self.screen, fill, pts, 0)
                pygame.draw.polygon(self.screen, GRID_EDGE, pts, 1)

    def draw_next_future_outline(self) -> None:
        if not self.app.future_moves:
            return
        mv = self.app.future_moves[-1]
        coords = self.core.move_coords(mv)
        if coords is None:
            return
        col, row = coords
        if not self.board.is_empty(col, row):
            return

        ax, ay = col - 1, row - 1
        pts = self.poly(ax, ay)
        colr = RED if self.core.current_side() == Side.RED else BLUE
        thickness = max(3, int(self.layout.r * 0.12))
        pygame.draw.polygon(self.screen, colr, pts, thickness)

    def draw_move_numbers(self, show_all: bool) -> None:
        if not self.board.history:
            return
        last_idx = len(self.board.history) - 1
        if show_all:
            number_tint = 0.45
            for idx in range(len(self.board.history)):
                mv = self.board.history[idx]
                coords = self.core.move_coords(mv)
                if coords is None:
                    continue
                ax, ay = coords[0] - 1, coords[1] - 1
                cx, cy = self.center(ax, ay)
                txt = "S" if self.core.is_swapped_stone_index(idx) else str(idx + 1)
                swap_active_here = (
                    self.core.is_swapped_stone_index(idx)
                    and self.board.history[last_idx].kind == MoveKind.SWAP
                )
                if idx == last_idx or swap_active_here:
                    colr = OFF_WHITE
                else:
                    base = RED if mv.side == Side.RED else BLUE
                    colr = lerp_rgb(base, OFF_WHITE, number_tint)
                self.text.board.blit_center(txt, colr, cx, cy)
            return

        mv = self.board.history[-1]
        coords = self.core.move_coords(mv)
        if coords is None and mv.kind == MoveKind.SWAP and self.board.history:
            coords = self.core.move_coords(self.board.history[0])
        if coords is None:
            return
        ax, ay = coords[0] - 1, coords[1] - 1
        cx, cy = self.center(ax, ay)
        dot_r = max(2, int(self.layout.r * 0.18))
        pygame.draw.circle(self.screen, OFF_WHITE, (int(cx), int(cy)), dot_r, 0)

    def draw_borders(self) -> None:
        thickness = 4
        sides = [
            {"color": RED, "segs": [(2, 3), (3, 4)], "coord": lambda i: (i, 0)},
            {"color": RED, "segs": [(5, 0), (0, 1)], "coord": lambda i: (i, self.board.n - 1)},
            {"color": BLUE, "segs": [(4, 5), (5, 0)], "coord": lambda i: (0, i)},
            {"color": BLUE, "segs": [(1, 2), (2, 3)], "coord": lambda i: (self.board.n - 1, i)},
        ]
        for side in sides:
            for i in range(self.board.n):
                ax, ay = side["coord"](i)
                for c1, c2 in side["segs"]:
                    pygame.draw.line(
                        self.screen, side["color"], self.corner(ax, ay, c1), self.corner(ax, ay, c2), thickness
                    )

    def draw_side_coords(self) -> None:
        col_color = GRAY
        row_color = GRAY
        for col in range(1, self.board.n + 1):
            ax, ay = col - 1, -1
            cx, cy = self.center(ax, ay)
            txt = col_to_human_letters(col)
            self.text.board.blit_center(txt, col_color, cx, cy)
        for row in range(1, self.board.n + 1):
            ax, ay = -1, row - 1
            cx, cy = self.center(ax, ay)
            txt = str(row)
            self.text.board.blit_center(txt, row_color, cx, cy)

    def draw_ghost_cell(self, cell: Tuple[int, int], side: Side) -> None:
        ax, ay = cell[0] - 1, cell[1] - 1
        pts = self.poly(ax, ay)
        base = RED if side == Side.RED else BLUE
        ghost = lerp_rgb(base, OFF_WHITE, 0.5)
        pygame.draw.polygon(self.screen, ghost, pts, 0)
        pygame.draw.polygon(self.screen, GRID_EDGE, pts, 1)

    def draw_pv_ghosts(self, pv: Tuple[Tuple[int, int], ...], start_side: Side) -> None:
        side = start_side
        for cell in pv:
            if self.board.is_empty(*cell):
                self.draw_ghost_cell(cell, side)
            side = self.core.flip_side(side)

    def draw_pv_numbers(self, pv: Tuple[Tuple[int, int], ...], start_side: Side) -> None:
        side = start_side
        for idx, cell in enumerate(pv):
            if idx == 0:
                side = self.core.flip_side(side)
                continue
            if not self.board.is_empty(*cell):
                side = self.core.flip_side(side)
                continue
            ax, ay = cell[0] - 1, cell[1] - 1
            cx, cy = self.center(ax, ay)
            self.text.board.blit_center(str(idx + 1), OFF_WHITE, cx, cy)
            side = self.core.flip_side(side)

    def draw_analysis_text(
        self,
        ui: UiStateLike,
        show_prior: bool,
        show_coords: bool,
        *,
        suppress_cells: Optional[set[Tuple[int, int]]] = None,
    ) -> None:
        if show_coords:
            for row in range(1, self.board.n + 1):
                for col in range(1, self.board.n + 1):
                    ax, ay = col - 1, row - 1
                    cx, cy = self.center(ax, ay)

                    occ = self.board.get(col, row)
                    colr = WHITE if occ >= 0 else BLACK

                    txt = coord_to_human(col, row)
                    self.text.board.blit_center(txt, colr, cx, cy)
            return

        for r in self.core.get_active_analysis():
            if r.col is None or r.row is None:
                continue
            col, row = r.col, r.row
            if suppress_cells and (col, row) in suppress_cells:
                continue
            if not self.board.is_empty(col, row):
                continue
            ax, ay = col - 1, row - 1
            cx, cy = self.center(ax, ay)

            if show_prior:
                pr = fmt_prior(r.prior)
                if not pr:
                    continue
                self.text.board.blit_center(pr, BLACK, cx, cy)
                continue

            wr = fmt_wr_or_elo(r.winrate, ui.show_elo)
            vv = fmt_visits(r.visits)
            if not wr and not vv:
                continue

            gap = 1
            rect1 = self.text.board.font.get_rect(wr)
            rect2 = self.text.board.font.get_rect(vv)
            total_h = rect1.height + gap + rect2.height
            top = cy - total_h / 2
            x1 = cx - rect1.width / 2
            y1 = top - self.text.board.line_ref_y + rect1.y
            self.text.board.blit_line(wr, BLACK, x1, y1)
            top2 = top + rect1.height + gap
            x2 = cx - rect2.width / 2
            y2 = top2 - self.text.board.line_ref_y + rect2.y
            self.text.board.blit_line(vv, BLACK, x2, y2)

    def blit_segments(
        self, x: int, y: int, parts: List[Tuple[str, Tuple[int, int, int]]], use_small: bool
    ) -> None:
        blit_line = self.text.hud_small.blit_line if use_small else self.text.hud.blit_line
        cx = x
        for txt, col in parts:
            cx += blit_line(txt, col, cx, y)

    def draw_eval_graph(
        self,
        rect: pygame.Rect,
        cursor_ply: int,
        total_moves: int,
        show_elo: bool,
    ) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, rect)
        pygame.draw.rect(self.screen, PANEL_EDGE, rect, 1)

        n_moves = total_moves
        if rect.width <= 1 or rect.height <= 1:
            return

        def best_reply_winrate(ply_len: int, side_to_play: Side) -> Optional[float]:
            recs = self.app.analysis_cache.get((ply_len, int(side_to_play)))
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
        pygame.draw.line(self.screen, GRID_EDGE, (rect.left, mid_y), (rect.right, mid_y), 1)

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
                pygame.draw.lines(self.screen, RED, False, points, GRAPH_LINE_WIDTH)
                pygame.draw.aalines(self.screen, RED, False, points)
            elif len(points) == 1:
                pygame.draw.circle(self.screen, RED, points[0], 2, 0)

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
                pygame.draw.circle(self.screen, RED, (cx, cy), GRAPH_DOT_RADIUS, 0)

                label = fmt_wr_or_elo(val, show_elo)
                label_rect = self.text.hud_small.font.get_rect(label)
                line_h = self.text.hud_small.font.get_sized_height()
                lx = cx + GRAPH_LABEL_PAD
                ly = cy - line_h / 2
                if lx + label_rect.width > rect.right - GRAPH_EDGE_PAD:
                    lx = cx - label_rect.width - GRAPH_LABEL_PAD
                if ly < rect.top + GRAPH_EDGE_PAD:
                    ly = rect.top + GRAPH_EDGE_PAD
                if ly + line_h > rect.bottom - GRAPH_EDGE_PAD:
                    ly = rect.bottom - line_h - GRAPH_EDGE_PAD
                self.text.hud_small.blit_line(label, RED, lx, ly)

    def draw_movelist_panel(self, ui: UiStateLike) -> None:
        x0 = self.layout.board_px_w
        pygame.draw.rect(self.screen, PANEL_BG, pygame.Rect(x0, 0, PANEL_W, self.screen.get_height()))

        pad = 12
        y = 10
        self.blit_segments(x0 + pad, y, [("Moves", BLACK)], use_small=False)
        y += 26

        moves = list(self.board.history)
        if self.app.future_moves:
            moves.extend(reversed(self.app.future_moves))

        total_moves = len(moves)
        cursor_ply = len(self.board.history)
        nrows = (total_moves + 1) // 2

        line_h = self.fonts.movelist_font.get_sized_height() + 4
        io_line_h = self.text.line_io + 2
        io_max_lines = 30
        io_panel_h = (io_line_h * io_max_lines) + 10
        graph_h = 0
        if not ui.show_engine_debug:
            avail = self.screen.get_height() - y
            graph_h = max(0, min(PANEL_W, avail))
        graph_top = self.screen.get_height() - graph_h - GRAPH_PAD
        io_top = max(y, self.screen.get_height() - io_panel_h) if ui.show_engine_debug else graph_top
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

            red_mv = self.core.move_to_label_in_sequence(moves, red_i)
            blue_mv = (
                self.core.move_to_label_in_sequence(moves, blue_i) if blue_i < total_moves else None
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
                cx += self.text.movelist.blit_line(txt, col, cx, y)
            y += line_h

        if ui.show_engine_debug:
            io_rect = pygame.Rect(x0, io_top, PANEL_W, self.screen.get_height() - io_top)
            pygame.draw.rect(self.screen, PANEL_BG, io_rect)
            pygame.draw.rect(self.screen, PANEL_EDGE, io_rect, 1)

            io_y = io_top + 4
            logs = self.core.engine.get_io_log(io_max_lines)
            start_idx = max(0, len(logs) - io_max_lines)
            for direction, msg, count in logs[start_idx:]:
                prefix = ">>" if direction == "out" else "<<"
                if count > 1:
                    line = f"{prefix} {msg} ({count})"
                else:
                    line = f"{prefix} {msg}"
                self.text.io.blit_line(line, BLACK, x0 + pad, io_y)
                io_y += io_line_h
        elif graph_h >= GRAPH_MIN_HEIGHT:
            graph_rect = pygame.Rect(
                x0,
                graph_top,
                PANEL_W,
                self.screen.get_height() - graph_top,
            )
            self.draw_eval_graph(graph_rect, cursor_ply, total_moves, ui.show_elo)

        pygame.draw.line(self.screen, PANEL_EDGE, (x0, 0), (x0, self.screen.get_height()), 1)

    def draw_hud(self, ui: UiStateLike) -> None:
        pygame.draw.rect(self.screen, BG, pygame.Rect(0, 0, self.screen.get_width(), HUD_H))

        turn_side = self.core.current_side()
        turn_color = RED if turn_side == Side.RED else BLUE
        turn_name = "Red" if turn_side == Side.RED else "Blue"
        analysis_txt = "ON" if self.app.analysis_running else "OFF"
        analysis_color = BLACK if self.app.analysis_running else GRAY

        parts: List[Tuple[str, Tuple[int, int, int]]] = [
            ("Size: ", BLACK),
            (f"{self.board.n}", BLACK),
        ]
        if self.app.pending_size != self.board.n:
            parts += [
                ("  (pending ", BLACK),
                (f"{self.app.pending_size}", BLACK),
                (")", BLACK),
            ]
        parts += [
            ("   |   Turn: ", BLACK),
            (turn_name, turn_color),
            ("   |   ", BLACK),
            ("Analysis: ", analysis_color),
            (analysis_txt, analysis_color),
        ]
        if self.app.candidates:
            cand_key = (
                self.app.candidate_run.key if self.app.candidate_run is not None else ui.last_cand_display
            )
            if cand_key is None:
                next_keys = self.core.sorted_candidates_by_visits()
                cand_key = next_keys[0] if next_keys else None
            parts += [("   |   ", BLACK), ("Cand: ", BLACK)]
            if cand_key is not None:
                parts += [(coord_to_human(*cand_key), turn_color)]
        else:
            display: Optional[AnalysisMove] = None
            for r in self.core.get_active_analysis():
                if r.col is None or r.row is None:
                    continue
                if not self.board.is_empty(r.col, r.row):
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
        self.blit_segments(12, 10, parts, use_small=False)

        help_line = "space:analysis • ,:play best • s:swap • +/-/enter:size • ?:help"
        self.text.hud_small.blit_line(help_line, BLACK, 12, 32)

        awrn = f"{self.app.analysis_wide_root_noise:.2f}".rstrip("0").rstrip(".")
        awrn_text = "AWRN –" if self.app.candidates else f"AWRN {awrn}"
        awrn_w = self.fonts.hud_small.get_rect(awrn_text).width
        awrn_x = max(12, self.layout.board_px_w - awrn_w - 12)
        self.text.hud_small.blit_line(awrn_text, GRAY, awrn_x, 10)
        vps_suffix = " visits/s"
        if ui.speed_vps is not None and ui.speed_vps > 0:
            vps_text = f"{fmt_visits(int(ui.speed_vps))}{vps_suffix}"
        else:
            vps_text = f"–{vps_suffix}"
        vps_w = self.fonts.hud_small.get_rect(vps_text).width
        vps_x = max(12, self.layout.board_px_w - vps_w - 12)
        vps_y = 10 + self.text.line_hud_small + 2
        self.text.hud_small.blit_line(vps_text, GRAY, vps_x, vps_y)

    def draw_help_overlay(self) -> None:
        lines = [
            "Help (? to hide)",
            "space:analysis   ,:play best/PV   esc:quit",
            "p:prev   n:next   f:first   l:last   scroll:prev/next",
            "ctrl+p:prev 10   ctrl+n:next 10",
            "shift+p:pass   s:swap",
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
        surfs = [self.fonts.hud_small.render(line, fgcolor=BLACK)[0] for line in lines]
        w = max(s.get_width() for s in surfs)
        line_h = self.text.line_hud_small
        h = line_h * len(lines) + gap * (len(surfs) - 1)
        x = 12
        y = HUD_H + 12
        rect = pygame.Rect(x - pad, y - pad, w + pad * 2, h + pad * 2)
        pygame.draw.rect(self.screen, PANEL_BG, rect)
        pygame.draw.rect(self.screen, PANEL_EDGE, rect, 1)
        for line in lines:
            self.text.hud_small.blit_line(line, BLACK, x, y)
            y += line_h + gap

    def draw_frame(
        self,
        ui: UiStateLike,
        show_prior: bool,
        show_coords: bool,
        top_cell: Optional[Tuple[int, int]],
        top_visits: int,
    ) -> None:
        self.screen.fill(BG)
        self.draw_hud(ui)
        pv = self.get_hover_pv(ui.hover_cell)
        show_pv = self.should_show_pv(pv)
        pv_cells = set(pv[1:]) if show_pv else None
        drag_target = None
        drag_side = None
        drag_source = None
        if ui.drag_move and ui.drag_move_from is not None and ui.drag_move_idx is not None:
            if 0 <= ui.drag_move_idx < len(self.board.history):
                drag_side = self.board.history[ui.drag_move_idx].side
                drag_source = ui.drag_move_from
                if (
                    ui.hover_cell is not None
                    and ui.hover_cell != ui.drag_move_from
                    and self.board.is_empty(*ui.hover_cell)
                ):
                    drag_target = ui.hover_cell
        self.draw_grid_and_stones(top_cell, top_visits, show_prior, skip_cell=None)
        if show_pv and pv is not None:
            self.draw_pv_ghosts(pv, self.core.current_side())
        if drag_side is not None:
            if drag_source is not None:
                self.draw_ghost_cell(drag_source, drag_side)
            if drag_target is not None:
                self.draw_ghost_cell(drag_target, drag_side)
        self.draw_next_future_outline()
        if drag_target is None and ui.hover_cell is not None and self.board.is_empty(*ui.hover_cell) and not show_pv:
            ax, ay = ui.hover_cell[0] - 1, ui.hover_cell[1] - 1
            cx, cy = self.center(ax, ay)
            dot_r = max(2, int(self.layout.r * 0.12))
            pygame.draw.circle(self.screen, HOVER_DOT, (int(cx), int(cy)), dot_r, 0)
        self.draw_borders()
        self.draw_side_coords()
        self.draw_analysis_text(ui, show_prior, show_coords, suppress_cells=pv_cells)
        if not show_coords and show_pv and pv is not None:
            self.draw_pv_numbers(pv, self.core.current_side())
        if not show_coords:
            self.draw_move_numbers(ui.show_move_numbers)
        self.draw_movelist_panel(ui)
        if ui.show_help:
            self.draw_help_overlay()
