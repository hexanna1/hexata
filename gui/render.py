from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Tuple

import pygame
import pygame.freetype

from board import HexBoard, Move, MoveKind, Side, col_to_human_letters, coord_to_human
from engine import AnalysisMove
from gui.core import GuiCore

RED = (220, 60, 60)
BLUE = (40, 100, 220)
TEXT_ON_LIGHT = (15, 15, 15)
TEXT_ON_DARK = (250, 250, 250)
TEXT_MUTED = (132, 132, 132)
SURFACE_BG = (237, 230, 217)
BOARD_EMPTY = (228, 216, 194)
LINE = (130, 118, 100)
HELP_HEADER = tuple((RED[i] + BLUE[i]) // 2 for i in range(3))


def _bg_relative_color(
    delta: Tuple[int, int, int], *, base: Optional[Tuple[int, int, int]] = None
) -> Tuple[int, int, int]:
    def clamp_byte(v: int) -> int:
        return max(0, min(255, v))

    if base is None:
        base = BOARD_EMPTY

    return (
        clamp_byte(base[0] + delta[0]),
        clamp_byte(base[1] + delta[1]),
        clamp_byte(base[2] + delta[2]),
    )


SURFACE_PANEL = _bg_relative_color((-4, -4, -6), base=SURFACE_BG)
ANALYSIS_LOW = _bg_relative_color((4, -5, -12))
ANALYSIS_HIGH = (160, 210, 140)
ANALYSIS_BEST = (175, 235, 240)
CANDIDATE_LOW = _bg_relative_color((-2, -9, 18))
CANDIDATE_HIGH = (170, 125, 210)
CANDIDATE_UNKNOWN = _bg_relative_color((-18, -27, 13))
CANDIDATE_ACTIVE = (250, 196, 92)

HUD_H = 48
BOARD_PAD = 16
MIN_R = 4
BASE_R = 24
MAX_R = 38
LABEL_PAD_K = 0.85
PANEL_W = 230
MOVELIST_GUTTER_GAP = 8
DEFAULT_WIN_W = 1456
DEFAULT_WIN_H = 808

SQ3 = math.sqrt(3)
POINTY_CORNER_DEG = (90, 30, -30, -90, -150, 150)

GRAPH_MIN_MOVES = 50
GRAPH_MIN_HEIGHT = 40
GRAPH_LINE_WIDTH = 2
GRAPH_DOT_RADIUS = 3
GRAPH_LABEL_PAD = 6
GRAPH_EDGE_PAD = 2
GRAPH_PAD = 0
GRAPH_ELO_CLAMP = 1000.0


class UiPrefsLike(Protocol):
    show_move_numbers: bool
    show_elo: bool


class UiStateLike(Protocol):
    drag_move: bool
    drag_move_from: Optional[Tuple[int, int]]
    drag_move_idx: Optional[int]
    show_help: bool
    prefs: UiPrefsLike
    show_engine_debug: bool
    last_cand_display: Optional[Tuple[int, int]]
    speed_vps: Optional[float]
    current_engine_name: Optional[str]
    has_multiple_engines: bool


@dataclass(frozen=True, slots=True)
class BorderSpec:
    side: Side
    corner_pairs: Tuple[Tuple[int, int], ...]
    edge_cell: Callable[[int, int], Tuple[int, int]]


@dataclass(frozen=True, slots=True)
class BoardProjection:
    corner_degrees: Tuple[int, ...]
    col_unit: Tuple[float, float]
    row_unit: Tuple[float, float]
    border_specs: Tuple[BorderSpec, ...]

    def center(
        self, origin_x: float, origin_y: float, r: int, ax: int, ay: int
    ) -> Tuple[float, float]:
        return (
            origin_x + r * (self.col_unit[0] * ax + self.row_unit[0] * ay),
            origin_y + r * (self.col_unit[1] * ax + self.row_unit[1] * ay),
        )

    def corner(
        self, origin_x: float, origin_y: float, r: int, ax: int, ay: int, i: int
    ) -> Tuple[int, int]:
        cx, cy = self.center(origin_x, origin_y, r, ax, ay)
        a = math.radians(self.corner_degrees[i % 6])
        return round(cx + r * math.cos(a)), round(cy + r * math.sin(a))

    def poly(
        self, origin_x: float, origin_y: float, r: int, ax: int, ay: int
    ) -> List[Tuple[int, int]]:
        return [self.corner(origin_x, origin_y, r, ax, ay, i) for i in range(6)]

    def board_bounds(self, n: int, r: int) -> Tuple[float, float, float, float]:
        xs: List[float] = []
        ys: List[float] = []
        for ay in (0, n - 1):
            for ax in (0, n - 1):
                cx, cy = self.center(0.0, 0.0, r, ax, ay)
                for deg in self.corner_degrees:
                    a = math.radians(deg)
                    xs.append(cx + r * math.cos(a))
                    ys.append(cy + r * math.sin(a))
        return min(xs), min(ys), max(xs), max(ys)

    def board_size(self, n: int, r: int) -> Tuple[float, float]:
        left, top, right, bottom = self.board_bounds(n, r)
        return right - left, bottom - top

    def layout_size(self, n: int, r: int) -> Tuple[float, float]:
        board_w, board_h = self.board_size(n, r)
        label_pad = LABEL_PAD_K * r
        return board_w + label_pad, board_h + label_pad

    def origin_for_board(self, left: float, top: float, n: int, r: int) -> Tuple[float, float]:
        bounds_left, bounds_top, _right, _bottom = self.board_bounds(n, r)
        return left - bounds_left, top - bounds_top

    def border_sides(self) -> Tuple[BorderSpec, ...]:
        return self.border_specs

    def col_label_anchor(self, col: int) -> Tuple[int, int]:
        return col - 1, -1

    def row_label_anchor(self, row: int) -> Tuple[int, int]:
        return -1, row - 1


POINTY_PROJECTION = BoardProjection(
    corner_degrees=POINTY_CORNER_DEG,
    col_unit=(SQ3, 0.0),
    row_unit=(SQ3 / 2, 1.5),
    border_specs=(
        BorderSpec(Side.RED, ((2, 3), (3, 4)), lambda i, n: (i, 0)),
        BorderSpec(Side.RED, ((5, 0), (0, 1)), lambda i, n: (i, n - 1)),
        BorderSpec(Side.BLUE, ((4, 5), (5, 0)), lambda i, n: (0, i)),
        BorderSpec(Side.BLUE, ((1, 2), (2, 3)), lambda i, n: (n - 1, i)),
    ),
)


@dataclass(slots=True)
class LayoutState:
    r: int
    origin_x: float
    origin_y: float
    board_px_w: int
    board_px_h: int


@dataclass(slots=True)
class BaselineText:
    font: pygame.freetype.Font
    baseline_offset: float
    line_ref_y: float

    @classmethod
    def for_font(cls, font: pygame.freetype.Font) -> "BaselineText":
        _surf, rect = font.render("0", fgcolor=TEXT_ON_LIGHT)
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


@dataclass(slots=True)
class FontState:
    board_small: pygame.freetype.Font
    hud_font: pygame.freetype.Font
    hud_small: pygame.freetype.Font
    movelist_font: pygame.freetype.Font
    io_font: pygame.freetype.Font


@dataclass(slots=True)
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
    if w <= 0.0:
        return "-inf"
    if w >= 1.0:
        return "+inf"
    elo = int(round(400.0 * math.log10(w / (1.0 - w))))
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
        self.projection = POINTY_PROJECTION
        self.screen = pygame.display.set_mode((DEFAULT_WIN_W, DEFAULT_WIN_H), flags)
        self.layout = LayoutState(
            r=BASE_R,
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
        self._eval_graph_sig: Optional[tuple[int, int, tuple[Move, ...]]] = None
        self._eval_graph_data: Optional[tuple[tuple[Move, ...], tuple[bytes, ...]]] = None
        self._movelist_sig: Optional[tuple] = None
        self._movelist_view = None
        self._frozen_pv_sig: Optional[Tuple[Optional[Tuple[int, int]], int]] = None
        self._frozen_pv: Optional[Tuple[Tuple[int, int], ...]] = None
        self.apply_window_size(DEFAULT_WIN_W, DEFAULT_WIN_H)

    def make_board_small(self, r: int) -> pygame.freetype.Font:
        t = clamp01((BASE_R - r) / (BASE_R - 8))
        fill = 0.65 + 0.10 * t
        small_sz = max(4, min(16, int(r * fill)))
        return pygame.freetype.Font(None, small_sz)

    def min_window_size(self, n: int) -> Tuple[int, int]:
        layout_w, layout_h = self.projection.layout_size(n, MIN_R)
        win_w = int(layout_w + PANEL_W + 2 * BOARD_PAD)
        win_h = int(layout_h + HUD_H + 2 * BOARD_PAD)
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

        denom_w, denom_h = self.projection.layout_size(self.board.n, 1)
        r_w = usable_w0 / denom_w if denom_w > 0 else MIN_R
        r_h = usable_h0 / denom_h if denom_h > 0 else MIN_R
        self.layout.r = int(max(MIN_R, min(r_w, r_h, MAX_R)))

        label_pad = LABEL_PAD_K * self.layout.r
        layout_w, layout_h = self.projection.layout_size(self.board.n, self.layout.r)
        extra_w = max(0.0, usable_w0 - layout_w)
        extra_h = max(0.0, usable_h0 - layout_h)
        board_left = BOARD_PAD + label_pad + extra_w / 2
        board_top = HUD_H + BOARD_PAD + label_pad + extra_h / 2
        self.layout.origin_x, self.layout.origin_y = self.projection.origin_for_board(
            board_left, board_top, self.board.n, self.layout.r
        )

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
        return self.projection.center(
            self.layout.origin_x, self.layout.origin_y, self.layout.r, ax, ay
        )

    def corner(self, ax: int, ay: int, i: int) -> Tuple[int, int]:
        return self.projection.corner(
            self.layout.origin_x, self.layout.origin_y, self.layout.r, ax, ay, i
        )

    def poly(self, ax: int, ay: int) -> List[Tuple[int, int]]:
        return self.projection.poly(
            self.layout.origin_x, self.layout.origin_y, self.layout.r, ax, ay
        )

    def pixel_to_cell(self, mx: int, my: int) -> Optional[Tuple[int, int]]:
        if my < HUD_H:
            return None
        if mx >= self.layout.board_px_w:
            return None
        best: Optional[Tuple[int, int]] = None
        best_d2 = 10**18
        for row in range(0, self.board.n + 2):
            for col in range(0, self.board.n + 2):
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
        col, row = best
        if 1 <= col <= self.board.n and 1 <= row <= self.board.n:
            return best
        return None

    def get_display_pv(
        self,
        hover_cell: Optional[Tuple[int, int]],
    ) -> Optional[Tuple[Tuple[int, int], ...]]:
        cell = hover_cell
        if (
            cell is None
            or self.app.candidate_state.candidates
            or (not self.board.is_empty(cell[0], cell[1]))
        ):
            cell = None

        sig = (cell, self.board.rev)
        # Intentionally freeze the displayed PV while hovering the same cell on the
        # same position to avoid distracting PV flicker as analysis updates stream in.
        if self._frozen_pv_sig == sig:
            return self._frozen_pv
        self._frozen_pv_sig = sig

        if cell is None:
            self._frozen_pv = None
            return None

        col, row = cell
        pv = None
        for r in self.core.get_active_analysis():
            if r.col == col and r.row == row and r.pv:
                pv = r.pv
                break
        self._frozen_pv = pv
        return pv

    @staticmethod
    def should_show_pv(pv: Optional[Tuple[Tuple[int, int], ...]]) -> bool:
        return bool(pv) and len(pv) > 1

    def draw_grid_and_stones(
        self,
        top_cell: Optional[Tuple[int, int]],
        top_visits: int,
        show_prior: bool,
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
        if self.app.candidate_state.candidates:
            for key in self.app.candidate_state.candidates:
                candidate_wr_map[key] = self.app.candidate_state.results.get(key, (None, None))[0]
        cand_count = len(self.app.candidate_state.candidates) if self.app.candidate_state.candidates else 0

        denom = math.log(max(2, top_visits))
        max_prior = max(prior_map.values()) if prior_map else None

        for row in range(1, self.board.n + 1):
            for col in range(1, self.board.n + 1):
                ax, ay = col - 1, row - 1
                occ = self.board.get(col, row)

                if occ < 0:
                    if (col, row) in self.app.candidate_state.candidates:
                        cand_wr = candidate_wr_map.get((col, row))
                        if cand_wr is None:
                            fill = CANDIDATE_UNKNOWN
                        else:
                            t = clamp01(cand_wr) ** 0.9
                            fill = lerp_rgb(CANDIDATE_LOW, CANDIDATE_HIGH, t)
                        if (
                            cand_count > 1
                            and self.app.candidate_state.run is not None
                            and self.app.candidate_state.run.key == (col, row)
                        ):
                            fill = CANDIDATE_ACTIVE
                    elif show_prior:
                        pr = prior_map.get((col, row))
                        if pr is None:
                            fill = BOARD_EMPTY
                        else:
                            t = clamp01(pr) ** 0.9
                            fill = lerp_rgb(ANALYSIS_LOW, ANALYSIS_HIGH, t)
                            if max_prior is not None and pr >= max_prior - 0.001:
                                fill = ANALYSIS_BEST
                    else:
                        v = visits_map.get((col, row))
                        wr = winrate_map.get((col, row))
                        if v is None or wr is None:
                            fill = BOARD_EMPTY
                        else:
                            t = 0.0 if denom <= 0 else (math.log(max(1, v)) / denom)
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
                pygame.draw.polygon(self.screen, LINE, pts, 1)

    def draw_next_future_outline(self) -> None:
        mv = self.core.next_mainline_move()
        thickness = max(3, int(self.layout.r * 0.12))
        if mv is not None:
            coords = self.core.move_coords(mv)
            if coords is not None:
                col, row = coords
                if self.board.is_empty(col, row):
                    ax, ay = col - 1, row - 1
                    pts = self.poly(ax, ay)
                    colr = RED if self.core.current_side() == Side.RED else BLUE
                    pygame.draw.polygon(self.screen, colr, pts, thickness)

        for mv in self.core.next_variation_moves():
            coords = self.core.move_coords(mv)
            if coords is None:
                continue
            col, row = coords
            if not self.board.is_empty(col, row):
                continue
            ax, ay = col - 1, row - 1
            pts = self.poly(ax, ay)
            colr = lerp_rgb(RED if mv.side == Side.RED else BLUE, BOARD_EMPTY, 0.5)
            pygame.draw.polygon(self.screen, colr, pts, thickness)

    def draw_move_numbers(self, show_all: bool) -> None:
        history = self.core.applied_history()
        if not history:
            return
        last_idx = len(history) - 1
        if show_all:
            number_tint = 0.45
            for idx, mv in enumerate(history):
                coords = self.core.move_coords(mv)
                if coords is None:
                    continue
                ax, ay = coords[0] - 1, coords[1] - 1
                cx, cy = self.center(ax, ay)
                txt = "S" if self.core.is_swapped_stone_index(idx) else str(idx + 1)
                swap_active_here = (
                    self.core.is_swapped_stone_index(idx)
                    and history[last_idx].kind == MoveKind.SWAP
                )
                if idx == last_idx or swap_active_here:
                    colr = TEXT_ON_DARK
                else:
                    base = RED if mv.side == Side.RED else BLUE
                    colr = lerp_rgb(base, TEXT_ON_DARK, number_tint)
                self.text.board.blit_center(txt, colr, cx, cy)
            return

        mv = history[-1]
        coords = self.core.move_coords(mv)
        if coords is None and mv.kind == MoveKind.SWAP:
            coords = self.core.move_coords(history[0])
        if coords is None:
            return
        ax, ay = coords[0] - 1, coords[1] - 1
        cx, cy = self.center(ax, ay)
        dot_r = max(2, int(self.layout.r * 0.18))
        pygame.draw.circle(self.screen, BOARD_EMPTY, (int(cx), int(cy)), dot_r, 0)

    def draw_borders(self) -> None:
        thickness = 4
        for border in self.projection.border_sides():
            color = RED if border.side == Side.RED else BLUE
            for i in range(self.board.n):
                ax, ay = border.edge_cell(i, self.board.n)
                for c1, c2 in border.corner_pairs:
                    pygame.draw.line(
                        self.screen,
                        color,
                        self.corner(ax, ay, c1),
                        self.corner(ax, ay, c2),
                        thickness,
                    )

    def draw_side_coords(self) -> None:
        col_color = TEXT_MUTED
        row_color = TEXT_MUTED
        for col in range(1, self.board.n + 1):
            ax, ay = self.projection.col_label_anchor(col)
            cx, cy = self.center(ax, ay)
            txt = col_to_human_letters(col)
            self.text.board.blit_center(txt, col_color, cx, cy)
        for row in range(1, self.board.n + 1):
            ax, ay = self.projection.row_label_anchor(row)
            cx, cy = self.center(ax, ay)
            txt = str(row)
            self.text.board.blit_center(txt, row_color, cx, cy)

    def draw_ghost_cell(self, cell: Tuple[int, int], side: Side) -> None:
        ax, ay = cell[0] - 1, cell[1] - 1
        pts = self.poly(ax, ay)
        base = RED if side == Side.RED else BLUE
        ghost = lerp_rgb(base, BOARD_EMPTY, 0.5)
        pygame.draw.polygon(self.screen, ghost, pts, 0)
        pygame.draw.polygon(self.screen, LINE, pts, 1)

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
            self.text.board.blit_center(str(idx + 1), TEXT_ON_DARK, cx, cy)
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
                    colr = TEXT_ON_DARK if self.board.get(col, row) >= 0 else TEXT_ON_LIGHT
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
                self.text.board.blit_center(pr, TEXT_ON_LIGHT, cx, cy)
                continue

            wr = fmt_wr_or_elo(r.winrate, ui.prefs.show_elo)
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
            self.text.board.blit_line(wr, TEXT_ON_LIGHT, x1, y1)
            top2 = top + rect1.height + gap
            x2 = cx - rect2.width / 2
            y2 = top2 - self.text.board.line_ref_y + rect2.y
            self.text.board.blit_line(vv, TEXT_ON_LIGHT, x2, y2)

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
        moves: Tuple[Move, ...],
        prefix_keys: Tuple[bytes, ...],
        cursor_ply: int,
        show_elo: bool,
    ) -> None:
        pygame.draw.rect(self.screen, SURFACE_PANEL, rect)
        pygame.draw.rect(self.screen, LINE, rect, 1)

        n_moves = len(moves)
        if rect.width <= 1 or rect.height <= 1:
            return

        def best_reply_winrate(key: bytes) -> Optional[float]:
            recs = self.app.analysis_cache.get(key)
            # Prefer ordered move analysis; candidate cache entries have no order.
            best = None
            if recs:
                for r in recs:
                    if r.order is None or r.winrate is None:
                        continue
                    if best is None or r.order < best.order:
                        best = r
            if best is not None and best.winrate is not None:
                return best.winrate
            return self.app.root_eval_cache.get(key)

        def x_for_move(m: int) -> int:
            return rect.left + int(round((m - 1) * (rect.width - 1) / denom))

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
            t = clamp01(t)
            return rect.top + int(round((1.0 - t) * (rect.height - 1)))

        mid_y = y_for_plot(0.5)
        pygame.draw.line(self.screen, LINE, (rect.left, mid_y), (rect.right, mid_y), 1)

        if n_moves <= 0:
            return

        values: dict[int, float] = {}
        max_analyzed = 0
        for m, key in enumerate(prefix_keys, start=1):
            side_to_play = self.core.flip_side(moves[m - 1].side)
            best_wr = best_reply_winrate(key)
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
            if val is None and self.core.is_batch_analysis_active():
                for probe in range(min(cursor_ply - 1, n_moves), 0, -1):
                    val = values.get(probe)
                    if val is not None:
                        m = probe
                        break
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

    def get_eval_graph_data(self) -> tuple[tuple[Move, ...], tuple[bytes, ...]]:
        sig = (self.board.n, self.board.rev, tuple(self.core.mainline_tail_moves()))
        if self._eval_graph_sig != sig or self._eval_graph_data is None:
            graph_data = self.core.build_eval_graph_data()
            self._eval_graph_sig = sig
            self._eval_graph_data = (graph_data.moves, graph_data.prefix_keys)
        return self._eval_graph_data

    def get_movelist_view(self):
        sig = self.core.tree.signature()
        if self._movelist_sig != sig or self._movelist_view is None:
            self._movelist_sig = sig
            self._movelist_view = self.core.build_movelist_view()
        return self._movelist_view

    @staticmethod
    def _movelist_start_row(focus_row: int, total_rows: int, max_lines: int) -> int:
        if not max_lines or total_rows <= max_lines:
            return 0
        if focus_row >= total_rows - 1:
            return total_rows - max_lines
        start = focus_row - (max_lines // 2)
        return max(0, min(start, total_rows - max_lines))

    @staticmethod
    def _movelist_left_col(rows, focus_row: int, cursor_ply: int, visible_cols: int) -> int:
        if cursor_ply <= 0 or not (0 <= focus_row < len(rows)):
            return 0
        active_cell = next((cell for cell in rows[focus_row].cells if cell.played), None)
        if active_cell is None:
            return 0

        active_lane = active_cell.column
        lane_end = active_lane + len(active_cell.label)
        max_end = lane_end
        for row in rows:
            for cell in row.cells:
                cell_end = cell.column + len(cell.label)
                if cell_end > max_end:
                    max_end = cell_end
                if cell.column == active_lane and cell_end > lane_end:
                    lane_end = cell_end
        max_left = max(0, max_end - visible_cols)
        return min(
            max(0, lane_end - visible_cols),
            max_left,
        )

    def draw_movelist_panel(self, ui: UiStateLike) -> None:
        x0 = self.layout.board_px_w
        pygame.draw.rect(self.screen, SURFACE_PANEL, pygame.Rect(x0, 0, PANEL_W, self.screen.get_height()))

        pad = 12
        y = 10
        self.blit_segments(x0 + pad, y, [("Moves", TEXT_ON_LIGHT)], use_small=False)
        y += 26

        view = self.get_movelist_view()
        rows = view.rows
        total_rows = len(rows)
        cursor_ply = self.core.current_ply()

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

        start = self._movelist_start_row(view.focus_row, total_rows, max_lines)
        end = min(total_rows, start + max_lines)
        space_w = self.text.movelist.font.get_rect(" ").width
        gutter_w = self.text.movelist.font.get_rect(f"{total_rows}.").width if total_rows else 0
        content_x = x0 + pad + gutter_w + MOVELIST_GUTTER_GAP
        content_right = x0 + PANEL_W - pad
        content_w = max(1, content_right - content_x)
        visible_cols = max(1, content_w // max(1, space_w))
        left_col = self._movelist_left_col(rows, view.focus_row, cursor_ply, visible_cols)
        scroll_px = left_col * space_w
        content_clip_rect = pygame.Rect(x0 + pad, 0, PANEL_W - (2 * pad), io_top)
        prior_clip = self.screen.get_clip()
        self.screen.set_clip(content_clip_rect)

        for row in rows[start:end]:
            ply_label = f"{row.ply}."
            ply_w = self.text.movelist.font.get_rect(ply_label).width
            self.text.movelist.blit_line(ply_label, TEXT_ON_LIGHT, x0 + pad + gutter_w - ply_w - scroll_px, y)
            for cell in row.cells:
                cx = content_x + (cell.column * space_w) - scroll_px
                if not cell.played:
                    color = TEXT_MUTED
                else:
                    color = RED if cell.side == Side.RED else BLUE
                self.text.movelist.blit_line(cell.label, color, cx, y)
            y += line_h
        self.screen.set_clip(prior_clip)

        if ui.show_engine_debug:
            io_rect = pygame.Rect(x0, io_top, PANEL_W, self.screen.get_height() - io_top)
            pygame.draw.rect(self.screen, SURFACE_PANEL, io_rect)
            pygame.draw.rect(self.screen, LINE, io_rect, 1)

            io_y = io_top + 4
            logs = self.core.engine.get_io_log(io_max_lines)
            start_idx = max(0, len(logs) - io_max_lines)
            for direction, msg, count in logs[start_idx:]:
                prefix = ">>" if direction == "out" else "<<"
                if count > 1:
                    line = f"{prefix} {msg} ({count})"
                else:
                    line = f"{prefix} {msg}"
                self.text.io.blit_line(line, TEXT_ON_LIGHT, x0 + pad, io_y)
                io_y += io_line_h
        elif graph_h >= GRAPH_MIN_HEIGHT:
            moves, prefix_keys = self.get_eval_graph_data()
            graph_rect = pygame.Rect(
                x0,
                graph_top,
                PANEL_W,
                self.screen.get_height() - graph_top,
            )
            self.draw_eval_graph(
                graph_rect,
                moves,
                prefix_keys,
                cursor_ply,
                ui.prefs.show_elo,
            )

        pygame.draw.line(self.screen, LINE, (x0, 0), (x0, self.screen.get_height()), 1)

    def _hud_candidate_key(self, ui: UiStateLike) -> Optional[Tuple[int, int]]:
        cand_key = (
            self.app.candidate_state.run.key
            if self.app.candidate_state.run is not None
            else ui.last_cand_display
        )
        if cand_key is not None and cand_key not in self.app.candidate_state.candidates:
            cand_key = None
        if cand_key is None:
            next_keys = self.core.sorted_candidates_by_visits()
            cand_key = next_keys[0] if next_keys else None
        return cand_key

    def _hud_best_analysis(self) -> Optional[AnalysisMove]:
        display: Optional[AnalysisMove] = None
        for r in self.core.get_active_analysis():
            if r.col is None or r.row is None:
                continue
            if r.order is None:
                continue
            if not self.board.is_empty(r.col, r.row):
                continue
            if display is None or r.order < display.order:
                display = r
        return display

    def _hud_parts(self, ui: UiStateLike) -> List[Tuple[str, Tuple[int, int, int]]]:
        turn_side = self.core.current_side()
        turn_color = RED if turn_side == Side.RED else BLUE
        turn_name = "Red" if turn_side == Side.RED else "Blue"
        analysis_txt = "ON" if self.app.analysis_enabled else "OFF"
        analysis_color = TEXT_ON_LIGHT if self.app.analysis_enabled else TEXT_MUTED
        parts: List[Tuple[str, Tuple[int, int, int]]] = [
            ("Size: ", TEXT_ON_LIGHT),
            (f"{self.board.n}", TEXT_ON_LIGHT),
        ]
        if self.app.pending_size != self.board.n:
            parts += [
                ("  (pending ", TEXT_ON_LIGHT),
                (f"{self.app.pending_size}", TEXT_ON_LIGHT),
                (")", TEXT_ON_LIGHT),
            ]
        parts += [
            ("   |   Turn: ", TEXT_ON_LIGHT),
            (turn_name, turn_color),
            ("   |   ", TEXT_ON_LIGHT),
            ("Analysis: ", analysis_color),
            (analysis_txt, analysis_color),
        ]
        if self.app.candidate_state.candidates:
            cand_key = self._hud_candidate_key(ui)
            parts += [("   |   ", TEXT_ON_LIGHT), ("Cand: ", TEXT_ON_LIGHT)]
            if cand_key is not None:
                parts += [(coord_to_human(*cand_key), turn_color)]
        else:
            display = self._hud_best_analysis()
            if display is not None:
                best_label = "Batch: " if self.core.is_batch_analysis_active() else "Best: "
                parts += [("   |   ", TEXT_ON_LIGHT), (best_label, TEXT_ON_LIGHT)]
                coord = coord_to_human(display.col, display.row)
                parts += [(coord, turn_color)]
                wr = fmt_wr_or_elo(display.winrate, ui.prefs.show_elo)
                if wr:
                    if ui.prefs.show_elo:
                        parts += [(" ", TEXT_ON_LIGHT), (wr, TEXT_ON_LIGHT)]
                    else:
                        parts += [(" ", TEXT_ON_LIGHT), (f"{wr}%", TEXT_ON_LIGHT)]
                vv = fmt_visits(display.visits)
                if vv:
                    parts += [(" ", TEXT_ON_LIGHT), (f"({vv})", TEXT_ON_LIGHT)]
        return parts

    def draw_hud(self, ui: UiStateLike) -> None:
        pygame.draw.rect(self.screen, SURFACE_BG, pygame.Rect(0, 0, self.screen.get_width(), HUD_H))
        parts = self._hud_parts(ui)
        self.blit_segments(12, 10, parts, use_small=False)

        help_line = "space:analysis • ,:play best • +/-/enter:size • ?:help"
        self.text.hud_small.blit_line(help_line, TEXT_ON_LIGHT, 12, 32)

    def draw_top_right_status(self, ui: UiStateLike) -> None:
        awrn = f"{self.app.analysis_wide_root_noise:.2f}".rstrip("0").rstrip(".")
        awrn_text = "AWRN –" if self.app.candidate_state.candidates else f"AWRN {awrn}"
        awrn_w = self.fonts.hud_small.get_rect(awrn_text).width
        awrn_x = max(12, self.layout.board_px_w - awrn_w - 12)
        self.text.hud_small.blit_line(awrn_text, TEXT_MUTED, awrn_x, 10)
        vps_suffix = " visits/s"
        engine_tag = ""
        if ui.has_multiple_engines and ui.current_engine_name:
            engine_name = ui.current_engine_name
            if len(engine_name) > 12:
                engine_name = engine_name[:11] + "…"
            engine_tag = f"[{engine_name}] "
        if ui.speed_vps is not None and ui.speed_vps > 0:
            vps_text = f"{engine_tag}{fmt_visits(int(ui.speed_vps))}{vps_suffix}"
        else:
            vps_text = f"{engine_tag}–{vps_suffix}"
        vps_w = self.fonts.hud_small.get_rect(vps_text).width
        vps_x = max(12, self.layout.board_px_w - vps_w - 12)
        vps_y = 10 + self.text.line_hud_small + 2
        self.text.hud_small.blit_line(vps_text, TEXT_MUTED, vps_x, vps_y)

    def draw_help_overlay(self) -> None:
        rows = [
            ("title", "Help (? to hide)"),
            ("header", "Navigation"),
            ("item", "p:prev   n:next   f:first   l:last   scroll:prev/next"),
            ("item", "ctrl+p:prev 10   ctrl+n:next 10   left/right:branch"),
            ("header", "Moves / edit"),
            ("item", ",:play best/PV   shift+p:pass   s:swap"),
            ("item", "left-drag:move stone   del:delete tail"),
            ("item", "ctrl+z:undo   ctrl+y:redo   shift+n:new"),
            ("header", "Display / analysis"),
            ("item", "space:analysis   t:priors   c:coords   m:moves   e:elo"),
            ("item", "[/]:set analysisWideRootNoise   shift+c:clear cache"),
            ("item", "d:engine debug   shift+e:cycle engine"),
            ("header", "Candidates / batch"),
            ("item", "right-click:toggle cand   right-drag:toggle cands"),
            ("item", "shift+x:clear cands   shift+b:fast batch   ctrl+shift+b:batch"),
            ("header", "Clipboard / misc"),
            ("item", "ctrl+v:load   ctrl+c:hexworld   ctrl+shift+c:hexata"),
            ("item", "+/-:pending size   enter:apply size   ctrl+s:screenshot"),
            ("item", "esc:quit"),
        ]
        pad = 8
        gap = 2
        surfs = [self.fonts.hud_small.render(text, fgcolor=TEXT_ON_LIGHT)[0] for _kind, text in rows]
        w = max(s.get_width() for s in surfs)
        line_h = self.text.line_hud_small
        header_gap = 6
        header_count = sum(1 for kind, _text in rows if kind == "header")
        h = line_h * len(rows) + gap * max(0, len(rows) - 1) + header_gap * header_count
        x = 12
        y = HUD_H + 12
        rect = pygame.Rect(x - pad, y - pad, w + pad * 2, h + pad * 2)
        pygame.draw.rect(self.screen, SURFACE_PANEL, rect)
        pygame.draw.rect(self.screen, LINE, rect, 1)
        for kind, text in rows:
            if kind == "header":
                y += header_gap
            color = HELP_HEADER if kind == "header" else TEXT_ON_LIGHT
            self.text.hud_small.blit_line(text, color, x, y)
            y += line_h + gap

    def draw_frame(
        self,
        ui: UiStateLike,
        hover_cell: Optional[Tuple[int, int]],
        show_prior: bool,
        show_coords: bool,
        top_cell: Optional[Tuple[int, int]],
        top_visits: int,
    ) -> None:
        self.screen.fill(SURFACE_BG)
        self.draw_hud(ui)
        pv = self.get_display_pv(hover_cell)
        show_pv = self.should_show_pv(pv)
        pv_cells = set(pv[1:]) if show_pv else None
        drag_target = None
        drag_side = None
        drag_source = None
        history = self.core.applied_history()
        if ui.drag_move and ui.drag_move_from is not None and ui.drag_move_idx is not None:
            if 0 <= ui.drag_move_idx < len(history):
                drag_side = history[ui.drag_move_idx].side
                drag_source = ui.drag_move_from
                if (
                    hover_cell is not None
                    and hover_cell != ui.drag_move_from
                    and self.board.is_empty(*hover_cell)
                ):
                    drag_target = hover_cell
        self.draw_grid_and_stones(top_cell, top_visits, show_prior)
        if show_pv and pv is not None:
            self.draw_pv_ghosts(pv, self.core.current_side())
        if drag_side is not None:
            if drag_source is not None:
                self.draw_ghost_cell(drag_source, drag_side)
            if drag_target is not None:
                self.draw_ghost_cell(drag_target, drag_side)
        self.draw_next_future_outline()
        if drag_target is None and hover_cell is not None and self.board.is_empty(*hover_cell) and not show_pv:
            ax, ay = hover_cell[0] - 1, hover_cell[1] - 1
            cx, cy = self.center(ax, ay)
            dot_r = max(2, int(self.layout.r * 0.12))
            pygame.draw.circle(self.screen, LINE, (int(cx), int(cy)), dot_r, 0)
        self.draw_borders()
        self.draw_side_coords()
        self.draw_analysis_text(ui, show_prior, show_coords, suppress_cells=pv_cells)
        if not show_coords and show_pv and pv is not None:
            self.draw_pv_numbers(pv, self.core.current_side())
        if not show_coords:
            self.draw_move_numbers(ui.prefs.show_move_numbers)
        self.draw_movelist_panel(ui)
        self.draw_top_right_status(ui)
        if ui.show_help:
            self.draw_help_overlay()
