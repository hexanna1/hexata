from __future__ import annotations

from collections import deque
import re
import sys
import threading
import subprocess
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from board import GameType, Side, coord_to_human, human_letters_to_col

# -------------------- coords --------------------
# Three coordinate systems:
# 1) Board coords (GUI): "a1" is top-left. col increases right, row increases down.
#    (col,row) are 1-indexed. Examples: a1 -> (1,1), b1 -> (2,1), a2 -> (1,2).
# 2) Hex engine "play" coords: send `play <B|W> (x,y)` with x=2*col+row-2, y=2*row-1.
#    Examples: a1 -> (1,1), b1 -> (3,1), a2 -> (2,3).
#    Y engine coords are direct board coords, so the adapter sends human coords like "a1".
# 3) Hex engine "kata-analyze" tokens (Go-style like "G4"):
#    Letters are Go-style columns skipping I (A=1..H=8,J=9,...) -> value p.
#    Number is 2 * (row counted from bottom of board), so row_from_bottom = num//2.
#    Decode with board size N: row = (N + 1) - row_from_bottom;
#    col = (p - row + 1)//2 (must be integer for a legal cell).
def board_to_engine_xy(col: int, row: int) -> Tuple[int, int]:
    # x = 2*col + row - 2, y = 2*row - 1
    return (2 * col + row - 2, 2 * row - 1)


def board_to_engine_vertex(col: int, row: int, game_type: GameType = GameType.HEX) -> str:
    match game_type:
        case GameType.HEX:
            x, y = board_to_engine_xy(col, row)
            return f"({x},{y})"
        case GameType.Y:
            return coord_to_human(col, row)
    raise AssertionError(f"Unhandled game type: {game_type}")


# -------------------- analysis token -> board (col,row) --------------------
_MOVE_TOKEN_RE = re.compile(r"^\s*([A-Za-z]+)\s*([0-9]+)\s*$")
_GTP_ALPH = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # skip I (Go-style)


def gtp_letters_to_int_skipI(s: str) -> int:
    v = 0
    for ch in s.upper():
        i = _GTP_ALPH.find(ch)
        if i < 0:
            return 0
        v = v * 25 + (i + 1)
    return v


def int_to_gtp_letters_skipI(v: int) -> Optional[str]:
    if v <= 0:
        return None
    out: List[str] = []
    while v:
        v, rem = divmod(v - 1, 25)
        out.append(_GTP_ALPH[rem])
    return "".join(reversed(out))


def _parse_y_analysis_move_token(tok: str, board_n: int) -> Optional[Tuple[int, int]]:
    m = _MOVE_TOKEN_RE.match(tok.strip())
    if not m:
        return None
    col = human_letters_to_col(m.group(1))
    row = int(m.group(2))
    if not GameType.Y.in_bounds(board_n, col, row):
        return None
    return (col, row)


def parse_analysis_move_token(
    tok: str,
    board_n: int,
    game_type: GameType | str = GameType.HEX,
) -> Optional[Tuple[int, int]]:
    t = tok.strip()
    if t.lower() in ("pass", "resign"):
        return None
    game_type = GameType.parse(game_type)
    match game_type:
        case GameType.Y:
            return _parse_y_analysis_move_token(t, board_n)
        case GameType.HEX:
            m = _MOVE_TOKEN_RE.match(t)
            if not m:
                return None

            p = gtp_letters_to_int_skipI(m.group(1))
            num = int(m.group(2))
            if p <= 0 or num <= 0 or num % 2:
                return None

            row_from_bottom = num // 2
            row = (board_n + 1) - row_from_bottom
            tmp = p - row + 1
            if tmp % 2:
                return None
            col = tmp // 2
            return (col, row) if game_type.in_bounds(board_n, col, row) else None
    raise AssertionError(f"Unhandled game type: {game_type}")


def to_analysis_token(
    col: int,
    row: int,
    board_n: int,
    game_type: GameType | str = GameType.HEX,
) -> Optional[str]:
    game_type = GameType.parse(game_type)
    if not game_type.in_bounds(board_n, col, row):
        return None
    match game_type:
        case GameType.Y:
            return coord_to_human(col, row)
        case GameType.HEX:
            letters = int_to_gtp_letters_skipI(2 * col + row - 1)
            if letters is None:
                return None
            row_from_bottom = (board_n + 1) - row
            return f"{letters}{2 * row_from_bottom}"
    raise AssertionError(f"Unhandled game type: {game_type}")


# -------------------- parse kata-analyze output (minimal fields) --------------------
_INFO_REC_RE = re.compile(
    r"\binfo\s+move\s+(\S+)\s+(.*?)(?=\binfo\s+move\s+\S+\s+|$)", re.DOTALL
)
_FIELD_RE: Dict[str, re.Pattern] = {}


def _get_field(txt: str, name: str, cast):
    if name not in _FIELD_RE:
        _FIELD_RE[name] = re.compile(rf"\b{name}\s+([^\s]+)")
    m = _FIELD_RE[name].search(txt)
    if not m:
        return None
    try:
        return cast(m.group(1))
    except Exception:
        return None


@dataclass(frozen=True, slots=True)
class AnalysisMove:
    move: str
    order: Optional[int]
    col: Optional[int]
    row: Optional[int]
    winrate: Optional[float]
    visits: Optional[int]
    prior: Optional[float]
    pv: Optional[Tuple[Tuple[int, int], ...]]


@dataclass(frozen=True, slots=True)
class RawNNResult:
    white_win: Optional[float]
    policy_rows: Tuple[Tuple[Optional[float], ...], ...]
    policy_pass: Optional[float]


def parse_kata_analyze_line(
    line: str,
    board_n: int,
    game_type: GameType | str = GameType.HEX,
) -> List[AnalysisMove]:
    if "info move " not in line:
        return []

    out: List[AnalysisMove] = []
    for m in _INFO_REC_RE.finditer(line):
        move, rest = m.group(1), m.group(2)
        pv_pos = rest.find(" pv ")
        pv: Optional[Tuple[Tuple[int, int], ...]] = None
        if pv_pos >= 0:
            pv_str = rest[pv_pos + 4 :].strip()
            rest = rest[:pv_pos]
            if pv_str:
                pv_list: List[Tuple[int, int]] = []
                for tok in pv_str.split():
                    coord = parse_analysis_move_token(
                        tok, board_n=board_n, game_type=game_type
                    )
                    if coord is None:
                        break
                    pv_list.append(coord)
                if pv_list:
                    pv = tuple(pv_list)

        visits = _get_field(rest, "visits", int)
        if visits == 0:
            continue
        winrate = _get_field(rest, "winrate", float)
        prior = _get_field(rest, "prior", float)
        order = _get_field(rest, "order", int)

        coord = parse_analysis_move_token(move, board_n=board_n, game_type=game_type)
        col: Optional[int] = None
        row: Optional[int] = None
        if coord is not None:
            col, row = coord

        out.append(
            AnalysisMove(
                move=move,
                order=order,
                col=col,
                row=row,
                winrate=winrate,
                visits=visits,
                prior=prior,
                pv=pv,
            )
        )

    out.sort(key=lambda r: r.order if r.order is not None else 10**9)
    return out


def _parse_float_maybe_nan(tok: str) -> Optional[float]:
    if tok == "NAN":
        return None
    v = float(tok)
    return None if math.isnan(v) else v


def parse_kata_raw_nn_block(text: str) -> RawNNResult:
    white_win: Optional[float] = None
    policy_pass: Optional[float] = None
    rows: List[Tuple[Optional[float], ...]] = []
    in_policy = False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("="):
            continue
        if line.startswith("whiteWin "):
            white_win = float(line.split()[1])
            continue
        if line == "policy":
            in_policy = True
            continue
        if not in_policy:
            continue
        if line.startswith("policyPass"):
            policy_pass = _parse_float_maybe_nan(line.split()[-1])
            break
        rows.append(tuple(_parse_float_maybe_nan(tok) for tok in line.split()))
    return RawNNResult(white_win=white_win, policy_rows=tuple(rows), policy_pass=policy_pass)


# -------------------- engine plumbing --------------------
def _pump(stream, cb: Optional[Callable[[str], None]] = None) -> None:
    for line in iter(stream.readline, ""):
        if cb:
            cb(line)


def send_line(p: subprocess.Popen, s: str, hook: Optional[Callable[[str], None]] = None) -> None:
    if p.stdin is None:
        return
    try:
        p.stdin.write(s + "\n")
        p.stdin.flush()
        if hook:
            hook(s)
    except Exception:
        pass


@dataclass(slots=True)
class _RawNNCapture:
    lines: List[str]
    started: bool = False
    done: bool = False


class KataHexEngine:
    def __init__(
        self,
        board_size: int,
        *,
        cmd: List[str],
        game_type: GameType = GameType.HEX,
        engine_echo: bool = False,
        suppress_stderr: bool = True,
    ):
        self.board_n = board_size
        self.game_type = game_type
        self._by_move: Dict[str, AnalysisMove] = {}
        self._lock = threading.Lock()
        self._analysis_mute_until_sync = False
        self._analysis_active = False
        self._param_cache: Dict[str, Union[str, float, int]] = {}
        self._io_lock = threading.Lock()
        self._io_log: List[Tuple[str, str, int]] = []
        self._io_max = 200
        self._stderr_lines: deque[str] = deque(maxlen=8)
        self._engine_echo = engine_echo
        self._suppress_stderr = suppress_stderr
        self._reply_lock = threading.Lock()
        self._raw_nn_capture: Optional[_RawNNCapture] = None

        # KataHex uses a nonstandard GTP-ish dialect. The simplest sync that
        # works reliably is to mute analysis until the first "=" response after
        # kata-analyze. We tried more complex schemes (fresh/stale detection,
        # monotonicity gates, cooldowns), but they were brittle and worse in
        # practice. This "= handshake" has been stress-tested and never failed.
        def on_line(line: str):
            if not self._analysis_active:
                return
            if self._analysis_mute_until_sync:
                if line.lstrip().startswith("="):
                    self._analysis_mute_until_sync = False
                return
            recs = parse_kata_analyze_line(
                line, board_n=self.board_n, game_type=self.game_type
            )
            if not recs:
                return
            with self._lock:
                if not self._analysis_active:
                    return
                self._by_move = {r.move: r for r in recs}

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        def on_stdout_line(line: str) -> None:
            self._echo_debug(line)
            self._log_io("in", line)
            if self._capture_raw_nn_line(line):
                return
            on_line(line)

        threading.Thread(
            target=_pump,
            args=(self.proc.stdout, on_stdout_line),
            daemon=True,
        ).start()
        if self.proc.stderr is not None:
            threading.Thread(target=_pump, args=(self.proc.stderr, self._on_stderr_line), daemon=True).start()

        self.set_board_size(board_size)
        self.clear_board()
        self._check_startup_health()

    def set_board_size(self, n: int) -> None:
        self.board_n = n
        self._reset_analysis_sync()
        self.clear_analysis()
        self._send(f"boardsize {n}")

    def clear_board(self) -> None:
        self._reset_analysis_sync()
        self.clear_analysis()
        self._send("clear_board")

    def clear_analysis(self) -> None:
        with self._lock:
            self._analysis_active = False
            self._by_move.clear()

    def clear_cache(self) -> None:
        self._reset_analysis_sync()
        self.clear_analysis()
        self._send("clear_cache")

    def stop_analysis(self) -> None:
        # Mark analysis inactive first so late lines get ignored.
        self._analysis_active = False
        self.cancel_reply_capture()
        self._send("stop")

    def play(self, side: Side, col: Optional[int], row: Optional[int]) -> None:
        # Engine expects "B"/"W"
        eng = "B" if side == Side.RED else "W"
        if col is None or row is None:
            self._send(f"play {eng} pass")
            return
        self._send(f"play {eng} {board_to_engine_vertex(col, row, self.game_type)}")

    def undo(self) -> None:
        self._send("undo")

    def kata_set_param(self, name: str, value: Union[str, float, int]) -> None:
        cached = self._param_cache.get(name)
        if cached == value:
            return
        self._param_cache[name] = value
        self._send(f"kata-set-param {name} {value}")

    @staticmethod
    def _engine_side(side: Side) -> str:
        return "B" if side == Side.RED else "W"

    def start_analysis(
        self,
        side_to_analyze: Side,
        interval_cs: int,
        allow_filters: Sequence[Tuple[Side, Sequence[Tuple[int, int]]]] = (),
    ) -> None:
        parts = ["kata-analyze", self._engine_side(side_to_analyze), str(interval_cs)]
        for side, moves in allow_filters:
            tokens: List[str] = []
            for col, row in moves:
                tok = to_analysis_token(col, row, self.board_n, self.game_type)
                if tok is None:
                    raise ValueError(f"Invalid analysis move: {(col, row)!r}")
                tokens.append(tok)
            if tokens:
                parts.extend(["allow", self._engine_side(side), ",".join(tokens), "1"])
        self._send(" ".join(parts))
        # Activate analysis and mute until we see the response header.
        self._analysis_active = True
        self._analysis_mute_until_sync = True

    def get_analysis(self) -> List[AnalysisMove]:
        with self._lock:
            items = list(self._by_move.values())
        items.sort(key=lambda r: r.order if r.order is not None else 10**9)
        return items

    def start_kata_raw_nn(self, symmetry: int = 0) -> bool:
        # KataHex implicitly stops live analysis when running kata-raw-nn, so callers
        # do not need to send an explicit stop first; reset our local analysis state
        # and switch to the raw-NN reply capture path here.
        self._reset_analysis_sync()
        self.clear_analysis()
        with self._reply_lock:
            cap = self._raw_nn_capture
            if cap is not None and not cap.done:
                return False
            self._raw_nn_capture = _RawNNCapture(lines=[])
        self._send(f"kata-raw-nn {symmetry}")
        return True

    def poll_kata_raw_nn(self) -> Tuple[bool, Optional[RawNNResult]]:
        with self._reply_lock:
            cap = self._raw_nn_capture
            if cap is None:
                return (True, None)
            if not cap.done:
                return (False, None)
            self._raw_nn_capture = None
            block = "".join(cap.lines) if cap.lines else None
        if block is None:
            return (True, None)
        return (True, parse_kata_raw_nn_block(block))

    def _reset_analysis_sync(self) -> None:
        with self._lock:
            self._analysis_mute_until_sync = False
        self._analysis_active = False

    def cancel_reply_capture(self) -> None:
        with self._reply_lock:
            cap = self._raw_nn_capture
            self._raw_nn_capture = None
            if cap is not None:
                cap.done = True

    def _capture_raw_nn_line(self, line: str) -> bool:
        with self._reply_lock:
            cap = self._raw_nn_capture
            if cap is None or cap.done:
                return False
            if not cap.started:
                # Start only on raw-NN's "= symmetry" header. We cannot key off
                # any "=" / "?" line because KataHex has no command IDs and
                # stale replies from earlier commands may appear first.
                if line.lstrip().startswith("= symmetry"):
                    cap.started = True
                    cap.lines.append(line)
                return True
            cap.lines.append(line)
            if line.startswith("policyPass") or (not line.strip()):
                cap.done = True
            return True

    def _send(self, s: str) -> None:
        send_line(self.proc, s, self._log_out)

    @staticmethod
    def _truncate_io(msg: str, n: int = 20) -> str:
        cleaned = msg.replace("\n", " ").strip()
        return cleaned[:n]

    def _echo_debug(self, msg: str) -> None:
        if not self._engine_echo:
            return
        sys.stderr.write(msg)
        sys.stderr.flush()

    def _on_stderr_line(self, line: str) -> None:
        if not self._suppress_stderr:
            self._echo_debug(line)
        stripped = line.strip()
        if stripped:
            self._stderr_lines.append(stripped)

    def _check_startup_health(self) -> None:
        time.sleep(0.2)
        code = self.proc.poll()
        if code is not None:
            detail = self._stderr_lines[-1] if self._stderr_lines else f"exit code {code}"
            raise RuntimeError(f"Engine exited during startup: {detail}")

    def _log_io(self, direction: str, msg: str) -> None:
        short = self._truncate_io(msg)
        with self._io_lock:
            if self._io_log and self._io_log[-1][0] == direction and self._io_log[-1][1] == short:
                d, s, c = self._io_log[-1]
                self._io_log[-1] = (d, s, c + 1)
            else:
                self._io_log.append((direction, short, 1))
            if len(self._io_log) > self._io_max:
                self._io_log = self._io_log[-self._io_max :]

    def _log_out(self, msg: str) -> None:
        self._log_io("out", msg)

    def get_io_log(self, max_items: int = 40) -> List[Tuple[str, str, int]]:
        with self._io_lock:
            if max_items <= 0:
                return []
            return list(self._io_log[-max_items:])

    def close(self) -> None:
        try:
            self.proc.terminate()
        except Exception:
            pass
