# Hexata

A lightweight, keyboard-first GUI and CLI for analyzing Hex with the [KataHex](https://www.hexwiki.net/index.php/KataHex) engine. Built with substantial AI assistance.

![GUI example](ex1.png)

## Main features
- Interactive board with optional move numbers and drag-to-move editing.
- Live engine analysis overlays (winrate/visits, priors, candidates, Elo view, PVs).
- Candidate filtering and batch analysis modes.
- Move list panel with undo/redo, navigation ([HexWorld](https://hexworld.org/board/#14c1)-style shortcuts), and eval graph (winrate or Elo).
- Native handling for pass and swap moves.
- Clipboard import/export to HexWorld and Hexata format, plus flexible move-text import.
- Adjustable board size, analysis noise, and multi-engine support.
- Fast, responsive feel with low engine and UI latency.

## Quick setup
- Install deps: `pip install -r requirements.txt` (pygame only).
- Install KataHex separately (engine binary, config, and model weights): [KataHex 20240812](https://github.com/hzyhhzy/KataGomo_fork/releases/tag/Hex_20240812)
- Edit `config.ini` and set `[engine.main].cmd` to point at your KataHex binary, config, and model.
  - Example format: `cmd = path/to/katahex gtp -config path/to/engine.cfg -model path/to/weights.bin.gz`
- Run: `python3 main.py gui`

## Quick controls
- `?` help
- `space` toggle analysis
- `p/n` (or scroll) prev/next, `f/l` first/last
- `m` move numbers, `c` coords, `t` priors, `e` Elo view
- Right-click or drag to toggle candidates

## Files at a glance
- `board.py`: Hex board model, linear history, and move rules.
- `history_tree.py`: Move-tree model for branching history and cursor navigation.
- `engine.py`: Engine process wrapper, GTP-ish parsing, and analysis I/O.
- `main.py`: Program entry point; loads config and dispatches GUI or CLI mode.
- `cli.py`: CLI for root analysis, candidate analysis, batch line analysis, and engine matches.
- `gui/`: GUI implementation.
  - `app.py`: Pygame app loop, input handling, and UI flow.
  - `core.py`: Game state, move history, import/export, and engine coordination.
  - `analysis.py`: Analysis state, candidate filtering, and cache management.
  - `render.py`: Pygame rendering, layout, HUD, and panels.
- `formats/`: Position format parsing and serialization.
  - `hexworld.py`: HexWorld import/export parsing utilities.
  - `hexata.py`: Parser/serializer for Hexata move-tree format.
  - `flexible_moves.py`: Import-only parser for flexible move text.
- `cli-output-schema.md`: Reference for emitted CLI output shapes.
- `config.ini`: Engine launch configuration.

## Notes on tricky parts
A few implementation details were tricky to get right and are useful background for understanding the design:
- Three coordinate systems are in play: GUI board coordinates, engine play coordinates, and KataHex analyze tokens.
- KataHex uses a nonstandard GTP-ish dialect. The GUI uses a minimal handshake (mute until the first "=" after `kata-analyze`) to keep latency low while avoiding analysis leakage.
- Swap support was especially tricky. Subtree pruning around moves 1 and 2 means editing either move can make the other collide with later descendants, and those invalid descendants need to be removed without disturbing surviving sibling branches. The inverse problem also appears after transposition: a coordinate that looks duplicated can still be valid because swap moved the opening stone away.
- Drag-edit branch merging was another tricky case. If a dragged move lands on a sibling move that already exists, the two branches must merge recursively, and sibling order becomes the tie-breaker for which continuation stays preferred.

## Limitations
- No buttons or clickable move list.
- Only tested on macOS; other platforms or some HiDPI setups may need tweaks.
- No robust engine error handling or recovery yet, though it hasn’t been an issue in testing.
