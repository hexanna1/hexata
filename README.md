# Hexata GUI

A lightweight, keyboard-first GUI for analyzing Hex with the [KataHex](https://www.hexwiki.net/index.php/KataHex) engine. Built with substantial AI assistance.

![GUI example](ex1.png)

## Main features
- Interactive board with optional move numbers and drag-to-move editing.
- Live engine analysis overlays (winrate/visits, priors, candidates, Elo view, PVs).
- Dedicated candidate search and batch analysis modes.
- Move list panel with undo/redo, navigation ([HexWorld](https://hexworld.org/board/#14c1)-style shortcuts), and eval graph (winrate or Elo).
- Native handling for pass and swap moves.
- Clipboard import/export to HexWorld and Hexata format.
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
- `main.py`: App entry point; creates the board, engine, and GUI.
- `config.ini`: Local engine launch configuration.
- `gui.py`: Pygame app loop, input handling, and UI flow.
- `gui_render.py`: Pygame rendering, layout, HUD, and panels.
- `gui_core.py`: Game state, move history, import/export, and engine coordination.
- `gui_core_analysis.py`: Analysis/candidate-search state and cache management.
- `history_tree.py`: Tree model for move history, branching, and cursor navigation.
- `engine.py`: Engine process wrapper, GTP-ish parsing, and analysis I/O.
- `hexata_format.py`: Parser/serializer for Hexata move-tree format.
- `board.py`: Hex board model, history, and move rules.
- `hexworld.py`: HexWorld import/export parsing utilities.

## Notes on tricky parts
A few implementation details were tricky to get right and are useful background for understanding the design:
- Three coordinate systems are in play: GUI board coordinates, engine play coordinates, and KataHex analyze tokens.
- KataHex uses a nonstandard GTP-ish dialect. The GUI uses a minimal handshake (mute until the first "=" after `kata-analyze`) to keep latency low while avoiding analysis leakage.
- Candidate search isn’t a native KataHex mode. The GUI simulates it by cycling candidates as temporary roots and collecting per-candidate results, but because each position is evaluated from scratch (with only partial internal caching), the switching policy needs to reduce wasted rebuild time and make steady progress per candidate.
  - Regular analysis applies `analysisWideRootNoise` at the root, but candidate mode is comparing what are effectively child positions, so candidate runs force `analysisWideRootNoise` to `0`.
- Swap support was especially tricky. Subtree pruning around moves 1 and 2 means editing either move can make the other collide with later descendants, and those invalid descendants need to be removed without disturbing surviving sibling branches. The inverse problem also appears after transposition: a coordinate that looks duplicated can still be valid because swap moved the opening stone away.
- Drag-edit branch merging was another tricky case. If a dragged move lands on a sibling move that already exists, the two branches must merge recursively, and sibling order becomes the tie-breaker for which continuation stays preferred.

## Limitations
- No buttons or clickable move list.
- Only tested on macOS; other platforms or some HiDPI setups may need tweaks.
- No robust engine error handling or recovery yet, though it hasn’t been an issue in testing.
