# CLI output schema

Current JSON shapes emitted by `python3 main.py cli ...`.

## Common wrapper

Successful CLI records emit:

```json
{
  "hexworld": "https://hexworld.org/board/#14c1,...",
  "ok": true,
  "error": null,
  "<analyze | batch | match>": {},
  "meta": {
    "elapsed_ms": 12
  }
}
```

## `analyze`

Example commands:

```bash
python3 main.py cli analyze 'https://hexworld.org/board/#14c1,a14'
python3 main.py cli analyze 'https://hexworld.org/board/#14c1,a14::e4k4j11' --search-seconds 1
```

Raw-NN:

```json
{
  "analyze": {
    "method": "raw_nn",
    "best": {"move": "g5", "prior": 0.23},
    "root_eval": {"red_winrate": 0.61},
    "moves": [
      {"move": "g5", "rank": 1, "prior": 0.23}
    ]
  }
}
```

Search:

```json
{
  "analyze": {
    "method": "search",
    "best": {"move": "g5", "red_winrate": 0.61, "visits": 120},
    "total_visits": 240,
    "moves": [
      {"move": "g5", "rank": 1, "red_winrate": 0.61, "visits": 120, "prior": 0.2}
    ]
  }
}
```

## `batch`

Example command:

```bash
python3 main.py cli batch 'https://hexworld.org/board/#14c1,a14k4d5d4e4d6c6e3f3c7j5d12'
```

```json
{
  "batch": {
    "plies": [
      {
        "ply": 1,
        "side": "red",
        "played": "a1"
      },
      {
        "ply": 2,
        "side": "blue",
        "played": "swap"
      },
      {
        "ply": 3,
        "side": "red",
        "played": "b1",
        "analyze": {
          "method": "search",
          "best": {"move": "f6", "red_winrate": 0.61, "visits": 120},
          "total_visits": 240,
          "moves": [
            {"move": "f6", "rank": 1, "red_winrate": 0.61, "visits": 120, "prior": 0.2}
          ]
        }
      }
    ]
  }
}
```

## `match`

Example command:

```bash
python3 main.py cli match --engine-a main --engine-b alt --size 14 --openings 'a3,a4,a6,a9,a10,a11,a12,a14,b4,b12,c2,d3,e3,f3,g3,h3,i3,j3,k3,l2'
```

```json
{
  "match": {
    "round": 1,
    "game_index": 1,
    "opening": "a1",
    "red": "main",
    "blue": "alt",
    "winner": "alt",
    "result": "red_resigned",
    "plies": [
      {"ply": 1, "side": "red", "played": "a1"},
      {
        "ply": 2,
        "side": "blue",
        "engine": "alt",
        "visits_temp": 0.5,
        "played": "b1",
        "analyze": {
          "method": "search",
          "best": {"move": "b1", "red_winrate": 0.61, "visits": 120},
          "total_visits": 240,
          "moves": [
            {"move": "b1", "rank": 1, "red_winrate": 0.61, "visits": 120, "prior": 0.2}
          ]
        }
      }
    ]
  }
}
```
