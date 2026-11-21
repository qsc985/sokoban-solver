[![make.fun](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/qsc985/sokoban-solver/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/qsc985/sokoban-solver/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)

# Sokoban Solver

A push-optimal Sokoban solver for the [make.fun bounty](https://make.fun/bounty/sokoban-solver-01-run-20261121) "Build the best Sokoban solver, 21 November 2026".

## How it works

The generator's levels are a hallway (one row, where the player starts) with 1–4 independent rooms off it, 3–4 boxes each, one door each. Boxes never leave their room in an optimal solution, so a level's push-optimum is the sum of its rooms' optima. This solver:

1. Detects the hallway row and each room region (connected floor excluding the hallway).
2. Solves every room push-optimally with A* over pushes: a state is (player, sorted boxes); a successor is one push (the player walks to the square behind a box and pushes it one step). Cost is 1 per push.
3. Uses an admissible heuristic — the min-cost assignment (Hungarian, brute-forced for ≤4 boxes) of boxes to goals with precomputed BFS distances — so A* returns the push-optimum.
4. Prunes dead states with a reverse-push live-cell analysis and a 2×2 freeze check.
5. Stitches the room solutions together with hallway walks (the room search starts from the player's current position, so the walk is part of the solution).

A greedy best-first search is the fallback if A* ever exhausts its work cap. Everything is deterministic (fixed direction order, FIFO tie-breaking, work-based caps only — no clock, no randomness), so two runs produce byte-identical output.

## Practice-set results

All 5 practice instances solved at exactly the known push-optimum P\* (1000 points each, score 1,000,000):

| instance | boxes | rooms | pushes | P\* |
|----------|-------|-------|--------|-----|
| i001 | 3 | 1 | 13 | 13 |
| i002 | 4 | 1 | 16 | 16 |
| i003 | 8 | 2 | 29 | 29 |
| i004 | 12 | 3 | 47 | 47 |
| i005 | 16 | 4 | 60 | 60 |

The whole practice set runs in ~0.5s (60s budget). The house baseline plain-astar 1.0.0 only solves i001 and gives up on the other four (score 200,000).

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads `/input/<id>/level.txt`, writes `/output/<id>/solution.txt` (LURD moves). Node 22, no dependencies, no network.
