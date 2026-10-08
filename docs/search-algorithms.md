# SCOuT Search Algorithms

## Overview

All SCOuT search methods do the same outer work:

1. choose a compiler/runtime configuration
2. build the program
3. measure it
4. score the result
5. choose what to try next

They mainly differ in how they generate the next candidate.

## Implemented methods

### Optuna

Code: [src/explore.py](/home/hari/git/SCOuT/src/explore.py)

- General-purpose default
- Supports single-objective and multi-objective runs
- Samples variants, parametric flags, pool flags, and environment settings
- Best choice when you want the most complete implementation

### Wavefront

Code: [src/searchMethods/wavefront_flags.py](/home/hari/git/SCOuT/src/searchMethods/wavefront_flags.py)

- Explores flag combinations layer by layer
- Best for small, interpretable flag-atom search spaces
- Uses the first objective for ranking

### Tabu

Code: [src/searchMethods/tabu_flags.py](/home/hari/git/SCOuT/src/searchMethods/tabu_flags.py)

- Local search with memory to avoid cycling
- Can move across variants, params, pool flags, and environment choices
- Useful when full enumeration is too expensive
- Persists every measurement to its results CSV and snapshots its state after each iteration, so an
  interrupted run can be continued with `--resume`.
- Reuses recorded results and on-disk build artifacts instead of repeating work

### Beam-Tabu

Code: [src/searchMethods/beam_tabu.py](/home/hari/git/SCOuT/src/searchMethods/beam_tabu.py)

- Keeps several promising candidates instead of one current state
- Mainly focused on flag-atom combinations
- Uses a configured beam plus tabu move memory

### Annealing

Code: [src/searchMethods/anneal_flags.py](/home/hari/git/SCOuT/src/searchMethods/anneal_flags.py)

- Local search with probabilistic acceptance of worse moves
- Helps escape local minima
- Useful as a lightweight heuristic alternative

### polyMorph Monte Carlo Tree Search

Code: [src/polyMorph/runner.py](/home/hari/git/SCOuT/src/polyMorph/runner.py)

- Sequential search over Tadashi transformation prefixes
- Re-enumerates legal transformations after each applied transformation
- Learns prefix statistics from completed, pruned, failed, cached, and historical evaluations
- Prunes subtrees whose prefixes are repeatedly worse than baseline
- Detailed description: [docs/polymorph-adaptive-tree-search.md](/home/hari/git/SCOuT/docs/polymorph-adaptive-tree-search.md)

## Practical guidance

- Use Optuna for the most reliable default.
- Use wavefront when you want structured flag-combination exploration.
- Use tabu when you want richer local search over a mixed configuration space.
- Use annealing when you want a simpler stochastic local search.
- Use polyMorph Monte Carlo tree search for Tadashi-based loop transformation search.
- Treat beam-tabu as experimental until its config path is cleaned up.
