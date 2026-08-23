# Optional backbone relax

**`relax` is not a score.** It moves nearby backbone while holding the theozyme still, then the real objectives run on that structure.

Flexibile backbone design can be added to the design process to allow for more natural design of the active site. This is optional and can be added to the design process at two points:

| when | what it does | how you opt in |
|---|---|---|
| **Placement** | Small nudge of host loops so other catalytic CBs can land | on by default (`--sat-mode shift`); `--no-accommodate` to skip |
| **Evolution** | Relax after each mutate, before the other models score | put `relax` in `--models` |

If you use it during evolution, keep it **second** (right after
`seq_model`). Model order is load-bearing: later scores must see the
relaxed PDB.

The theozyme residues stay fixed. `emit_objective: false` in the config
means relax does not appear on the Pareto front or get factored into the fitness scoring.

---

## How to run it (evolution)

Generic:

```bash
python run.py -c configs/your_run.yml \
  --models seq_model relax <your other scores>
```

Leave `relax` out of `--models` for a fixed backbone. Which method moves the backbone is a config detail (`models.relax.backend`); you do not pick it on the command line.

## Files

| file | role |
|---|---|
| [`models/relax.py`](../models/relax.py) | evolution-time relax (not a score) |
| [`theozyme/prepare_placements.py`](../theozyme/prepare_placements.py) | placement-time loop nudge |
