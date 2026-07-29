"""
evolve/recovery.py -- rebuild run outputs from pareto_front_gen*.pkl checkpoints.

Evolution.evolve() accumulates `statistics`, `fronts` and `best` in memory and
returns them for save_statistics() to pickle at the very end. Anything that
stops the generation loop from running leaves those empty on disk even though
the per-generation checkpoints are intact. 
"""

import glob
import logging
import os
import pickle
import re

logger = logging.getLogger("recovery")

# keys in each statistics record that are not objectives
_SEQ_KEY = "sequences"


def gen_files(run_dir):
    """[(gen_number, path)] for pareto_front_gen*.pkl, ordered numerically."""
    keyed = []
    for path in glob.glob(os.path.join(run_dir, "pareto_front_gen*.pkl")):
        m = re.search(r"gen(\d+)\.pkl$", os.path.basename(path))
        if m:
            keyed.append((int(m.group(1)), path))
    return sorted(keyed)


def load_fronts(run_dir):
    """[(gen, front)] loaded from the checkpoints, skipping unreadable files."""
    out = []
    for gen, path in gen_files(run_dir):
        try:
            with open(path, "rb") as fh:
                out.append((gen, pickle.load(fh)))
        except Exception as exc:
            logger.warning("skipping unreadable %s: %s", os.path.basename(path), exc)
    return out


def statistics_from_fronts(gen_fronts):
    """Per-generation records in the shape save_statistics() would have written.

    Objective keys are read from the individuals, not hardcoded, so extra
    objectives (pocket_shape etc.) carry through.
    """
    stats = []
    for _, front in gen_fronts:
        rec = {_SEQ_KEY: []}
        if not front:
            stats.append(rec)
            continue
        keys = list(front[0].fitnesses.keys())
        for k in keys:
            rec[k] = []
        for ind in front:
            rec[_SEQ_KEY].append(ind.sequence)
            for k in keys:
                rec[k].append(float(ind.fitnesses.get(k, 0.0)))
        stats.append(rec)
    return stats


def _is_empty(value):
    return value is None or (hasattr(value, "__len__") and len(value) == 0)


def needs_recovery(run_dir, evo_out, n_generations=None):
    """True if any output is empty OR covers fewer generations than exist on disk.

    A resumed run (preemption, requeue, walltime) only accumulates the
    generations that process actually ran: preempted at gen 40 and resubmitted,
    evolve() returns 20 entries covering gens 41..60, not 60. Those are
    non-empty, so an emptiness check alone lets a silently truncated file
    through -- worse than an empty one, because it looks valid.
    """
    if not evo_out:
        return True
    if any(not v for v in evo_out.values()):
        return True
    gens = [g for g, _ in gen_files(run_dir)]
    expected = n_generations or (max(gens) if gens else 0)
    return len(evo_out.get("fronts") or []) < expected


def recover_run(run_dir, evo_out=None, checkpoint_freq=1, n_generations=None):
    """Fill in empty *or partial* entries of `evo_out` from the checkpoints.

    Returns (evo_out, notes). `evo_out` may be None, in which case all three
    keys are built from scratch -- that is the path used by the standalone
    recovery of a finished run. `notes` is a list of human-readable strings
    describing what was rebuilt and how faithful it is.

    For a resumed run the in-memory entries cover the trailing generations that
    this process ran. Those are kept as-is (they are full-fidelity), and the
    leading generations are reconstructed from checkpoints and prepended. The
    resulting statistics therefore has MIXED fidelity per generation; the notes
    say where the seam is.
    """
    notes = []
    out = dict(evo_out) if evo_out else {}

    gen_fronts = load_fronts(run_dir)
    if not gen_fronts:
        notes.append("no pareto_front_gen*.pkl checkpoints; nothing to recover")
        return out, notes

    gens = [g for g, _ in gen_fronts]
    notes.append("found %d checkpoints, generations %d..%d"
                 % (len(gens), min(gens), max(gens)))
    missing = sorted(set(range(min(gens), max(gens) + 1)) - set(gens))
    if missing:
        notes.append("WARNING gaps at generations %s" % missing)

    if _is_empty(out.get("best")):
        out["best"] = gen_fronts[-1][1]
        notes.append("best: rebuilt from pareto_front_gen%d.pkl (exact, %d individuals)"
                     % (gen_fronts[-1][0], len(out["best"])))

    if _is_empty(out.get("fronts")):
        out["fronts"] = [f for _, f in gen_fronts]
        if checkpoint_freq == 1:
            notes.append("fronts: rebuilt from checkpoints (exact, %d generations)"
                         % len(out["fronts"]))
        else:
            notes.append("fronts: rebuilt from checkpoints but checkpoint_freq=%d, "
                         "so only every %dth generation is present (%d of %d)"
                         % (checkpoint_freq, checkpoint_freq, len(out["fronts"]), max(gens)))

    if _is_empty(out.get("statistics")):
        out["statistics"] = statistics_from_fronts(gen_fronts)
        rows = sum(len(r.get(_SEQ_KEY, [])) for r in out["statistics"])
        uniq = len({str(s) for r in out["statistics"] for s in r.get(_SEQ_KEY, []) if s})
        notes.append("statistics: rebuilt from checkpoints (APPROXIMATE -- Pareto "
                     "fronts only, not population history; %d rows, %d unique sequences)"
                     % (rows, uniq))
        if rows and uniq < rows:
            notes.append("  %d of %d rows are repeat appearances of a surviving front "
                         "member (%.0f%% distinct)" % (rows - uniq, rows, 100.0 * uniq / rows))

    # --- partial outputs from a resumed run: prepend the missing leading gens ---
    expected = n_generations or max(gens)
    n_mem = len(out.get("fronts") or [])
    if 0 < n_mem < expected:
        # this process ran the trailing generations; earlier ones exist only as
        # checkpoints, from whichever attempt was interrupted
        n_missing = expected - n_mem
        head = [(g, f) for g, f in gen_fronts if g <= n_missing]
        if len(head) < n_missing:
            notes.append("WARNING resumed run is missing generations 1..%d but only "
                         "%d checkpoints cover them; result will be short"
                         % (n_missing, len(head)))
        notes.append("resumed run: in-memory outputs cover the last %d of %d "
                     "generations; prepending gens 1..%d from checkpoints"
                     % (n_mem, expected, n_missing))
        out["fronts"] = [f for _, f in head] + list(out["fronts"])
        if out.get("statistics"):
            head_stats = statistics_from_fronts(head)
            out["statistics"] = head_stats + list(out["statistics"])
            notes.append("  MIXED FIDELITY statistics: gens 1..%d are Pareto fronts "
                         "only (~%d rows/gen), gens %d..%d are full population "
                         "(~%d rows/gen)"
                         % (n_missing,
                            len(head_stats[0].get(_SEQ_KEY, [])) if head_stats else 0,
                            n_missing + 1, expected,
                            len(out["statistics"][-1].get(_SEQ_KEY, []))))

    return out, notes