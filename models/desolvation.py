# models/desolvation.py
"""
Desolvation penalty for buried charge.

This is the counterbalancing term for `electric_field`. Optimizing
-dMu . F alone is unbounded: the cheapest way to strengthen the field at
the active site is to stack formal charges near it, which costs nothing in
a fixed-charge Coulomb sum but is heavily penalized in a real protein.
Directed evolution of RA95 did the opposite -- Hunt et al. report that the
distal mutations in RA95.5-8F introduce a net surface charge change of -4,
replacing three arginines with neutral residues.

Physics
-------
Transferring a charge from water into a low-dielectric protein interior
costs Born energy:

    dG_desolv = (q^2 / 2R) (1/eps_p - 1/eps_w)

With q = 1 e, R = 2.5 A, eps_p = 4, eps_w = 80 this is ~15 kcal/mol for a
fully buried, uncompensated charge, which is the literature range for
burying an unpaired ionizable group. `born_scale` is therefore anchored in
theory rather than fitted; change eps_protein / born_radius instead of
tuning the prefactor blindly.

Burial is measured as a neighbour count around each charged group,
normalized between the exposed and buried extremes observed in a reference
structure (calibrate with `calibrate()` below -- for 4PA8 these are ~65 and
~181 heavy atoms within 10 A).

Salt bridges are credited: a charged group with compensating opposite
charge nearby is much cheaper to bury than an isolated one, so the penalty
scales with the LOCAL NET charge rather than with each charge separately.

Fitness is the negative total cost in kcal/mol, so higher is better and it
is on the same energy scale as `electric_field`. Use them as two separate
objectives and let the Pareto front expose the trade-off, rather than
folding them together with an arbitrary weight.

Limitations: no explicit solvent, no pKa shifts, and burial is a neighbour
count rather than a real SASA. It is a design filter, not a free energy.
"""

import os
import sys
import traceback
import numpy as np
from typing import Dict

from core.interfaces import BaseModel
from core.registry import register_model
from evolve.individual import Individual


# charged groups: residue -> (atoms defining the group centroid, formal charge)
CHARGED = {
    "ASP": (("OD1", "OD2"), -1.0),
    "GLU": (("OE1", "OE2"), -1.0),
    "LYS": (("NZ",), +1.0),
    "ARG": (("NH1", "NH2", "NE"), +1.0),
}
BACKBONE = {"N", "CA", "C", "O"}


def parse_structure(pdb_path, include_hetatm=False):
    """Returns (all heavy-atom coords, [(chain,resi,resname,centroid,charge), ...])."""
    coords, groups = [], {}
    for line in open(pdb_path):
        rec = line[:6]
        if rec not in ("ATOM  ", "HETATM"):
            continue
        if rec == "HETATM" and not include_hetatm:
            continue
        if line[16] not in (" ", "A"):
            continue
        resn = line[17:20].strip()
        if resn in ("HOH", "WAT", "TIP3"):
            continue
        atom = line[12:16].strip()
        if (line[76:78].strip().upper() or atom[:1]).upper() == "H":
            continue
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        coords.append(xyz)
        spec = CHARGED.get(resn)
        if spec and atom in spec[0]:
            key = (line[21], line[22:27].strip())
            groups.setdefault(key, [resn, [], spec[1]])[1].append(xyz)

    out = []
    for (ch, ri), (resn, pts, q) in groups.items():
        out.append((ch, ri, resn, np.mean(pts, axis=0), q))
    return np.asarray(coords, dtype=np.float64).reshape(-1, 3), out


@register_model("desolvation")
class Desolvation(BaseModel):

    def __init__(self):
        pass

    def setup(self, config: Dict, device: str = "cpu") -> None:
        self.config = config
        self.device = device
        mc = self.model_config = self.config.models.desolvation

        # Born prefactor, in kcal/mol for a fully buried unit charge
        eps_p = float(mc.get("eps_protein", 4.0))
        eps_w = float(mc.get("eps_water", 80.0))
        R = float(mc.get("born_radius", 2.5))
        self.born_scale = float(mc.get(
            "born_scale", 332.0 / (2.0 * R) * (1.0 / eps_p - 1.0 / eps_w)))

        # Burial anchors from protein density (rho ~0.061 heavy atoms / A^3),
        # NOT from percentiles of the observed distribution. A group on a flat
        # surface sees roughly half a sphere of protein; a fully buried group
        # sees a whole one. Percentile anchors would force the median charged
        # group to look half-buried and penalize ordinary surface residues.
        self.r_burial = float(mc.get("r_burial", 10.0))
        _V = 4.0 / 3.0 * np.pi * self.r_burial ** 3
        _full = float(mc.get("density", 0.061)) * _V
        self.n_exposed = float(mc.get("n_exposed", 0.5 * _full))
        self.n_buried = float(mc.get("n_buried", _full))
        self.r_pair = float(mc.get("r_pair", 6.0))
        self.include_hetatm = bool(mc.get("include_hetatm", False))
        self.verbose = bool(mc.get("verbose", False))

        outputs = self.config.general.outputs
        self.output_dir = os.path.join(outputs, "desolvation")
        os.makedirs(self.output_dir, exist_ok=True)

        ref = self.config.input.pdb
        c = self.get_components(ref)
        self.reference_cost = c["cost"]
        if self.verbose:
            print(f"[desolvation] born_scale {self.born_scale:.1f} kcal/mol "
                  f"(eps_p={eps_p}, R={R} A)", flush=True)
            print(f"[desolvation] reference {os.path.basename(ref)}: "
                  f"{c['n_charged']} charged groups, net {c['net_charge']:+.0f}, "
                  f"cost {c['cost']:.1f} kcal/mol", flush=True)

    # ------------------------------------------------------------------
    def get_components(self, pdb_path):
        coords, groups = parse_structure(pdb_path, self.include_hetatm)
        if len(groups) == 0:
            return {"cost": 0.0, "n_charged": 0, "net_charge": 0.0,
                    "mean_burial": 0.0, "fitness": 0.0}

        cen = np.array([g[3] for g in groups])
        q = np.array([g[4] for g in groups])

        # burial: neighbour count around each charged group
        d = np.linalg.norm(coords[:, None, :] - cen[None, :, :], axis=2)
        n_nb = ((d < self.r_burial) & (d > 1e-6)).sum(0).astype(float)
        burial = np.clip((n_nb - self.n_exposed) /
                         max(self.n_buried - self.n_exposed, 1e-6), 0.0, 1.0)

        # local net charge: salt bridges are cheap, isolated charges are not
        dg = np.linalg.norm(cen[:, None, :] - cen[None, :, :], axis=2)
        near = dg < self.r_pair
        q_local = (near * q[None, :]).sum(1)          # includes self

        # Born cost, driven by the uncompensated part of each charge
        cost = float((self.born_scale * burial * q_local ** 2).sum())

        return {"cost": cost,
                "n_charged": len(groups),
                "net_charge": float(q.sum()),
                "mean_burial": float(burial.mean()),
                "fitness": -cost}

    # ------------------------------------------------------------------
    def score(self, individual: Individual):
        pdb_path = individual.get_name()
        gen, index = individual.get_gen(), individual.get_index()
        try:
            c = self.get_components(pdb_path)
            value = c["fitness"]
            if self.verbose:
                print(f"[desolvation] gen {gen} idx {index}  "
                      f"n={c['n_charged']}  net={c['net_charge']:+.0f}  "
                      f"burial={c['mean_burial']:.2f}  "
                      f"cost={c['cost']:.1f}  fitness={value:+.1f}", flush=True)
        except Exception:
            print(f"[desolvation] scoring failed for {pdb_path}", file=sys.stderr)
            traceback.print_exc()
            value = -1000.0
        individual.add_fitness({"desolvation": float(value)})


def calibrate(pdb_path, r_burial=10.0, density=0.061):
    """
    Diagnostic only. The burial anchors are derived from protein density and
    should NOT normally be set by hand; this just reports how a structure sits
    relative to them.
    """
    coords, groups = parse_structure(pdb_path)
    cen = np.array([g[3] for g in groups])
    d = np.linalg.norm(coords[:, None, :] - cen[None, :, :], axis=2)
    n = ((d < r_burial) & (d > 1e-6)).sum(0)
    full = density * 4.0 / 3.0 * np.pi * r_burial ** 3
    half = 0.5 * full
    burial = np.clip((n - half) / (full - half), 0.0, 1.0)
    print(f"{pdb_path}: {len(groups)} charged groups")
    print(f"  neighbours within {r_burial} A: "
          f"min {n.min()}  median {np.median(n):.0f}  max {n.max()}")
    print(f"  density anchors: n_exposed {half:.0f}  n_buried {full:.0f}")
    print(f"  mean burial {burial.mean():.2f}; "
          f"{(burial > 0).sum()}/{len(burial)} groups are buried at all")
    return half, full


if __name__ == "__main__":
    calibrate(sys.argv[1])