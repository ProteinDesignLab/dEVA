#!/usr/bin/env python3
"""
Derive the probe point and reaction axis for the electric_field objective
from a pair of QM theozyme structures (reactant and transition state).

    python theozyme_axis.py Reactant.xyz TS.xyz                     # inspect
    python theozyme_axis.py Reactant.xyz TS.xyz \
        --pdb design.pdb --map 4:LYS:1083:NZ 14:2K6:2001:C10 \
                               15:2K6:2001:C9 16:2K6:2001:C8 \
        [--delta-mu 1.2 -3.4 0.8]

The breaking bond is found automatically as the bonded heavy-atom pair whose
distance increases most from reactant to TS. With --pdb and --map, the
theozyme is superposed onto the structure (Kabsch) so the axis, probe point,
and optional delta-mu vector are reported in the PDB frame, ready to paste
into the YAML.

--map entries are THEOZYME_INDEX:RESNAME:RESNUM:ATOMNAME, using 0-based
indices into the xyz file.
"""
import sys
import argparse
import numpy as np


def read_xyz(path):
    lines = open(path).read().strip().split("\n")
    n = int(lines[0].split()[0])
    Z, X = [], []
    for ln in lines[2:2 + n]:
        f = ln.split()
        Z.append(int(f[0]))
        X.append([float(v) for v in f[1:4]])
    return np.array(Z), np.array(X), lines[1].strip()


def atom_coord(pdb, resname, resnum, atom):
    for line in open(pdb):
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        if line[16] not in (" ", "A"):
            continue
        if line[22:27].strip() != str(resnum) or line[12:16].strip() != atom:
            continue
        if line[17:20].strip() == resname or line[21] == resname:
            return np.array([float(line[30:38]), float(line[38:46]),
                             float(line[46:54])])
    raise SystemExit(f"atom {resname}:{resnum}:{atom} not found in {pdb}")


def kabsch(P, Q):
    """Rotation+translation taking P onto Q. Returns (R, t, rmsd)."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = Qc - R @ Pc
    rmsd = np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1).mean())
    return R, t, rmsd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reactant")
    ap.add_argument("ts")
    ap.add_argument("--pdb")
    ap.add_argument("--map", nargs="*", default=[])
    ap.add_argument("--bond", nargs=2, type=int, default=None, metavar=("I", "J"),
                    help="force the breaking bond (0-based xyz indices)")
    ap.add_argument("--max-ts-dist", type=float, default=3.0,
                    help="ignore pairs longer than this at the TS: a bond at a "
                         "transition state is stretched, not dissociated")
    ap.add_argument("--delta-mu", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "Z"),
                    help="dMu = mu(TS) - mu(reactant), Debye, theozyme frame")
    a = ap.parse_args()

    SYM = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}
    Z, R, hr = read_xyz(a.reactant)
    _, T, ht = read_xyz(a.ts)
    print(f"reactant : {hr}")
    print(f"TS       : {ht}")
    try:
        Er = float([w for w in hr.split() if w.replace('-', '').replace('.', '').isdigit()][0])
        Et = float([w for w in ht.split() if w.replace('-', '').replace('.', '').isdigit()][0])
        print(f"barrier  : {(Et - Er) * 627.5095:.2f} kcal/mol")
    except Exception:
        pass

    # breaking bond = bonded heavy pair with the largest elongation
    heavy = [i for i in range(len(Z)) if Z[i] > 1]
    cands = []
    for i, ai in enumerate(heavy):
        for aj in heavy[i + 1:]:
            lr = np.linalg.norm(R[ai] - R[aj])
            if lr < 1.9:
                lt = np.linalg.norm(T[ai] - T[aj])
                cands.append((lt - lr, ai, aj, lr, lt))
    cands.sort(reverse=True)
    print("\nbonds elongating most (reactant -> TS):")
    for dl, i, j, lr, lt in cands[:5]:
        flag = "" if lt <= a.max_ts_dist else "   <- dissociated, not a TS bond"
        print(f"  {SYM.get(Z[i],'?')}{i:<3d}-{SYM.get(Z[j],'?')}{j:<3d} "
              f"{lr:.2f} -> {lt:.2f} A  ({dl:+.2f}){flag}")

    if a.bond:
        i0, j0 = a.bond
        print(f"\n=> breaking bond (user-specified): atom {i0} -> atom {j0}")
    else:
        keep = [c for c in cands if c[4] <= a.max_ts_dist]
        if not keep:
            raise SystemExit("no partially-broken bond found; use --bond I J")
        _, i0, j0, _, _ = keep[0]
        print(f"\n=> breaking bond: atom {i0} -> atom {j0}   "
              f"(pairs beyond {a.max_ts_dist} A treated as already dissociated)")

    axis_tz = T[j0] - T[i0]
    axis_tz /= np.linalg.norm(axis_tz)
    probe_tz = 0.5 * (T[i0] + T[j0])
    print(f"   axis  (theozyme frame): {np.round(axis_tz, 4).tolist()}")
    print(f"   probe (theozyme frame): {np.round(probe_tz, 3).tolist()}")

    if not a.pdb or not a.map:
        print("\n(supply --pdb and --map to convert into the structure frame)")
        return

    P, Q, labels = [], [], []
    for m in a.map:
        idx, resn, resi, atom = m.split(":")
        P.append(T[int(idx)])
        Q.append(atom_coord(a.pdb, resn, resi, atom))
        labels.append(f"{resn}:{resi}:{atom}")
    P, Q = np.array(P), np.array(Q)
    Rm, t, rmsd = kabsch(P, Q)
    print(f"\nsuperposition on {len(P)} anchors: RMSD {rmsd:.2f} A")
    if rmsd > 1.0:
        print("  WARNING: high RMSD -- check that the --map correspondence is right")

    axis_pdb = Rm @ axis_tz
    probe_pdb = Rm @ probe_tz + t
    print("\n--- paste into configs/*.yml ---")
    print("  electric_field:")
    print(f"    probe_xyz: [{probe_pdb[0]:.3f}, {probe_pdb[1]:.3f}, {probe_pdb[2]:.3f}]")
    print(f"    axis_xyz:  [{axis_pdb[0]:.4f}, {axis_pdb[1]:.4f}, {axis_pdb[2]:.4f}]")
    if a.delta_mu is not None:
        dmu_pdb = Rm @ np.array(a.delta_mu)
        print(f"    delta_mu:  [{dmu_pdb[0]:.4f}, {dmu_pdb[1]:.4f}, {dmu_pdb[2]:.4f}]   # Debye")
    print("    exclude_residues: []   # residues represented in the QM model")


if __name__ == "__main__":
    main()
