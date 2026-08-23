# models/electric_field.py
"""
Local electric field (LEF) objective for dEVA.

Implements the electrostatic-preorganization score used by Hunt et al.,
J. Am. Chem. Soc. 2025, 147, 30723-30736 (retro-aldolase RA95):

  1. The field exerted by the protein at a point in the active site is
     computed from Coulomb's law over all partial charges (their eq 6):

         F(r0) = SUM_i  (1/4*pi*eps0) * q_i * (r0 - r_i) / |r0 - r_i|^3

  2. The barrier for the chemical step is modulated by that field through
     the field-dependent energy barrier (FDB) expansion (their eq 7).
     This objective uses the dipole term only:

         dE!(F) ≈ dE!(0) - dMu . F

     where dMu is the reactant -> TS dipole difference from QM on the
     theozyme. Higher-order polarizability terms are ignored.

  The fitness reported is the BARRIER REDUCTION in kcal/mol,
  -(dE!(F) - dE!(0)), so higher is better, consistent with dEVA convention.

  If dMu is not supplied, the objective falls back to reporting the field
  projection onto a user-defined reaction axis, in MV/cm. This is a proxy
  for the direction of charge separation and is still a valid objective,
  but it is not an energy.

Following the paper, residues that are part of the QM theozyme model are
excluded from the Coulomb sum (they are already treated quantum
mechanically); set `exclude_residues` accordingly.

IMPORTANT CAVEATS
  * Charges are fixed-point ff14SB values from a built-in table. They are
    adequate for ranking designs but are NOT a substitute for charges
    derived from a real parametrization. For numbers going into a paper,
    dump charges from tleap and pass them via `charge_file`.
  * The paper computes LEFs over MD ensembles of open and closed states.
    This objective scores a single static structure, which is the right
    trade-off inside a genetic algorithm but is an approximation. Treat the
    result as a preorganization proxy.
  * No electronic polarization: a fixed-charge field will systematically
    overestimate |F| in a low-dielectric pocket.
  * Only the dipole term of eq 7 is used. delta_alpha / delta_beta are
    accepted for forward compatibility but ignored (their unit conversion
    is not defined for the MV/cm field used here).
"""

import os
import sys
import traceback
import numpy as np
from typing import Dict

from core.interfaces import BaseModel
from core.registry import register_model
from evolve.individual import Individual


# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
# 1/(4*pi*eps0) with q in e and r in Angstrom, giving F in MV/cm
COULOMB_MV_CM = 1439.9645  # (MV/cm) * A^2 / e
# 1 Debye * 1 MV/cm  ->  kcal/mol
D_MVCM_TO_KCAL = 0.0480080
# 1 atomic unit of field = 5142.21 MV/cm
AU_PER_MVCM = 1.0 / 5142.2067

BACKBONE_ATOMS = {"N", "CA", "C", "O"}


# --------------------------------------------------------------------------
# ff14SB partial charges (neutral-pH forms: ASP/GLU -1, LYS/ARG +1, HIS neutral)
# --------------------------------------------------------------------------
_Q = {
"ALA": {"N":-0.4157,"H":0.2719,"CA":0.0337,"HA":0.0823,"CB":-0.1825,
        "HB1":0.0603,"HB2":0.0603,"HB3":0.0603,"C":0.5973,"O":-0.5679},
"ARG": {"N":-0.3479,"H":0.2747,"CA":-0.2637,"HA":0.1560,"CB":-0.0007,
        "HB2":0.0327,"HB3":0.0327,"CG":0.0390,"HG2":0.0285,"HG3":0.0285,
        "CD":0.0486,"HD2":0.0687,"HD3":0.0687,"NE":-0.5295,"HE":0.3456,
        "CZ":0.8076,"NH1":-0.8627,"HH11":0.4478,"HH12":0.4478,
        "NH2":-0.8627,"HH21":0.4478,"HH22":0.4478,"C":0.7341,"O":-0.5894},
"ASN": {"N":-0.4157,"H":0.2719,"CA":0.0143,"HA":0.1048,"CB":-0.2041,
        "HB2":0.0797,"HB3":0.0797,"CG":0.7130,"OD1":-0.5931,"ND2":-0.9191,
        "HD21":0.4196,"HD22":0.4196,"C":0.5973,"O":-0.5679},
"ASP": {"N":-0.5163,"H":0.2936,"CA":0.0381,"HA":0.0880,"CB":-0.0303,
        "HB2":-0.0122,"HB3":-0.0122,"CG":0.7994,"OD1":-0.8014,"OD2":-0.8014,
        "C":0.5366,"O":-0.5819},
"CYS": {"N":-0.4157,"H":0.2719,"CA":0.0213,"HA":0.1124,"CB":-0.1231,
        "HB2":0.1112,"HB3":0.1112,"SG":-0.3119,"HG":0.1933,
        "C":0.5973,"O":-0.5679},
"GLN": {"N":-0.4157,"H":0.2719,"CA":-0.0031,"HA":0.0850,"CB":-0.0036,
        "HB2":0.0171,"HB3":0.0171,"CG":-0.0645,"HG2":0.0352,"HG3":0.0352,
        "CD":0.6951,"OE1":-0.6086,"NE2":-0.9407,"HE21":0.4251,"HE22":0.4251,
        "C":0.5973,"O":-0.5679},
"GLU": {"N":-0.5163,"H":0.2936,"CA":0.0397,"HA":0.1105,"CB":0.0560,
        "HB2":-0.0173,"HB3":-0.0173,"CG":0.0136,"HG2":-0.0425,"HG3":-0.0425,
        "CD":0.8054,"OE1":-0.8188,"OE2":-0.8188,"C":0.5366,"O":-0.5819},
"GLY": {"N":-0.4157,"H":0.2719,"CA":-0.0252,"HA2":0.0698,"HA3":0.0698,
        "C":0.5973,"O":-0.5679},
"HIS": {"N":-0.4157,"H":0.2719,"CA":-0.0581,"HA":0.1360,"CB":-0.0074,
        "HB2":0.0367,"HB3":0.0367,"CG":0.1868,"ND1":-0.5432,"CD2":-0.2207,
        "HD2":0.1862,"CE1":0.1635,"HE1":0.1435,"NE2":-0.2795,"HE2":0.3339,
        "C":0.5973,"O":-0.5679},
"ILE": {"N":-0.4157,"H":0.2719,"CA":-0.0597,"HA":0.0869,"CB":0.1303,
        "HB":0.0187,"CG1":-0.0430,"HG12":0.0236,"HG13":0.0236,
        "CG2":-0.3204,"HG21":0.0882,"HG22":0.0882,"HG23":0.0882,
        "CD1":-0.0660,"HD11":0.0186,"HD12":0.0186,"HD13":0.0186,
        "C":0.5973,"O":-0.5679},
"LEU": {"N":-0.4157,"H":0.2719,"CA":-0.0518,"HA":0.0922,"CB":-0.1102,
        "HB2":0.0457,"HB3":0.0457,"CG":0.3531,"HG":-0.0361,
        "CD1":-0.4121,"HD11":0.1000,"HD12":0.1000,"HD13":0.1000,
        "CD2":-0.4121,"HD21":0.1000,"HD22":0.1000,"HD23":0.1000,
        "C":0.5973,"O":-0.5679},
"LYS": {"N":-0.3479,"H":0.2747,"CA":-0.2400,"HA":0.1426,"CB":-0.0094,
        "HB2":0.0362,"HB3":0.0362,"CG":0.0187,"HG2":0.0103,"HG3":0.0103,
        "CD":-0.0479,"HD2":0.0621,"HD3":0.0621,"CE":-0.0143,
        "HE2":0.1135,"HE3":0.1135,"NZ":-0.3854,
        "HZ1":0.3400,"HZ2":0.3400,"HZ3":0.3400,"C":0.7341,"O":-0.5894},
"MET": {"N":-0.4157,"H":0.2719,"CA":-0.0237,"HA":0.0880,"CB":0.0342,
        "HB2":0.0241,"HB3":0.0241,"CG":0.0018,"HG2":0.0440,"HG3":0.0440,
        "SD":-0.2737,"CE":-0.0536,"HE1":0.0684,"HE2":0.0684,"HE3":0.0684,
        "C":0.5973,"O":-0.5679},
"PHE": {"N":-0.4157,"H":0.2719,"CA":-0.0024,"HA":0.0978,"CB":-0.0343,
        "HB2":0.0295,"HB3":0.0295,"CG":0.0118,"CD1":-0.1256,"HD1":0.1330,
        "CE1":-0.1704,"HE1":0.1430,"CZ":-0.1072,"HZ":0.1297,
        "CD2":-0.1256,"HD2":0.1330,"CE2":-0.1704,"HE2":0.1430,
        "C":0.5973,"O":-0.5679},
"PRO": {"N":-0.2548,"CD":0.0192,"HD2":0.0391,"HD3":0.0391,"CG":0.0189,
        "HG2":0.0213,"HG3":0.0213,"CB":-0.0070,"HB2":0.0253,"HB3":0.0253,
        "CA":-0.0266,"HA":0.0641,"C":0.5896,"O":-0.5748},
"SER": {"N":-0.4157,"H":0.2719,"CA":-0.0249,"HA":0.0843,"CB":0.2117,
        "HB2":0.0352,"HB3":0.0352,"OG":-0.6546,"HG":0.4275,
        "C":0.5973,"O":-0.5679},
"THR": {"N":-0.4157,"H":0.2719,"CA":-0.0389,"HA":0.1007,"CB":0.3654,
        "HB":0.0043,"CG2":-0.2438,"HG21":0.0642,"HG22":0.0642,"HG23":0.0642,
        "OG1":-0.6761,"HG1":0.4102,"C":0.5973,"O":-0.5679},
"TRP": {"N":-0.4157,"H":0.2719,"CA":-0.0275,"HA":0.1123,"CB":-0.0050,
        "HB2":0.0339,"HB3":0.0339,"CG":-0.1415,"CD1":-0.1638,"HD1":0.2062,
        "NE1":-0.3418,"HE1":0.3412,"CE2":0.1380,"CZ2":-0.2601,"HZ2":0.1572,
        "CH2":-0.1134,"HH2":0.1417,"CZ3":-0.1972,"HZ3":0.1447,
        "CE3":-0.2387,"HE3":0.1700,"CD2":0.1243,"C":0.5973,"O":-0.5679},
"TYR": {"N":-0.4157,"H":0.2719,"CA":-0.0014,"HA":0.0876,"CB":-0.0152,
        "HB2":0.0295,"HB3":0.0295,"CG":-0.0011,"CD1":-0.1906,"HD1":0.1699,
        "CE1":-0.2341,"HE1":0.1656,"CZ":0.3226,"OH":-0.5579,"HH":0.3992,
        "CD2":-0.1906,"HD2":0.1699,"CE2":-0.2341,"HE2":0.1656,
        "C":0.5973,"O":-0.5679},
"VAL": {"N":-0.4157,"H":0.2719,"CA":-0.0875,"HA":0.0969,"CB":0.2985,
        "HB":-0.0297,"CG1":-0.3192,"HG11":0.0791,"HG12":0.0791,"HG13":0.0791,
        "CG2":-0.3192,"HG21":0.0791,"HG22":0.0791,"HG23":0.0791,
        "C":0.5973,"O":-0.5679},
}
# common alternates
_Q["HID"] = _Q["HIE"] = _Q["HIS"]
_Q["CYX"] = _Q["CYS"]
_Q["MSE"] = _Q["MET"]


def _parent_heavy(resname, hname):
    """
    Map a hydrogen name to the heavy atom it is attached to, by greek-letter
    suffix, preferring polar parents. Used to build united-atom charges for
    structures that lack hydrogens.
    """
    table = _Q.get(resname)
    if table is None:
        return None
    core = hname.lstrip("0123456789")
    if not core.startswith("H"):
        return None
    core = core[1:]                       # 'HD21' -> 'D21', 'H' -> ''
    if core == "":                        # backbone amide H
        return "N" if "N" in table else None
    for k in range(len(core), 0, -1):     # 'D21' -> 'D2' -> 'D'
        suf = core[:k]
        for elem in ("N", "O", "S", "C"):  # polar parents win ties
            cand = elem + suf
            if cand in table:
                return cand
    return None


def _build_united(all_atom):
    """Fold each hydrogen's charge into its parent heavy atom."""
    united = {}
    for res, table in all_atom.items():
        merged = {a: q for a, q in table.items() if not a.lstrip("0123456789").startswith("H")}
        for a, q in table.items():
            if a.lstrip("0123456789").startswith("H"):
                p = _parent_heavy(res, a)
                if p is not None and p in merged:
                    merged[p] += q
        united[res] = merged
    return united


_Q_UNITED = _build_united(_Q)


# --------------------------------------------------------------------------
# PDB parsing
# --------------------------------------------------------------------------
# formal net charge per residue, used to detect incomplete side chains
FORMAL = {"ASP": -1.0, "GLU": -1.0, "LYS": 1.0, "ARG": 1.0}


def _normalize_exclude(exclude_keys):
    """Accept 'A210', 'A:210', or '210' and expand to all match forms."""
    out = set()
    for x in exclude_keys:
        s = str(x).strip()
        if not s:
            continue
        out.add(s)
        if ':' in s:
            chain, resi = s.split(':', 1)
            out.add(f'{chain}{resi}')
            out.add(resi)
            out.add(f'{chain}:{resi}')
        else:
            # 'A210' or bare '210'
            i = 0
            while i < len(s) and s[i].isalpha():
                i += 1
            if i and i < len(s) and s[i:].lstrip('-').isdigit():
                chain, resi = s[:i], s[i:]
                out.add(f'{chain}:{resi}')
                out.add(resi)
                out.add(f'{chain}{resi}')
            elif s.lstrip('-').isdigit():
                out.add(s)
    return out


def parse_pdb_charged(pdb_path, charges, exclude_keys=(),
                      include_hetatm=False, skip_water=True,
                      fix_incomplete=True, tol=0.05, warn=None):
    """
    Returns (coords Nx3, charges N, has_hydrogens bool, n_fixed).

    Handles two failure modes that silently corrupt a Coulomb sum:

      * alternate conformations -- only altloc ' ' or 'A' is kept, otherwise
        every atom of a disordered residue is counted twice.
      * incomplete side chains -- a disordered GLU missing its carboxylate
        contributes ~+0.4 e instead of -1 e. With fix_incomplete, each
        residue's charge is renormalized to its formal value by spreading
        the deficit over the atoms that are present, which keeps the
        monopole exact at the cost of a small local dipole distortion.
    """
    exclude = _normalize_exclude(exclude_keys)
    residues = {}   # key -> [resname, [xyz...], [q...]]
    has_h = False

    for line in open(pdb_path):
        rec = line[:6]
        if rec not in ("ATOM  ", "HETATM"):
            continue
        if rec == "HETATM" and not include_hetatm:
            continue
        if line[16] not in (" ", "A"):          # altloc: keep primary only
            continue
        resn = line[17:20].strip()
        if skip_water and resn in ("HOH", "WAT", "TIP3"):
            continue
        chain, resi = line[21], line[22:27].strip()
        if (f"{chain}:{resi}" in exclude or f"{chain}{resi}" in exclude
                or resi in exclude):
            continue
        atom = line[12:16].strip()
        if (line[76:78].strip().upper() or atom[:1]).upper() == "H":
            has_h = True
        table = charges.get(resn)
        if table is None:
            continue
        cq = table.get(atom)
        if cq is None:
            continue
        slot = residues.setdefault((chain, resi), [resn, [], []])
        slot[1].append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        slot[2].append(cq)

    xyz, q, n_fixed = [], [], 0
    for key, (resn, coords, qs) in residues.items():
        qs = np.asarray(qs, dtype=np.float64)
        target = FORMAL.get(resn, 0.0)
        dev = qs.sum() - target
        if fix_incomplete and abs(dev) > tol and len(qs):
            qs = qs - dev / len(qs)
            n_fixed += 1
            if warn is not None:
                warn.append(f"{resn}{key[1]} chain {key[0]}: "
                            f"net {qs.sum() + dev:+.2f} -> {target:+.1f} "
                            f"({len(qs)} atoms; incomplete side chain?)")
        xyz.extend(coords)
        q.extend(qs.tolist())

    return (np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
            np.asarray(q, dtype=np.float64), has_h, n_fixed)


def atom_coord(pdb_path, spec):
    """
    spec: "RESNAME:RESNUM:ATOMNAME" or "CHAIN:RESNUM:ATOMNAME".
    Returns the coordinate of the matching atom.
    """
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"atom spec must be 'RES:NUM:ATOM', got {spec!r}")
    key, num, atom = (p.strip() for p in parts)
    for line in open(pdb_path):
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        if line[22:27].strip() != num or line[12:16].strip() != atom:
            continue
        if line[17:20].strip() == key or line[21] == key:
            return np.array([float(line[30:38]), float(line[38:46]),
                             float(line[46:54])], dtype=np.float64)
    raise ValueError(f"atom {spec!r} not found in {pdb_path}")


# --------------------------------------------------------------------------
# objective
# --------------------------------------------------------------------------
@register_model("electric_field")
class ElectricField(BaseModel):
    """
    Scores electrostatic preorganization of the active site.

    fitness = barrier reduction (kcal/mol) if delta_mu is given,
              otherwise the field projection on the reaction axis (MV/cm).
    Higher is better in both cases.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    @staticmethod
    def _read_xyz(path):
        lines = open(path).read().strip().split("\n")
        n = int(lines[0].split()[0])
        Z, X = [], []
        for ln in lines[2:2 + n]:
            f = ln.split()
            Z.append(int(f[0]))
            X.append([float(v) for v in f[1:4]])
        return np.array(Z), np.array(X, dtype=np.float64)

    @staticmethod
    def _kabsch(P, Q):
        Pc, Qc = P.mean(0), Q.mean(0)
        H = (P - Pc).T @ (Q - Qc)
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        t = Qc - R @ Pc
        rmsd = float(np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1).mean()))
        return R, t, rmsd

    def _from_theozyme(self, mc, ref_pdb):
        """
        Derive probe point, reaction axis and (optionally) delta_mu directly
        from the QM theozyme pair, superposed onto the design scaffold.

        Because dEVA is fixed-backbone and the ligand does not move, these
        are constants for the whole run and are computed once, here.
        """
        Z, Rr = self._read_xyz(mc.theozyme_reactant)
        _, Rt = self._read_xyz(mc.theozyme_ts)

        # breaking bond: largest elongation among pairs still bonded at the TS
        if mc.get("theozyme_bond", None):
            i0, j0 = (int(v) for v in mc.theozyme_bond)
        else:
            max_ts = float(mc.get("theozyme_max_ts_dist", 3.0))
            heavy = [i for i in range(len(Z)) if Z[i] > 1]
            best = None
            for a_i, ai in enumerate(heavy):
                for aj in heavy[a_i + 1:]:
                    lr = float(np.linalg.norm(Rr[ai] - Rr[aj]))
                    if lr >= 1.9:
                        continue
                    lt = float(np.linalg.norm(Rt[ai] - Rt[aj]))
                    if lt > max_ts:          # dissociated fragment, not a TS bond
                        continue
                    if best is None or (lt - lr) > best[0]:
                        best = (lt - lr, ai, aj)
            if best is None:
                raise ValueError("no partially-broken bond found in the theozyme; "
                                 "set models.electric_field.theozyme_bond [i, j]")
            _, i0, j0 = best

        axis_tz = Rt[j0] - Rt[i0]
        axis_tz /= np.linalg.norm(axis_tz)
        probe_tz = 0.5 * (Rt[i0] + Rt[j0])

        # superpose theozyme onto the scaffold via the user's atom mapping
        P, Q = [], []
        for m in mc.theozyme_map:
            idx, resn, resi, atom = str(m).split(":")
            P.append(Rt[int(idx)])
            Q.append(atom_coord(ref_pdb, f"{resn}:{resi}:{atom}"))
        Rm, t, rmsd = self._kabsch(np.array(P), np.array(Q))
        if rmsd > float(mc.get("theozyme_max_rmsd", 1.5)):
            raise ValueError(
                f"theozyme superposition RMSD {rmsd:.2f} A is too high; "
                "check theozyme_map correspondences")

        dmu = None
        if mc.get("delta_mu_theozyme", None) is not None:
            dmu = Rm @ np.asarray(mc.delta_mu_theozyme, dtype=np.float64)

        self._tz_info = (i0, j0, rmsd)
        return Rm @ probe_tz + t, Rm @ axis_tz, dmu

    def setup(self, config: Dict, device: str = "cpu") -> None:
        self.config = config
        self.device = device
        mc = self.model_config = self.config.models.electric_field

        ref_pdb = self.config.input.pdb

        # ---- theozyme route: derive probe / axis / delta_mu automatically --
        self._tz_info = None
        tz_probe = tz_axis = tz_dmu = None
        if mc.get("theozyme_reactant", None) and mc.get("theozyme_ts", None):
            if not mc.get("theozyme_map", None):
                raise ValueError("theozyme_map is required with theozyme_reactant/ts")
            tz_probe, tz_axis, tz_dmu = self._from_theozyme(mc, ref_pdb)

        # ---- probe point: where the field is evaluated -------------------
        if tz_probe is not None:
            self.probe = tz_probe
        elif mc.get("probe_xyz", None):
            self.probe = np.asarray(mc.probe_xyz, dtype=np.float64)
        elif mc.get("probe_atoms", None):
            pts = [atom_coord(ref_pdb, s) for s in mc.probe_atoms]
            self.probe = np.mean(pts, axis=0)
        else:
            raise ValueError(
                "electric_field needs a probe point: set either "
                "models.electric_field.probe_atoms (list of 'RES:NUM:ATOM') "
                "or probe_xyz.")

        # ---- reaction axis: direction that stabilizes the TS -------------
        if tz_axis is not None:
            axis = tz_axis
        elif mc.get("axis_xyz", None):
            axis = np.asarray(mc.axis_xyz, dtype=np.float64)
        elif mc.get("axis_atoms", None):
            a, b = mc.axis_atoms
            axis = atom_coord(ref_pdb, b) - atom_coord(ref_pdb, a)
        else:
            raise ValueError(
                "electric_field needs a reaction axis: set either "
                "models.electric_field.axis_atoms ['RES:NUM:ATOM_from', "
                "'RES:NUM:ATOM_to'] or axis_xyz.")
        n = np.linalg.norm(axis)
        if n < 1e-8:
            raise ValueError("reaction axis has zero length")
        self.axis = axis / n

        # ---- FDB parameters from QM on the theozyme ----------------------
        # delta_mu in Debye, as a vector in the same frame as the axis, or a
        # scalar magnitude (then taken along the axis).
        dmu = mc.get("delta_mu", None)
        if tz_dmu is not None:
            self.dmu = tz_dmu
        elif dmu is None:
            self.dmu = None
        elif np.isscalar(dmu):
            self.dmu = float(dmu) * self.axis
        else:
            self.dmu = np.asarray(dmu, dtype=np.float64)
        # isotropic higher-order FDB terms are not applied (see module docstring)
        self.dalpha = float(mc.get("delta_alpha", 0.0))
        self.dbeta = float(mc.get("delta_beta", 0.0))
        if self.dalpha or self.dbeta:
            print("[electric_field] WARNING: delta_alpha/delta_beta are ignored; "
                  "only the dipole term of the FDB expansion is scored.",
                  file=sys.stderr)

        # ---- charges -----------------------------------------------------
        cf = mc.get("charge_file", None)
        if cf:
            self.charges_all = self._load_charge_file(cf)
            self.charges_united = self.charges_all
            self.custom_charges = True
        else:
            self.charges_all = _Q
            self.charges_united = _Q_UNITED
            self.custom_charges = False

        # residues excluded because they are in the QM model
        self.exclude = [str(x) for x in mc.get("exclude_residues", [])]
        self.include_hetatm = bool(mc.get("include_hetatm", False))
        self.report_au = bool(mc.get("report_au", False))
        self.fix_incomplete = bool(mc.get("fix_incomplete", True))
        self.verbose = bool(mc.get("verbose", False))

        # ---- decide all-atom vs united-atom once, from the reference -----
        _, _, has_h, _ = parse_pdb_charged(ref_pdb, self.charges_all,
                                           self.exclude, self.include_hetatm)
        self.use_united = (not has_h) and (not self.custom_charges)
        self._table = self.charges_united if self.use_united else self.charges_all

        # ---- sanity check on the reference structure ---------------------
        warn = []
        xyz, q, _, nfix = parse_pdb_charged(ref_pdb, self._table, self.exclude,
                                            self.include_hetatm,
                                            fix_incomplete=self.fix_incomplete,
                                            warn=warn)
        if warn:
            print(f"[electric_field] renormalized {nfix} residue(s) with "
                  f"non-formal charge in the reference structure:", file=sys.stderr)
            for w in warn[:8]:
                print("    " + w, file=sys.stderr)
            if len(warn) > 8:
                print(f"    ... and {len(warn)-8} more", file=sys.stderr)
        if len(xyz) == 0:
            raise ValueError(f"no chargeable atoms parsed from {ref_pdb}")
        dmin = np.linalg.norm(xyz - self.probe, axis=1).min()
        if dmin > 8.0:
            raise ValueError(
                f"probe point is {dmin:.1f} A from the nearest protein atom; "
                "check that probe_atoms and input.pdb share a frame.")

        outputs = self.config.general.outputs
        self.output_dir = os.path.join(outputs, "electric_field")
        os.makedirs(self.output_dir, exist_ok=True)

        if self.verbose:
            F = self._field(xyz, q)
            if self._tz_info:
                i0, j0, rmsd = self._tz_info
                print(f"[electric_field] theozyme: breaking bond {i0}->{j0}, "
                      f"superposition RMSD {rmsd:.2f} A", flush=True)
            print(f"[electric_field] reference {os.path.basename(ref_pdb)}: "
                  f"|F|={np.linalg.norm(F):.1f} MV/cm  "
                  f"F.axis={float(F @ self.axis):+.1f} MV/cm  "
                  f"({'united' if self.use_united else 'all'}-atom charges, "
                  f"{len(xyz)} atoms, net {q.sum():+.2f} e)", flush=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _load_charge_file(path):
        """
        Whitespace/comma separated: RESNAME ATOMNAME CHARGE
        (e.g. dumped from a tleap-generated prmtop).
        """
        table = {}
        for line in open(path):
            line = line.split("#")[0].replace(",", " ").split()
            if len(line) < 3:
                continue
            res, atom, q = line[0].upper(), line[1], float(line[2])
            table.setdefault(res, {})[atom] = q
        if not table:
            raise ValueError(f"no charges parsed from {path}")
        return table

    def _field(self, xyz, q):
        """Vacuum Coulomb field at self.probe, Hunt et al. eq 6, in MV/cm."""
        d = self.probe - xyz
        r2 = (d * d).sum(1)
        r = np.sqrt(r2)
        ok = r > 1e-6
        w = (q[ok] / (r2[ok] * r[ok]))[:, None]
        return COULOMB_MV_CM * (d[ok] * w).sum(0)

    def get_components(self, pdb_path):
        xyz, q, _, _ = parse_pdb_charged(pdb_path, self._table, self.exclude,
                                         self.include_hetatm,
                                         fix_incomplete=self.fix_incomplete)
        if len(xyz) == 0:
            raise ValueError("no chargeable atoms parsed")
        F = self._field(xyz, q)
        Fmag = float(np.linalg.norm(F))
        Fproj = float(F @ self.axis)

        if self.dmu is not None:
            # Dipole term of FDB eq 7 only. Barrier drops when F aligns with dMu.
            ddE = -(self.dmu @ F) * D_MVCM_TO_KCAL
            fitness = -ddE                        # barrier reduction, kcal/mol
        else:
            fitness = Fproj * (AU_PER_MVCM if self.report_au else 1.0)

        # cosine similarity to the axis, as in the paper's Table 5a
        cos = Fproj / Fmag if Fmag > 1e-12 else 0.0
        return {"F": F, "F_magnitude": Fmag, "F_projection": Fproj,
                "cos_to_axis": cos, "fitness": fitness}

    # ------------------------------------------------------------------
    def score(self, individual: Individual):
        pdb_path = individual.get_name()
        gen, index = individual.get_gen(), individual.get_index()
        try:
            c = self.get_components(pdb_path)
            value = c["fitness"]
            if self.verbose:
                print(f"[electric_field] gen {gen} idx {index}  "
                      f"|F|={c['F_magnitude']:.1f}  "
                      f"F.axis={c['F_projection']:+.1f} MV/cm  "
                      f"cos={c['cos_to_axis']:+.2f}  "
                      f"fitness={value:+.3f}", flush=True)
        except Exception:
            print(f"[electric_field] scoring failed for {pdb_path}", file=sys.stderr)
            traceback.print_exc()
            value = -10.0
        individual.add_fitness({"electric_field": float(value)})
