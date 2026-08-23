"""Backbone accommodation around a frozen theozyme."""
import numpy as np
from scipy.spatial import cKDTree

from .relax import RestrainedRelax, propagate_sidechains
from .rigid import IDEAL, IDEAL_N_CB, IDEAL_C_CB, IDEAL_N_C

PEPTIDE_C_N = 1.329
from .loops import displacement_demand, tier_of

HEAVY = lambda names: np.array([not str(n).startswith('H') for n in names])


class SubstrateAwareRelax(RestrainedRelax):
    """RestrainedRelax that understands who owns each no-go atom."""

    def __init__(self, st, mobile_resis, cloud_xyz=None, cloud_owner=None,
                 cloud_kind=None, **kw):
        super().__init__(st, mobile_resis, substrate_xyz=cloud_xyz, **kw)
        self.cloud_owner = None if cloud_owner is None else np.asarray(cloud_owner, int)
        self.cloud_kind = (np.zeros(len(self.cloud_owner), int)
                           if cloud_kind is None and self.cloud_owner is not None
                           else (None if cloud_kind is None else np.asarray(cloud_kind, int)))
        self._kenm_soft = float(kw.get('k_mobile', 0.6))
        self._kenm_stiff = float(kw.get('k_rigid', 8.0))
        self.compliance = None
        self.pin_mask = np.zeros(len(self.idx), bool)
        self.k_pin = 2000.0
        self.pairs = []

    def apply_compliance_gradient(self, focus_xyz, r0=6.0, r1=None, floor=0.03,
                                  stiff_resis=()):
        """Replace binary mobile/outer split with a smooth compliance ramp."""
        d = cKDTree(np.asarray(focus_xyz, float)).query(self.x0)[0]
        if r1 is None:
            r1 = float(d.max())
        r1 = max(r1, r0 + 1e-6)
        t = np.clip((d - r0) / (r1 - r0), 0.0, 1.0)
        w = 1.0 - t * t * (3.0 - 2.0 * t)
        w = floor + (1.0 - floor) * w
        if len(stiff_resis):
            keep = np.zeros(len(w), bool)
            for r in stiff_resis:
                keep |= np.abs(self.resi - int(r)) <= 1
            w = np.where(keep, floor, w)
        self.compliance = w

        def blend(k_soft, k_stiff, weight):
            return np.exp(np.log(k_stiff)
                          + weight * (np.log(max(k_soft, 1e-8)) - np.log(k_stiff)))

        self.kpos = blend(self._kposm, self._kposr, w)
        wij = np.maximum(w[self.pi], w[self.pj])
        self.kenm = blend(self._kenm_soft, self._kenm_stiff, wij)
        self.kenm = np.where(self.bonded, max(self._kenm_stiff, 20.0), self.kenm)
        return w

    def pin(self, resi, names=('CA', 'CB')):
        """Freeze these atoms in place with a stiff positional restraint."""
        n = 0
        for k, j in enumerate(self.idx):
            if int(self.st.resi[j]) == int(resi) and str(self.st.name[j]) in names:
                self.pin_mask[k] = True
                self.is_mob[k] = False
                n += 1
        return n

    def _apply_pins(self):
        if self.pin_mask.any():
            self.kpos = np.where(self.pin_mask, self.k_pin, self.kpos)

    def local_index(self, resi, name):
        for k, j in enumerate(self.idx):
            if int(self.st.resi[j]) == int(resi) and str(self.st.name[j]) == name:
                return k
        return None

    def add_pair(self, resi_a, name_a, resi_b, name_b, lo, hi, k=60.0):
        """Flat-bottom restraint between two atoms that are both in the relaxation."""
        i = self.local_index(resi_a, name_a)
        j = self.local_index(resi_b, name_b)
        if i is None or j is None:
            return False
        self.pairs.append((i, j, float(lo), float(hi), float(k)))
        return True

    def restrain_backbone_geometry(self, tol=0.03, k=300.0, resis=None):
        """Hold N-CA, CA-C and peptide C-N at ideal length."""
        chain = self.chain
        nums = sorted({int(r) for c, r in self.st.protein_res if c == chain}
                      if resis is None else {int(r) for r in resis})
        n = 0
        for r in nums:
            n += self.add_pair(r, 'N', r, 'CA',
                               IDEAL['N_CA'] - tol, IDEAL['N_CA'] + tol, k)
            n += self.add_pair(r, 'CA', r, 'C',
                               IDEAL['CA_C'] - tol, IDEAL['CA_C'] + tol, k)
            n += self.add_pair(r, 'CA', r, 'CB',
                               IDEAL['CA_CB'] - tol, IDEAL['CA_CB'] + tol, k)
            if r + 1 in nums:
                n += self.add_pair(r, 'C', r + 1, 'N',
                                   PEPTIDE_C_N - tol, PEPTIDE_C_N + tol, k)
        return n

    def _build_pairs(self, X):
        super()._build_pairs(X)
        if self._sp is None or not len(self._sp) or self.cloud_owner is None:
            return
        pi_, sj = self._sp[:, 0], self._sp[:, 1]
        own = self.cloud_owner[sj]
        kind = self.cloud_kind[sj]
        rp = self.resi[pi_]
        drop_sc = (kind == 0) & (own >= 0) & (np.abs(rp - own) <= 1)
        drop_lig = (kind == 1) & (own >= 0) & (rp == own)
        self._sp = self._sp[~(drop_sc | drop_lig)]

    def _energy(self, x):
        E, g = super()._energy(x)
        if not self.pairs:
            return E, g
        X = x.reshape(-1, 3)
        G = g.reshape(-1, 3)
        for i, j, lo, hi, k in self.pairs:
            v = X[i] - X[j]
            d = float(np.linalg.norm(v))
            if lo <= d <= hi:
                continue
            e = (lo - d) if d < lo else (d - hi)
            s = -1.0 if d < lo else 1.0
            E += k * e ** 2
            gv = 2 * k * e * s * v / max(d, 1e-8)
            G[i] += gv
            G[j] -= gv
        return E, g.ravel()

    def run(self, maxiter=400):
        if self.compliance is not None:
            self.kpos = np.exp(
                np.log(self._kposr)
                + self.compliance * (np.log(max(self._kposm, 1e-8)) - np.log(self._kposr)))
        self._apply_pins()
        return super().run(maxiter=maxiter)

    def run_staged(self, schedule=((1.0, 1.0), (2.5, 0.4), (6.0, 0.15)), maxiter=300,
                   target_clearance=None):
        if self.sub is None:
            target_clearance = None
        return super().run_staged(schedule=schedule, maxiter=maxiter,
                                  target_clearance=target_clearance)


def backbone_bond_audit(st, xyz, chain=None):
    """Worst backbone bond deviations."""
    chain = chain or str(st.chain[0])
    tgt = {'N-CA': IDEAL['N_CA'], 'CA-C': IDEAL['CA_C'], 'C-N': PEPTIDE_C_N}
    got = {k: [] for k in tgt}
    pos = {}
    for c, r in st.protein_res:
        if c != chain:
            continue
        pos[int(r)] = {str(st.name[i]): xyz[i] for i in st.residues[(c, r)]}
    for r, a in pos.items():
        if 'N' in a and 'CA' in a:
            got['N-CA'].append(float(np.linalg.norm(a['CA'] - a['N'])))
        if 'CA' in a and 'C' in a:
            got['CA-C'].append(float(np.linalg.norm(a['C'] - a['CA'])))
        b = pos.get(r + 1)
        if b and 'C' in a and 'N' in b:
            got['C-N'].append(float(np.linalg.norm(b['N'] - a['C'])))
    out = {}
    worst = 0.0
    for k, v in got.items():
        if not v:
            continue
        d = np.abs(np.array(v) - tgt[k])
        out[k] = dict(mean=round(float(np.mean(v)), 3), max_dev=round(float(d.max()), 3),
                      rms_dev=round(float(np.sqrt((d ** 2).mean())), 3), n=len(v))
        worst = max(worst, float(d.max()))
    out['worst_dev'] = round(worst, 3)
    return out


def all_atom_clearance(st, xyz, rigid, ignore_resis=()):
    """Closest approach between ligand and non-theozyme protein heavy atoms."""
    lig = rigid.ligand_heavy()
    if not len(lig):
        return None
    skip = set(int(r) for r in list(rigid.resis) + list(ignore_resis))
    keep = [i for i in range(len(xyz))
            if str(st.rec[i]) == 'ATOM'
            and int(st.resi[i]) not in skip
            and not str(st.elem[i]).startswith('H')]
    if not keep:
        return None
    d, _ = cKDTree(lig).query(xyz[keep])
    k = int(np.argmin(d))
    i = keep[k]
    return dict(min_dist=round(float(d[k]), 2),
                atom=f'{st.resn[i]}{int(st.resi[i])}.{st.name[i]}')


_BB = frozenset(('N', 'CA', 'C', 'O', 'OXT'))


def count_sidechain_clashes(st, xyz, rigid=None, lig_xyz=None, ignore_resis=(),
                            lig_clearance=3.40, protein_clearance=3.1, chain=None):
    """Count side-chain steric clashes after ``propagate_sidechains_cb`` (no pack).

    Protein–ligand: non-theozyme SC heavy atoms within ``lig_clearance`` of the
    ligand. Protein–protein: SC–SC pairs on non-adjacent residues within
    ``protein_clearance``. Useful for ranking backbone-only PD draws by how well
    they relieve clashes when SCs are rigidly carried on the new frame.
    """
    xyz = np.asarray(xyz, float)
    chain = chain or (rigid.chain if rigid is not None else str(st.chain[0]))
    skip = set(int(r) for r in ignore_resis)
    if rigid is not None:
        skip.update(int(r) for r in rigid.resis)
        if lig_xyz is None:
            lig_xyz = rigid.ligand_heavy()

    sc_by_res = {}
    for c, r in st.protein_res:
        if c != chain or int(r) in skip:
            continue
        idx = [i for i in st.residues[(c, r)]
               if str(st.name[i]) not in _BB
               and not str(st.elem[i]).startswith('H')]
        if idx:
            sc_by_res[int(r)] = idx

    n_lig = 0
    min_lig = None
    if lig_xyz is not None and len(lig_xyz) and sc_by_res:
        all_sc = [i for idx in sc_by_res.values() for i in idx]
        d, _ = cKDTree(np.asarray(lig_xyz, float)).query(xyz[all_sc])
        n_lig = int(np.sum(d < lig_clearance))
        min_lig = round(float(d.min()), 2) if len(d) else None

    n_pp = 0
    if len(sc_by_res) >= 2:
        resis = sorted(sc_by_res)
        pts, owners = [], []
        for r in resis:
            for i in sc_by_res[r]:
                pts.append(xyz[i])
                owners.append(r)
        pts = np.asarray(pts, float)
        owners = np.asarray(owners, int)
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=protein_clearance)
        for i, j in pairs:
            if abs(int(owners[i]) - int(owners[j])) > 1:
                n_pp += 1

    return dict(n_lig=n_lig, n_protein=n_pp, n_total=n_lig + n_pp,
                min_lig_dist=min_lig,
                lig_clearance=float(lig_clearance),
                protein_clearance=float(protein_clearance))


def propagate_sidechains_cb(st, old_xyz, new_xyz, chain=None):
    """Carry CG-and-beyond on the CA-CB-N frame instead of N-CA-C."""
    from .rigid import local_frame
    from .geometry import kabsch

    chain = chain or str(st.chain[0])
    out = np.asarray(new_xyz, float).copy()
    old = np.asarray(old_xyz, float)
    fixed = 0
    for (c, r), idx in st.residues.items():
        if c != chain or str(st.rec[idx[0]]) != 'ATOM':
            continue
        name = {str(st.name[i]): i for i in idx}
        beyond = [i for i in idx
                  if str(st.name[i]) not in ('N', 'CA', 'C', 'O', 'CB', 'OXT')]
        if not beyond:
            continue
        if all(a in name for a in ('N', 'CA', 'CB')):
            try:
                Fo = local_frame(old[name['CA']], old[name['CB']], old[name['N']])
                Fn = local_frame(out[name['CA']], out[name['CB']], out[name['N']])
            except ValueError:
                continue
            R = Fn.T @ Fo
            t = out[name['CB']] - R @ old[name['CB']]
        elif all(a in name for a in ('N', 'CA', 'C')):
            core = [name[a] for a in ('N', 'CA', 'C')]
            R, t = kabsch(old[core], out[core])
        else:
            continue
        out[beyond] = (R @ old[beyond].T).T + t
        fixed += 1
    return out


def pick_mobile(st, rigid, shell=10.0, clearance=3.40, chain=None):
    """Residues allowed to move near the ligand. Locked residues are never mobile."""
    chain = chain or rigid.chain
    lig = rigid.ligand_heavy()
    locked = set(rigid.resis)
    mobile = set()
    if len(lig):
        t = cKDTree(lig)
        for c, r in st.protein_res:
            if c != chain or int(r) in locked:
                continue
            idx = [i for i in st.residues[(c, r)] if str(st.name[i]) in ('N', 'CA', 'C', 'O', 'CB')]
            if not idx:
                continue
            d, _ = t.query(st.xyz[idx])
            if d.min() <= shell:
                mobile.add(int(r))
    per, worst = ({}, 0.0)
    if len(lig):
        per, worst = displacement_demand(st, lig, sorted(mobile), chain, clearance)
    return sorted(mobile), per, worst


def relieve_sidechain_clashes(st, xyz, rigid, clearance=3.40, step=15.0,
                              wells=True, well_tol=(50.0, 60.0), min_protein=3.1,
                              max_residues=12, chain=None):
    """Rotate offending side chains off the ligand, staying on-rotamer."""
    from .sidechains import CHI_DEF, ROTAMER_WELLS, off_rotamer
    from .geometry import rotate_about, dihedral

    chain = chain or rigid.chain
    lig = rigid.ligand_heavy()
    if not len(lig):
        return xyz, []
    xyz = np.asarray(xyz, float).copy()
    locked = set(rigid.resis)
    lig_tree = cKDTree(lig)

    cands = []
    for c, r in st.protein_res:
        if c != chain or int(r) in locked:
            continue
        idx = [i for i in st.residues[(c, r)]
               if str(st.name[i]) not in ('N', 'CA', 'C', 'O')
               and not str(st.elem[i]).startswith('H')]
        if not idx:
            continue
        resn = str(st.resn[idx[0]])
        if not CHI_DEF.get(resn):
            continue
        d, _ = lig_tree.query(xyz[idx])
        if d.min() < clearance:
            cands.append((float(d.min()), int(r), resn, idx))
    cands.sort()
    cands = cands[:max_residues]
    if not cands:
        return xyz, []

    changed = []
    for d0, r, resn, idx in cands:
        pos = {str(st.name[i]): xyz[i].copy() for i in st.residues[(chain, r)]}
        if not all(a in pos for a in ('N', 'CA', 'CB')):
            continue
        chis = CHI_DEF[resn]
        n_chi = min(2, len(chis))
        own = set(st.residues[(chain, r)])
        env_idx = [i for i in range(len(xyz))
                   if i not in own and str(st.rec[i]) == 'ATOM'
                   and not str(st.elem[i]).startswith('H')
                   and abs(int(st.resi[i]) - r) > 1
                   and int(st.resi[i]) not in locked]
        env = cKDTree(xyz[env_idx]) if env_idx else None

        def build(deltas):
            s = {k: v.copy() for k, v in pos.items()}
            for k, ((b0, b1), moving) in enumerate(chis):
                if k >= len(deltas) or abs(deltas[k]) < 1e-9:
                    continue
                if b0 not in s or b1 not in s:
                    continue
                mv = [m for m in moving if m in s]
                if not mv:
                    continue
                P = rotate_about(np.array([s[m] for m in mv]), s[b0], s[b1] - s[b0],
                                 np.radians(deltas[k]))
                for m, p in zip(mv, P):
                    s[m] = p
            return s

        def current_chi(k):
            (b0, b1), moving = chis[k]
            prev = 'N' if k == 0 else chis[k - 1][0][0]
            nm = [prev, b0, b1, moving[0]]
            if not all(x in pos for x in nm):
                return None
            return float(dihedral(*[pos[x] for x in nm]))

        base = [current_chi(k) for k in range(n_chi)]
        W = ROTAMER_WELLS.get(resn) if wells else None
        sc_names = [nm for nm in pos if nm not in ('N', 'CA', 'C', 'O')]
        P0 = np.array([pos[nm] for nm in sc_names])
        dp0 = float(env.query(P0)[0].min()) if env is not None else np.inf
        floor = min(min_protein, dp0)
        grid = np.arange(-180.0, 180.0, step)
        best = None
        combos = ([[g] for g in grid] if n_chi == 1
                  else [[a, b] for a in grid for b in grid])
        for delta in combos:
            if W is not None and all(b is not None for b in base):
                bad = False
                for k, dk in enumerate(delta):
                    if k >= len(W):
                        break
                    tol = well_tol[k] if k < len(well_tol) else well_tol[-1]
                    if off_rotamer(base[k] + dk, W[k]) > tol:
                        bad = True
                        break
                if bad:
                    continue
            s = build(delta)
            P = np.array([s[nm] for nm in sc_names])
            dl = float(lig_tree.query(P)[0].min())
            if env is not None:
                dp = float(env.query(P)[0].min())
                if dp < floor - 1e-6:
                    continue
            move = float(np.linalg.norm(P - P0, axis=1).max())
            score = (min(dl, clearance), -move)
            if best is None or score > best[0]:
                best = (score, s, dl, delta)
        if best is None or best[2] <= d0 + 0.05:
            continue
        _, s, dl, delta = best
        for i in st.residues[(chain, r)]:
            nm = str(st.name[i])
            if nm in s:
                xyz[i] = s[nm]
        changed.append(dict(resi=r, resn=resn, before=round(d0, 2),
                            after=round(dl, 2),
                            dchi=[round(float(x), 1) for x in delta]))
    return xyz, changed


def accommodate(st, rigid, xyz=None, mobile_resis=None, mode='fixed',
                clearance=3.40, shell=10.0, enm_cutoff=9.0,
                k_rigid=8.0, k_mobile=0.6, k_pos_rigid=2.0, k_pos_mobile=0.02,
                rep_dmin=3.30, k_rep=25.0, closure_tol=0.08, k_closure=60.0,
                idealise_backbone=True, bond_tol=0.03, k_bond=300.0,
                gradient=False, grad_r0=6.0, grad_r1=None, grad_floor=0.03,
                maxiter=300, schedule=((1.0, 1.0), (2.5, 0.4), (6.0, 0.15)),
                dchi1=None, relieve_sidechains=True, verbose=False):
    """Relax backbone around a frozen theozyme. Returns (new_xyz, report)."""
    xyz = st.xyz.copy() if xyz is None else np.asarray(xyz, float).copy()
    work = type(st).__new__(type(st))
    work.__dict__.update(st.__dict__)
    work.xyz = xyz

    cloud, owner = rigid.no_go_cloud()
    kind = np.zeros(len(owner), int)
    n_sc = sum(len([n for n in rr.atoms if not n.startswith('H')])
               for rr in rigid.residues.values())
    kind[n_sc:] = 1

    if mobile_resis is None:
        mobile_resis, demand_before, worst_before = pick_mobile(
            work, rigid, shell=shell, clearance=clearance)
    else:
        mobile_resis = [int(r) for r in mobile_resis if int(r) not in set(rigid.resis)]
        lig = rigid.ligand_heavy()
        demand_before, worst_before = (displacement_demand(
            work, lig, mobile_resis, rigid.chain, clearance) if len(lig) else ({}, 0.0))

    rx = SubstrateAwareRelax(
        work, mobile_resis, cloud_xyz=cloud if len(cloud) else None,
        cloud_owner=owner if len(cloud) else None,
        cloud_kind=kind if len(cloud) else None, chain=rigid.chain,
        enm_cutoff=enm_cutoff, k_rigid=k_rigid, k_mobile=k_mobile,
        k_pos_rigid=k_pos_rigid, k_pos_mobile=k_pos_mobile,
        rep_dmin=rep_dmin, k_rep=k_rep, sub_dmin=clearance)

    n_pinned = 0
    for r, rr in rigid.residues.items():
        n_pinned += rx.pin(r, names=('CA', 'CB'))
        ca, cb = rr.atoms['CA'], rr.atoms['CB']
        t = closure_tol
        rx.add_reach(r, ca, IDEAL['N_CA'] - t, IDEAL['N_CA'] + t, k_closure, name='N')
        rx.add_reach(r, ca, IDEAL['CA_C'] - t, IDEAL['CA_C'] + t, k_closure, name='C')
        rx.add_reach(r, cb, IDEAL_N_CB - t, IDEAL_N_CB + t, k_closure, name='N')
        rx.add_reach(r, cb, IDEAL_C_CB - t, IDEAL_C_CB + t, k_closure, name='C')
        rx.add_pair(r, 'N', r, 'C', IDEAL_N_C - t, IDEAL_N_C + t, k_closure)

    if gradient and len(cloud):
        rx.apply_compliance_gradient(cloud, r0=grad_r0, r1=grad_r1, floor=grad_floor,
                                     stiff_resis=rigid.resis)

    if idealise_backbone:
        rx.restrain_backbone_geometry(tol=bond_tol, k=k_bond)

    res = rx.run_staged(schedule=schedule, maxiter=maxiter,
                        target_clearance=clearance if len(cloud) else None)

    new = res['xyz']
    new = propagate_sidechains_cb(work, xyz, new, rigid.chain)

    if mode == 'fixed':
        new = rigid.restore(work, new)
    elif mode == 'follow':
        new = rigid.follow(work, new, dchi1=dchi1)
    else:
        raise ValueError(f"mode must be 'fixed' or 'follow', got {mode!r}")

    sc_changed = []
    if relieve_sidechains:
        new, sc_changed = relieve_sidechain_clashes(
            work, new, rigid, clearance=clearance, chain=rigid.chain)

    lig = rigid.ligand_heavy()
    demand_after, worst_after = (displacement_demand(
        _with(work, new), lig, mobile_resis, rigid.chain, clearance)
        if len(lig) else ({}, 0.0))

    report = dict(
        mode=mode,
        n_mobile=len(mobile_resis),
        mobile_resis=list(mobile_resis),
        n_pinned_atoms=n_pinned,
        energy=round(res['energy'], 2),
        niter=res['niter'],
        max_disp_mobile=round(res['max_disp_mobile'], 3),
        rmsd_mobile=round(res['rmsd_mobile'], 3),
        max_disp_rigid=round(res['max_disp_rigid'], 3),
        rmsd_rigid=round(res['rmsd_rigid'], 3),
        substrate_demand_before=round(worst_before, 2),
        substrate_demand_after=round(worst_after, 2),
        tier_before=tier_of(worst_before),
        tier_after=tier_of(worst_after),
        n_residues_in_debt_before=len(demand_before),
        n_residues_in_debt_after=len(demand_after),
        clearance_before=all_atom_clearance(work, xyz, rigid),
        clearance_after=all_atom_clearance(work, new, rigid),
        bonds_before=backbone_bond_audit(work, xyz, rigid.chain),
        bonds_after=backbone_bond_audit(work, new, rigid.chain),
        sidechains_relieved=sc_changed,
        theozyme=rigid.verify(work, new),
    )
    if verbose:
        print(_fmt(report))
    return new, report


def _with(st, xyz):
    s = type(st).__new__(type(st))
    s.__dict__.update(st.__dict__)
    s.xyz = xyz
    return s


def _fmt(rep):
    L = ['[accommodate] mode=%s  mobile=%d  pinned=%d atoms'
         % (rep['mode'], rep['n_mobile'], rep['n_pinned_atoms']),
         '  backbone   rmsd(mobile)=%.2f  max(mobile)=%.2f  rmsd(rest)=%.2f'
         % (rep['rmsd_mobile'], rep['max_disp_mobile'], rep['rmsd_rigid']),
         '  substrate  demand %.2f -> %.2f A   tier %s -> %s   residues in debt %d -> %d'
         % (rep['substrate_demand_before'], rep['substrate_demand_after'],
            rep['tier_before'], rep['tier_after'],
            rep['n_residues_in_debt_before'], rep['n_residues_in_debt_after'])]
    cb, ca = rep['clearance_before'], rep['clearance_after']
    if cb and ca:
        L.append('  clearance  %.2f A (%s)  ->  %.2f A (%s)'
                 % (cb['min_dist'], cb['atom'], ca['min_dist'], ca['atom']))
    for c in rep.get('sidechains_relieved', []):
        L.append('  rotated    %s%-5d %.2f -> %.2f A   dchi %s'
                 % (c['resn'], c['resi'], c['before'], c['after'], c['dchi']))
    bb, ba = rep.get('bonds_before'), rep.get('bonds_after')
    if bb and ba:
        L.append('  bonds      worst backbone deviation %.3f -> %.3f A'
                 % (bb['worst_dev'], ba['worst_dev']))
    t = rep['theozyme']
    L.append('  theozyme   internal dev %.2e A   locked-chi drift %.2e deg   ok=%s'
             % (t['max_internal_rmsd'], t['max_locked_chi_drift'], t['ok']))
    for r, d in sorted(t['per_residue'].items()):
        L.append('             %s%-5d chi1=%-7s CA-CB=%-5s chir_err=%-5s shift=%.3f A'
                 % (d['resn'], r, d['chi1'], d['ca_cb'],
                    d['chirality_error'], d['lab_frame_shift']))
    return '\n'.join(L)
