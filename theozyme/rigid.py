"""Frozen theozyme geometry: CA outward plus ligand, chi1 free."""
import numpy as np

from .geometry import dihedral, rotate_about
from .sidechains import CHI_DEF

BACKBONE_FREE = ('N', 'C', 'O', 'OXT')

IDEAL = dict(N_CA=1.458, CA_C=1.525, CA_CB=1.521,
             N_CA_CB=110.5, C_CA_CB=110.1, N_CA_C=111.0)

CB_OUT_OF_PLANE = 50.0


def _unit(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError('degenerate vector')
    return v / n


def ideal_cb(n, ca, c):
    """Build CB on an L-amino-acid backbone."""
    b1, b2 = n - ca, c - ca
    bis = -(_unit(b1) + _unit(b2))
    perp = np.cross(b1, b2)
    v = (_unit(bis) * np.cos(np.radians(CB_OUT_OF_PLANE))
         + _unit(perp) * np.sin(np.radians(CB_OUT_OF_PLANE)))
    return ca + IDEAL['CA_CB'] * _unit(v)


def ideal_distance(a_b, a_c, angle_deg):
    """Law of cosines: b..c distance from two bonds and the angle between."""
    return float(np.sqrt(a_b ** 2 + a_c ** 2 - 2 * a_b * a_c * np.cos(np.radians(angle_deg))))


IDEAL_N_CB = ideal_distance(IDEAL['N_CA'], IDEAL['CA_CB'], IDEAL['N_CA_CB'])
IDEAL_C_CB = ideal_distance(IDEAL['CA_C'], IDEAL['CA_CB'], IDEAL['C_CA_CB'])
IDEAL_N_C = ideal_distance(IDEAL['N_CA'], IDEAL['CA_C'], IDEAL['N_CA_C'])


def local_frame(ca, cb, n):
    """Right-handed orthonormal frame at CA: e1 along CA->CB, e2 towards N."""
    e1 = _unit(cb - ca)
    v = n - ca
    e2 = _unit(v - np.dot(v, e1) * e1)
    return np.stack([e1, e2, np.cross(e1, e2)])


def chirality(n, ca, c, cb):
    """Improper dihedral N-C-CA-CB (~+122 deg for L-amino acids)."""
    return float(dihedral(n, c, ca, cb))


L_IMPROPER = 122.0
L_IMPROPER_TOL = 40.0


def chirality_error(n, ca, c, cb):
    """Angular distance from ideal L geometry, in degrees."""
    v = chirality(n, ca, c, cb)
    return abs((v - L_IMPROPER + 180.0) % 360.0 - 180.0)


class RigidResidue:
    """One theozyme residue: locked atoms plus backbone reference at capture."""

    def __init__(self, resi, resn, chain, atoms, n_ref, c_ref):
        self.resi = int(resi)
        self.resn = str(resn)
        self.chain = str(chain)
        self.atoms = {k: np.asarray(v, float).copy() for k, v in atoms.items()}
        self.n_ref = np.asarray(n_ref, float).copy()
        self.c_ref = np.asarray(c_ref, float).copy()
        if 'CA' not in self.atoms or 'CB' not in self.atoms:
            raise ValueError(f'{resn}{resi}: rigid group needs CA and CB '
                             f'(glycine cannot anchor a theozyme)')
        self.frame_ref = local_frame(self.atoms['CA'], self.atoms['CB'], self.n_ref)

    def chis(self, atoms=None, n=None):
        """chi1.. for this residue from the given atom dict."""
        a = dict(self.atoms if atoms is None else atoms)
        a['N'] = self.n_ref if n is None else np.asarray(n, float)
        out = {}
        for k, ((b0, b1), moving) in enumerate(CHI_DEF.get(self.resn, [])):
            prev = 'N' if k == 0 else CHI_DEF[self.resn][k - 1][0][0]
            names = [prev, b0, b1, moving[0]]
            if not all(nm in a for nm in names):
                continue
            out[f'chi{k + 1}'] = float(dihedral(*[a[nm] for nm in names]))
        return out

    def locked_chis(self, atoms=None, n=None):
        """chi2 onward."""
        return {k: v for k, v in self.chis(atoms, n).items() if k != 'chi1'}

    def transform_for(self, n_new, ca_new, c_new, dchi1=0.0):
        """Rotation+translation taking the frozen group onto a relaxed backbone."""
        cb_new = ideal_cb(n_new, ca_new, c_new)
        frame_new = local_frame(ca_new, cb_new, n_new)
        R = frame_new.T @ self.frame_ref
        t = np.asarray(ca_new, float) - R @ self.atoms['CA']
        if abs(dchi1) > 1e-9:
            axis = cb_new - ca_new
            Rx = _rotmat(axis, np.radians(dchi1))
            R = Rx @ R
            t = np.asarray(ca_new, float) - R @ self.atoms['CA']
        return R, t

    def placed(self, R, t):
        return {k: R @ v + t for k, v in self.atoms.items()}


def _rotmat(axis, theta):
    u = _unit(np.asarray(axis, float))
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


class RigidTheozyme:
    """Frozen catalytic assembly: locked residues plus ligand."""

    def __init__(self, st, resis, ligand_resi=None, ligand_chain=None, chain=None):
        self.chain = chain or str(st.chain[0])
        self.st_ref = st
        self.residues = {}
        for r in resis:
            key = (self.chain, int(r))
            if key not in st.residues:
                raise KeyError(f'theozyme residue {self.chain}{r} not present in the input pdb')
            idx = st.residues[key]
            names = {str(st.name[i]): st.xyz[i] for i in idx}
            for req in ('N', 'CA', 'C'):
                if req not in names:
                    raise ValueError(f'{self.chain}{r} is missing backbone atom {req}')
            locked = {nm: xyz for nm, xyz in names.items() if nm not in BACKBONE_FREE}
            self.residues[int(r)] = RigidResidue(
                r, str(st.resn[idx[0]]), self.chain, locked, names['N'], names['C'])

        self.lig_key = None
        self.lig_names, self.lig_xyz = [], np.zeros((0, 3))
        het = st.ligand_res()
        if ligand_resi is not None:
            self.lig_key = (str(ligand_chain or self.chain), int(ligand_resi))
        elif len(het) == 1:
            self.lig_key = het[0]
        elif len(het) > 1:
            raise ValueError(f'{len(het)} HETATM residues present {het}; '
                             f'set ligand_resi explicitly')
        if self.lig_key is not None:
            if self.lig_key not in st.residues:
                raise KeyError(f'ligand {self.lig_key} not found in the input pdb')
            idx = st.residues[self.lig_key]
            self.lig_names = [str(st.name[i]) for i in idx]
            self.lig_resn = str(st.resn[idx[0]])
            self.lig_elem = [str(st.elem[i]) for i in idx]
            self.lig_xyz = st.xyz[idx].copy()

        self.lig_owner = None
        if len(self.lig_xyz):
            best = (1e9, None)
            for r, rr in self.residues.items():
                for nm, p in rr.atoms.items():
                    d = float(np.min(np.linalg.norm(self.lig_xyz - p, axis=1)))
                    if d < best[0]:
                        best = (d, r)
            self.lig_owner, self.lig_link_dist = best[1], best[0]

        self.locked_chis_ref = {r: rr.locked_chis() for r, rr in self.residues.items()}

    @property
    def resis(self):
        return sorted(self.residues)

    def heavy(self, names, xyz):
        keep = [i for i, nm in enumerate(names) if not nm.startswith('H')]
        return xyz[keep]

    def ligand_heavy(self):
        return self.heavy(self.lig_names, self.lig_xyz)

    def no_go_cloud(self):
        """Atoms the relaxing backbone must not run into, with owner tags."""
        pts, owner = [], []
        for r, rr in self.residues.items():
            for nm, p in rr.atoms.items():
                if nm.startswith('H'):
                    continue
                pts.append(p)
                owner.append(r)
        for p in self.ligand_heavy():
            pts.append(p)
            owner.append(self.lig_owner if self.lig_owner is not None else -1)
        if not pts:
            return np.zeros((0, 3)), np.zeros(0, int)
        return np.asarray(pts, float), np.asarray(owner, int)

    def atom_indices(self, st):
        """Indices in `st` of every frozen atom."""
        out = []
        for r in self.residues:
            for i in st.residues[(self.chain, r)]:
                if str(st.name[i]) not in BACKBONE_FREE:
                    out.append(i)
        if self.lig_key is not None and self.lig_key in st.residues:
            out.extend(st.residues[self.lig_key])
        return np.asarray(sorted(out), int)

    def backbone_indices(self, st):
        """Indices of N/C/O of locked residues."""
        out = []
        for r in self.residues:
            for i in st.residues[(self.chain, r)]:
                if str(st.name[i]) in BACKBONE_FREE:
                    out.append(i)
        return np.asarray(sorted(out), int)

    def graft_missing(self, st):
        """Insert any locked theozyme / ligand atoms that the packed PDB omitted.

        LigandMPNN packing sometimes drops tip atoms (mask bit off). In fixed
        mode those coords are replaced from the reference anyway, so missing
        slots are safe to recreate. Returns a list of human-readable grafts.
        Hard failures (missing residue, wrong amino acid) are not repaired.
        """
        added = []
        new = []
        for r, rr in self.residues.items():
            key = (self.chain, r)
            if key not in st.residues:
                continue
            idx = st.residues[key]
            if str(st.resn[idx[0]]) != rr.resn:
                continue
            have = {str(st.name[i]) for i in idx}
            for nm, xyz in rr.atoms.items():
                if nm in have:
                    continue
                new.append(dict(rec='ATOM', name=nm, resn=rr.resn,
                                chain=self.chain, resi=r, xyz=xyz))
                added.append(f'{rr.resn}{r}.{nm}')
        if self.lig_key is not None and self.lig_key not in st.residues and len(self.lig_names):
            ch, resi = self.lig_key
            for nm, xyz, el in zip(self.lig_names, self.lig_xyz, self.lig_elem):
                new.append(dict(rec='HETATM', name=nm, resn=self.lig_resn,
                                chain=ch, resi=resi, xyz=xyz, elem=el))
            added.append(f'ligand {self.lig_resn}{resi} ({len(self.lig_names)} atoms)')
        if new:
            st.append_atoms(new)
        return added

    def restore(self, st, xyz):
        """Write frozen coordinates back into `xyz` ('fixed' mode)."""
        out = np.asarray(xyz, float).copy()
        for r, rr in self.residues.items():
            for i in st.residues[(self.chain, r)]:
                nm = str(st.name[i])
                if nm in rr.atoms:
                    out[i] = rr.atoms[nm]
        if self.lig_key is not None and self.lig_key in st.residues:
            out[st.residues[self.lig_key]] = self.lig_xyz
        return out

    def follow(self, st, xyz, dchi1=None):
        """Carry frozen group onto relaxed backbone ('follow' mode)."""
        out = np.asarray(xyz, float).copy()
        dchi1 = dchi1 or {}
        applied = {}
        for r, rr in self.residues.items():
            idx = st.residues[(self.chain, r)]
            bb = {str(st.name[i]): out[i] for i in idx if str(st.name[i]) in ('N', 'C')}
            ca_i = [i for i in idx if str(st.name[i]) == 'CA']
            if 'N' not in bb or 'C' not in bb or not ca_i:
                continue
            R, t = rr.transform_for(bb['N'], out[ca_i[0]], bb['C'],
                                    float(dchi1.get(r, 0.0)))
            placed = rr.placed(R, t)
            for i in idx:
                nm = str(st.name[i])
                if nm in placed:
                    out[i] = placed[nm]
            applied[r] = (R, t)
        if self.lig_key is not None and self.lig_key in st.residues and self.lig_owner in applied:
            R, t = applied[self.lig_owner]
            out[st.residues[self.lig_key]] = (R @ self.lig_xyz.T).T + t
        return out

    def verify(self, st, xyz, tol=1e-4):
        """Confirm locked geometry survived. Returns report; `ok` is the gate."""
        xyz = np.asarray(xyz, float)
        rep = {'per_residue': {}, 'max_internal_rmsd': 0.0,
               'max_locked_chi_drift': 0.0, 'ok': True}
        for r, rr in self.residues.items():
            idx = st.residues[(self.chain, r)]
            got = {str(st.name[i]): xyz[i] for i in idx}
            names = [nm for nm in rr.atoms if nm in got]
            P = np.array([rr.atoms[nm] for nm in names])
            Q = np.array([got[nm] for nm in names])
            dP = np.linalg.norm(P[:, None] - P[None], axis=-1)
            dQ = np.linalg.norm(Q[:, None] - Q[None], axis=-1)
            internal = float(np.abs(dP - dQ).max()) if len(names) > 1 else 0.0
            drift = 0.0
            if 'N' in got:
                now = rr.locked_chis(got, got['N'])
                for k, v in self.locked_chis_ref[r].items():
                    if k in now:
                        d = abs((now[k] - v + 180.0) % 360.0 - 180.0)
                        drift = max(drift, d)
            moved = float(np.linalg.norm(Q - P, axis=1).max()) if len(names) else 0.0
            ca = got.get('CA')
            cb = got.get('CB')
            bond = float(np.linalg.norm(ca - cb)) if ca is not None and cb is not None else None
            chi = None
            chi_err = None
            if all(k in got for k in ('N', 'CA', 'C', 'CB')):
                chi = chirality(got['N'], got['CA'], got['C'], got['CB'])
                chi_err = chirality_error(got['N'], got['CA'], got['C'], got['CB'])
            rep['per_residue'][r] = dict(
                resn=rr.resn, internal_max_dev=round(internal, 6),
                locked_chi_drift=round(drift, 4), lab_frame_shift=round(moved, 4),
                ca_cb=None if bond is None else round(bond, 3),
                improper_N_C_CA_CB=None if chi is None else round(chi, 1),
                chirality_error=None if chi_err is None else round(chi_err, 1),
                chi1=round(rr.chis(got, got['N']).get('chi1', float('nan')), 1)
                if 'N' in got else None)
            rep['max_internal_rmsd'] = max(rep['max_internal_rmsd'], internal)
            rep['max_locked_chi_drift'] = max(rep['max_locked_chi_drift'], drift)
            if chi_err is not None and chi_err > L_IMPROPER_TOL:
                rep['ok'] = False
                rep['per_residue'][r]['ERROR'] = (
                    f'CA chirality {chi:.0f} deg is {chi_err:.0f} deg off L geometry')
            if bond is not None and not (1.35 <= bond <= 1.70):
                rep['ok'] = False
                rep['per_residue'][r]['ERROR'] = f'CA-CB bond {bond:.2f} A out of range'

        if self.lig_key is not None and self.lig_key in st.residues:
            L = xyz[st.residues[self.lig_key]]
            dP = np.linalg.norm(self.lig_xyz[:, None] - self.lig_xyz[None], axis=-1)
            dQ = np.linalg.norm(L[:, None] - L[None], axis=-1)
            rep['ligand_internal_max_dev'] = round(float(np.abs(dP - dQ).max()), 6)
            rep['ligand_lab_frame_shift'] = round(
                float(np.linalg.norm(L - self.lig_xyz, axis=1).max()), 4)
            rep['max_internal_rmsd'] = max(rep['max_internal_rmsd'],
                                           rep['ligand_internal_max_dev'])
        if rep['max_internal_rmsd'] > tol or rep['max_locked_chi_drift'] > 1e-2:
            rep['ok'] = False
        return rep

    def summary(self):
        L = [f'rigid theozyme: {len(self.residues)} locked residue(s), '
             f'{len(self.lig_names)} ligand atoms']
        for r, rr in sorted(self.residues.items()):
            chis = rr.chis()
            txt = '  '.join(f'{k}={v:+7.1f}' for k, v in sorted(chis.items()))
            L.append(f'  {rr.resn}{r}  frozen atoms {len(rr.atoms):2d}   {txt}')
            L.append(f'{"":14s}locked: ' +
                     ', '.join(sorted(self.locked_chis_ref[r])) + '   (chi1 is free)')
        if self.lig_owner is not None:
            L.append(f'  ligand {self.lig_resn}{self.lig_key[1]} bonded to residue '
                     f'{self.lig_owner} at {self.lig_link_dist:.2f} A')
        return '\n'.join(L)
