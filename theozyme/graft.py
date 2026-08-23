"""CB grafting: shift backbone triads instead of pulling CB off its cone."""
import numpy as np

from .rigid import ideal_cb, chirality, chirality_error, IDEAL, L_IMPROPER

BB3 = ('N', 'CA', 'C')


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(
        np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2), -1.0, 1.0))))


def cb_backbone_shift(st, resi, cb_target, chain='A'):
    """Return (shift, info) to place ideal CB on cb_target, or (None, info) if triad incomplete."""
    n, ca, c = (st.atom(resi, x, chain) for x in BB3)
    if any(v is None for v in (n, ca, c)):
        return None, dict(resi=int(resi), reason='incomplete backbone triad')
    cb_now = ideal_cb(n, ca, c)
    shift = np.asarray(cb_target, float) - cb_now
    return shift, dict(resi=int(resi), magnitude=round(float(np.linalg.norm(shift)), 3),
                       vector=[round(float(v), 3) for v in shift],
                       cb_native_offset=round(float(np.linalg.norm(
                           cb_target - (st.atom(resi, 'CB', chain)
                                        if st.atom(resi, 'CB', chain) is not None
                                        else cb_now))), 3))


def add_graft_restraints(rx, st, resi, cb_target, chain='A', tol=0.15, k=120.0,
                         max_shift=None):
    """Restrain N/CA/C so implied CB lands on target. Returns info dict."""
    shift, info = cb_backbone_shift(st, resi, cb_target, chain)
    if shift is None:
        info['applied'] = False
        return info
    if max_shift is not None and info['magnitude'] > max_shift:
        info['applied'] = False
        info['reason'] = f'shift {info["magnitude"]} A exceeds max_shift {max_shift}'
        return info
    for nm in BB3:
        p0 = st.atom(resi, nm, chain)
        if p0 is not None:
            rx.add_reach(resi, p0 + shift, 0.0, tol, k=k, name=nm)
    info['applied'] = True
    return info


def residue_geometry(st, resi, chain='A', xyz=None):
    """Bond lengths, angles and chirality for one residue."""
    xyz = st.xyz if xyz is None else np.asarray(xyz, float)
    idx = st.residues.get((chain, int(resi)))
    if idx is None:
        return None
    at = {str(st.name[i]): xyz[i] for i in idx}
    if not all(a in at for a in ('N', 'CA', 'C', 'CB')):
        return None
    n, ca, c, cb = at['N'], at['CA'], at['C'], at['CB']
    return dict(
        resi=int(resi), resn=str(st.resn[idx[0]]),
        n_ca=round(float(np.linalg.norm(ca - n)), 3),
        ca_c=round(float(np.linalg.norm(c - ca)), 3),
        ca_cb=round(float(np.linalg.norm(cb - ca)), 3),
        n_ca_cb=round(_angle(n, ca, cb), 1),
        c_ca_cb=round(_angle(c, ca, cb), 1),
        improper=round(chirality(n, ca, c, cb), 1),
        chirality_error=round(chirality_error(n, ca, c, cb), 1),
        cb_off_cone=round(float(np.linalg.norm(ideal_cb(n, ca, c) - cb)), 3))


def check_residue(g, bond_tol=0.08, angle_tol=12.0, chir_tol=25.0):
    """Reasons this residue is not a valid L-amino acid. Empty list means fine."""
    if g is None:
        return ['missing backbone or CB']
    bad = []
    if abs(g['ca_cb'] - IDEAL['CA_CB']) > bond_tol:
        bad.append(f'CA-CB {g["ca_cb"]} (ideal {IDEAL["CA_CB"]})')
    if abs(g['n_ca'] - IDEAL['N_CA']) > bond_tol:
        bad.append(f'N-CA {g["n_ca"]} (ideal {IDEAL["N_CA"]})')
    if abs(g['n_ca_cb'] - IDEAL['N_CA_CB']) > angle_tol:
        bad.append(f'N-CA-CB {g["n_ca_cb"]} deg (ideal {IDEAL["N_CA_CB"]})')
    if g['chirality_error'] > chir_tol:
        bad.append(f'improper {g["improper"]:+} deg, {g["chirality_error"]} off L '
                   f'(ideal {L_IMPROPER:+.0f})')
    return bad


def snap_cb_to_cone(st, xyz, chain='A', resis=None):
    """Move every CB onto the ideal cone of its current N/CA/C."""
    xyz = np.asarray(xyz, float).copy()
    if resis is None:
        resis = [r for c, r in st.protein_res if c == chain]
    for r in resis:
        idx = st.residues.get((chain, int(r)))
        if idx is None:
            continue
        at = {str(st.name[i]): i for i in idx}
        if not all(a in at for a in ('N', 'CA', 'C', 'CB')):
            continue
        xyz[at['CB']] = ideal_cb(xyz[at['N']], xyz[at['CA']], xyz[at['C']])
    return xyz


def peptide_breaks(st, chain='A', xyz=None, max_cn=1.70, max_caca=4.20):
    """Sequential peptide junctions that are stretched or broken.

    Ideal C-N is ~1.33 A and CA-CA ~3.8 A. Defaults sit well above crystal
    noise (scaffold max C-N ~1.35) but catch near-breaks like 1.99 A that a
    hard 2.0 A cutoff used to miss.
    """
    xyz = st.xyz if xyz is None else np.asarray(xyz, float)
    resis = [r for c, r in st.protein_res if c == chain]
    out = []
    for a, b in zip(resis, resis[1:]):
        if b != a + 1:
            continue
        ia, ib = st.residues.get((chain, a)), st.residues.get((chain, b))
        if ia is None or ib is None:
            continue
        at_a = {str(st.name[i]): xyz[i] for i in ia}
        at_b = {str(st.name[i]): xyz[i] for i in ib}
        if 'C' not in at_a or 'N' not in at_b:
            continue
        d = float(np.linalg.norm(at_b['N'] - at_a['C']))
        dca = None
        if 'CA' in at_a and 'CA' in at_b:
            dca = float(np.linalg.norm(at_b['CA'] - at_a['CA']))
        if d > max_cn or (dca is not None and dca > max_caca):
            row = dict(resi_i=int(a), resi_j=int(b), c_n=round(d, 3))
            if dca is not None:
                row['ca_ca'] = round(dca, 3)
            out.append(row)
    return out


def validate_build(st, resis, chain='A', xyz=None, **tol):
    """Gate a built structure before it is written. Returns (ok, rows)."""
    rows, ok = [], True
    for r in resis:
        g = residue_geometry(st, r, chain, xyz)
        problems = check_residue(g, **tol)
        if g is None:
            g = dict(resi=int(r), resn='?')
        g['problems'] = problems
        if problems:
            ok = False
        rows.append(g)
    return ok, rows


def format_validation(rows, only_bad=False):
    L = [f'{"residue":<10} {"N-CA":>6} {"CA-CB":>6} {"N-CA-CB":>8} {"improper":>9} '
         f'{"CB off cone":>12}']
    for g in rows:
        if only_bad and not g.get('problems'):
            continue
        if 'n_ca' not in g:
            L.append(f'{g["resn"]}{g["resi"]:<6} incomplete')
            continue
        mark = '  <-- ' + '; '.join(g['problems']) if g.get('problems') else ''
        L.append(f'{g["resn"]}{g["resi"]:<6} {g["n_ca"]:6.3f} {g["ca_cb"]:6.3f} '
                 f'{g["n_ca_cb"]:8.1f} {g["improper"]:+9.1f} '
                 f'{g["cb_off_cone"]:12.3f}{mark}')
    L.append(f'ideal      {IDEAL["N_CA"]:6.3f} {IDEAL["CA_CB"]:6.3f} '
             f'{IDEAL["N_CA_CB"]:8.1f} {L_IMPROPER:+9.1f} {0.0:12.3f}')
    return '\n'.join(L)
