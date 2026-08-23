"""Shared theozyme placement: explore → accommodate → build → write PDBs.

Used by:
  - ``theozyme/prepare_placements.py``  — any theozyme, PDBs + report only
  - ``project_retroaldolase/prepare_deva.py`` — RA95 campaign (PDBs + dEVA YMLs)

Covalent vs non-covalent
------------------------
Both modes use the same geometry search: χ1/χ2 graft of the *frame* residue
(``role: "anchor"``) places the rigid theozyme XYZ, then satellite CBs are matched.

- **Covalent**: spec has ``covalent_to_ligand``. PDB gets a LINK; CST1 is the
  covalent tip–ligand distance; locked χ3+ come from the QM adduct geometry.
- **Non-covalent**: omit ``covalent_to_ligand``. No LINK. Anchor and satellites are
  placement / catalytic-geometry constraints (CB + tip distances), not a bond.
  CST ranking uses tip⋯ligand distances for all catalytic residues.

Not handled here (RA95 / campaign-specific): ``delta_mu``, reactant XYZ, electric
field maps, breaking ``partial_bonds`` into probe/axis, or writing dEVA YML configs.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.spatial import cKDTree

from .accommodate import backbone_bond_audit, propagate_sidechains_cb
from .explore import Explorer
from .graft import (add_graft_restraints, format_validation, peptide_breaks,
                    snap_cb_to_cone, validate_build)
from .loops import LoopSegment, displacement_demand, rama_fraction, tier_of
from .relax import RestrainedRelax
from .rigid import ideal_cb as _ideal_cb
from .sidechains import (TIP, apply_mutations, fit_ring, mutate,
                         solve_segment_shift)
from .structure import Structure, barrel_shell, pocket_center

R_OCC, SOFTNESS = 8.0, 1.0


def parse_resi_list(s):
    """Parse '51,83,110' or '50-60,180-190' into a list of ints."""
    out = []
    for tok in filter(None, (t.strip() for t in s.split(','))):
        if '-' in tok:
            lo, hi = tok.split('-')
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(tok))
    return out


def resolve_pocket_center(st, chain, pocket_pdb=None, scaffold_path=None):
    """Ligand centroid if available; else strand C-mouth; optional companion PDB."""
    cen, src = pocket_center(st, chain)
    if src.startswith('ligand'):
        return cen, src
    candidates = []
    if pocket_pdb:
        candidates.append(pocket_pdb)
    if scaffold_path:
        base = os.path.abspath(scaffold_path)
        if base.endswith('_scaffold.pdb'):
            candidates.append(base[:-len('_scaffold.pdb')] + '.pdb')
        elif base.endswith('_scaffold.cif'):
            candidates.append(base[:-len('_scaffold.cif')] + '.cif')
    seen = set()
    for path in candidates:
        path = os.path.abspath(path)
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        h = Structure(path)
        c2, s2 = pocket_center(h, chain)
        if s2.startswith('ligand'):
            return c2, f'{s2}@{os.path.basename(path)}'
    return cen, src


def occlusion(protein_xyz, lig_xyz, r_occ=R_OCC, softness=SOFTNESS):
    """Reproduces models/pocket_shape.py::_occlusion so target_occ can be calibrated."""
    d = np.sqrt(((protein_xyz[:, None, :] - lig_xyz[None, :, :]) ** 2).sum(-1))
    return float((1.0 / (1.0 + np.exp((d - r_occ) / softness))).sum())


def heavy_xyz(st, rec=None):
    return st.xyz[[i for i in range(len(st.xyz))
                   if str(st.elem[i]).upper() != 'H'
                   and (rec is None or str(st.rec[i]) == rec)]]


def kabsch_rmsd(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    t = Qc - R @ Pc
    return float(np.sqrt((((P @ R.T) + t - Q) ** 2).sum(1).mean()))


def is_covalent(spec):
    """True when the frame residue declares covalent_to_ligand."""
    return bool(getattr(spec.anchor, 'covalent', None))


def tip_partner(spec, sat):
    """Nearest ligand atom to a residue tip in the theozyme XYZ frame."""
    tip = TIP.get(sat.resn)
    if tip is None or tip not in sat.atoms:
        return None
    near = min(spec.lig_atoms, key=lambda nm: np.linalg.norm(
        spec.p(sat.atoms[tip]) - spec.p(spec.lig_atoms[nm])))
    want = float(np.linalg.norm(spec.p(sat.atoms[tip]) - spec.p(spec.lig_atoms[near])))
    return tip, near, want


def measure_cst(spec, sol, fin, sub, chain):
    """Built tip / covalent distances vs theozyme targets.

    Covalent: CST1 = covalent bond; CST2+ = satellite tips.
    Non-covalent: CST1 = anchor tip⋯ligand; CST2+ = satellite tips.
    """
    cst = {}
    if is_covalent(spec):
        cc = spec.anchor.covalent
        tgt = float(np.linalg.norm(spec.p(spec.anchor.atoms[cc['atom']])
                                   - spec.p(spec.lig_atoms[cc['ligand_atom']])))
        got = float(np.linalg.norm(
            fin.atom(sol['anchor'], cc['atom'], chain)
            - sub[list(spec.lig_atoms).index(cc['ligand_atom'])]))
        cst['CST1'] = dict(atoms=f"{cc['atom']}...{cc['ligand_atom']}",
                           target=round(tgt, 2), built=round(got, 2),
                           deviation=round(abs(got - tgt), 2), kind='covalent')
        start = 2
    else:
        tip_info = tip_partner(spec, spec.anchor)
        if tip_info is not None:
            tip, near, tgt = tip_info
            got = float(np.linalg.norm(
                fin.atom(sol['anchor'], tip, chain)
                - sub[list(spec.lig_atoms).index(near)]))
            cst['CST1'] = dict(
                atoms=f"{spec.anchor.resn}{sol['anchor']}.{tip}...{near}",
                target=round(tgt, 2), built=round(got, 2),
                deviation=round(abs(got - tgt), 2), kind='noncovalent_tip')
        start = 2
    for si, h in enumerate(sol['satellites'], start=start):
        sat = [x for x in spec.satellites if x.resn == h['resn']][0]
        tip_info = tip_partner(spec, sat)
        if tip_info is None:
            continue
        tip, near, tgt = tip_info
        got = float(np.linalg.norm(
            fin.atom(h['resi'], tip, chain)
            - sub[list(spec.lig_atoms).index(near)]))
        cst[f'CST{si}'] = dict(
            atoms=f"{sat.resn}{h['resi']}.{tip}...{near}",
            target=round(tgt, 2), built=round(got, 2),
            deviation=round(abs(got - tgt), 2), kind='satellite_tip')
    worst = max((v['deviation'] for v in cst.values()), default=99.0)
    return cst, worst


def write_placement_pdbs(spec, sol, fin, sub, chain, pdb_path, lig_path,
                         remarks=None):
    """Write complex + ligand-only PDBs. LINK only if covalent_to_ligand is set."""
    lig_extra = [dict(name=n, resn=spec.lig_resn, resi=901, xyz=q,
                      elem=(n[0] if n[0] in 'CNOSP' else 'C'), chain=chain)
                 for n, q in zip(spec.lig_atoms, sub)]
    fin.write(pdb_path, extra=lig_extra)
    S = fin.serial
    con = {}
    for x, y in spec.lig_bonds + spec.partial_bonds:
        ia, ib = S[(chain, 901, x)], S[(chain, 901, y)]
        con.setdefault(ia, []).append(ib)
        con.setdefault(ib, []).append(ia)
    links = []
    if is_covalent(spec):
        c = spec.anchor.covalent
        ia = S[(chain, sol['anchor'], c['atom'])]
        ib = S[(chain, 901, c['ligand_atom'])]
        con.setdefault(ia, []).append(ib)
        con.setdefault(ib, []).append(ia)
        d = float(np.linalg.norm(
            fin.atom(sol['anchor'], c['atom'], chain)
            - sub[list(spec.lig_atoms).index(c['ligand_atom'])]))
        links = [dict(
            a=dict(name=c['atom'], resn=spec.anchor.resn, chain=chain,
                   resi=sol['anchor']),
            b=dict(name=c['ligand_atom'], resn=spec.lig_resn, chain=chain,
                   resi=901),
            dist=d)]
    cat = [f'{chain}{sol["anchor"]}'] + [f'{chain}{h["resi"]}' for h in sol['satellites']]
    mode = 'covalent' if is_covalent(spec) else 'non-covalent'
    rem = list(remarks or [])
    rem += [
        f'theozyme {spec.name} ({mode} placement)',
        f'anchor {spec.anchor.resn}{sol["anchor"]} chi1={sol["chi1"]:.0f} chi2={sol["chi2"]:.0f}',
        'catalytic residues: ' + ' '.join(cat),
    ]
    if is_covalent(spec):
        rem.append('LINK declares the covalent theozyme-ligand bond; '
                   'CONECT gives ligand connectivity')
    else:
        rem.append('non-covalent: no protein-ligand LINK; CONECT is ligand-only; '
                   'anchor/satellites are geometric placement constraints')
    fin.write(pdb_path, extra=lig_extra, conect=con, links=links, remarks=rem)

    L = Structure()
    L.rec = np.array(['HETATM'] * len(sub))
    L.name = np.array(list(spec.lig_atoms))
    L.resn = np.array([spec.lig_resn] * len(sub))
    L.chain = np.array([chain] * len(sub))
    L.resi = np.array([901] * len(sub))
    L.occ = np.ones(len(sub))
    L.b = np.zeros(len(sub))
    L.elem = np.array([(n[0] if n[0] in 'CNOSP' else 'C') for n in spec.lig_atoms])
    L.xyz = sub
    L._index()
    L.write(lig_path)
    return cat


def build_one(spec, st, sol, mobile, chain, seeds=3, seed=0,
              accommodate=True, window=3, sat_tol=0.30, sat_mode='shift',
              sat_wells=(25.0, 35.0), max_shift=2.5,
              relaxer=None, strict_geometry=True, n_protpardelle_attempts=1):
    """Relax the backbone around the fixed substrate, then install the side chains."""
    X = sol['X']
    sub = sol['sub']
    segs = [LoopSegment(st, a, b, chain) for a, b in mobile] if mobile else []
    mob = sorted({r for a, b in mobile for r in range(a, b + 1)}) if mobile else []
    anchor_res = sol['anchor']
    sat_res = {h['resi'] for h in sol['satellites']}
    if accommodate:
        for r in sat_res:
            mob += [x for x in range(r - window, r + window + 1)
                    if (chain, x) in st.residues]
    mob = sorted({r for r in mob if r != anchor_res})
    per, mx = displacement_demand(st, sub, mob, chain) if mob else ({}, 0.0)
    active = [s for s in segs if any(r in per for r in s.res)]
    rng = np.random.default_rng(seed)
    cands = [st.xyz.copy()]
    for _ in range(seeds):
        if not active:
            break
        t2 = st.xyz.copy()
        ok = True
        for s in active:
            t2 = s.perturb(t2, rng, 10.0)
            t2, _, cl = s.close(t2)
            ok &= cl
        if ok:
            cands.append(t2)
    bb = st.backbone_idx()
    tsub = cKDTree(sub)
    best = None
    shift_info = {}
    anchor = sol['anchor']
    for c0 in cands:
        s2 = Structure()
        s2.__dict__.update(st.__dict__)
        s2.xyz = c0
        rx = RestrainedRelax(s2, mob, substrate_xyz=sub, chain=chain)
        for nm in ('N', 'CA', 'C'):
            p0 = st.atom(anchor, nm, chain)
            if p0 is not None:
                rx.add_reach(anchor, p0, 0.0, 0.10, k=200.0, name=nm)
        for h in sol['satellites']:
            sat = [s for s in spec.satellites if s.resn == h['resn']][0]
            tgt_cb = X[sat.atoms['CB'] - 1]
            if sat_mode == 'shift':
                tip = TIP.get(sat.resn)
                sh = None
                if tip and tip in sat.atoms:
                    nearn = min(spec.lig_atoms, key=lambda nm: np.linalg.norm(
                        spec.p(sat.atoms[tip]) - spec.p(spec.lig_atoms[nm])))
                    want = float(np.linalg.norm(spec.p(sat.atoms[tip])
                                                - spec.p(spec.lig_atoms[nearn])))
                    sh = solve_segment_shift(
                        st, h['resi'], sat.resn, tip, None, chain=chain,
                        well_tol=sat_wells, chi_step=3.0,
                        partner=sub[list(spec.lig_atoms).index(nearn)],
                        target_dist=want)
                if sh is not None and sh['magnitude'] <= max_shift:
                    for nm in ('N', 'CA', 'C'):
                        p0 = st.atom(h['resi'], nm, chain)
                        if p0 is not None:
                            rx.add_reach(h['resi'], p0 + sh['shift'], 0.0, 0.40,
                                         k=70.0, name=nm)
                    shift_info[h['resi']] = dict(
                        magnitude=sh['magnitude'], chi=sh['chi'],
                        off_rotamer=sh['off_rotamer'],
                        vector=[round(float(v), 3) for v in sh['shift']])
                else:
                    shift_info[h['resi']] = add_graft_restraints(
                        rx, st, h['resi'], tgt_cb, chain, tol=0.15, k=120.0,
                        max_shift=max_shift)
            elif sat_mode == 'cb':
                shift_info[h['resi']] = add_graft_restraints(
                    rx, st, h['resi'], tgt_cb, chain, tol=sat_tol, k=150.0,
                    max_shift=max_shift)
            elif sat_mode == 'ca':
                cb_nat = st.atom(h['resi'], 'CB', chain)
                ca_nat = st.atom(h['resi'], 'CA', chain)
                if cb_nat is not None and ca_nat is not None:
                    rx.add_reach(h['resi'], ca_nat + (tgt_cb - cb_nat), 0.0,
                                 sat_tol, k=150.0, name='CA')
            elif sat_mode == 'soft':
                shift_info[h['resi']] = add_graft_restraints(
                    rx, st, h['resi'], tgt_cb, chain, tol=0.40, k=25.0,
                    max_shift=max_shift)
        rr = rx.run_staged(target_clearance=3.20)
        d, _ = tsub.query(rr['xyz'][bb])
        rama = float(np.mean([rama_fraction(st, s, rr['xyz']) for s in active])) if active else 1.0
        sc = float(d.min()) - 0.4 * max(0, rr['max_disp_rigid'] - 0.30) - 0.3 * (1 - rama)
        if best is None or sc > best[0]:
            best = (sc, rr, rama)
    _, rr, rama = best

    rr_base = {**rr, 'xyz': snap_cb_to_cone(st, rr['xyz'], chain)}

    cat_resis = [anchor] + [h['resi'] for h in sol['satellites']]
    n_try = max(1, int(n_protpardelle_attempts if relaxer is not None else 1))
    last = None
    best_ok = None
    for attempt in range(n_try):
        rr = {**rr_base, 'xyz': rr_base['xyz'].copy()}
        pp_info = None
        if relaxer is not None:
            s3 = Structure()
            s3.__dict__.update(st.__dict__)
            s3.xyz = rr['xyz']
            try:
                rr = {**rr}
                rr['xyz'], pp_info = relaxer.relax_structure(
                    s3, cat_resis, chain=chain, seed=seed + 17 * attempt)
                rr['xyz'] = snap_cb_to_cone(st, rr['xyz'], chain)
                if n_try > 1:
                    print(f'      protpardelle attempt {attempt+1}/{n_try}'
                          f'  bb_rmsd={pp_info.get("bb_rmsd")}')
            except Exception as e:
                print(f'      protpardelle skipped: {type(e).__name__}: {e}')

        Xf = propagate_sidechains_cb(st, st.xyz, rr['xyz'], chain)
        rel = Structure()
        rel.__dict__.update(st.__dict__)
        rel.xyz = Xf
        muts = []
        diag = {}
        a = spec.anchor
        acoord = {k: X[v - 1] for k, v in a.atoms.items()}
        for nm in ('N', 'CA', 'C'):
            acoord[nm] = rel.atom(anchor, nm, chain)
        acoord['CB'] = _ideal_cb(acoord['N'], acoord['CA'], acoord['C'])
        muts.append(mutate(rel, anchor, a.resn, acoord))
        diag[f'{a.resn}{anchor}'] = dict(source='QM coordinates verbatim')
        fit_ok = True
        for h in sol['satellites']:
            sat = [s for s in spec.satellites if s.resn == h['resn']][0]
            tgt = {k: X[v - 1] for k, v in sat.atoms.items() if k != 'CB'}
            tip = TIP.get(sat.resn)
            tip_target = None
            if tip and tip in sat.atoms:
                near = min(spec.lig_atoms, key=lambda nm: np.linalg.norm(
                    spec.p(sat.atoms[tip]) - spec.p(spec.lig_atoms[nm])))
                want = float(np.linalg.norm(
                    spec.p(sat.atoms[tip]) - spec.p(spec.lig_atoms[near])))
                tip_target = (tip, sub[list(spec.lig_atoms).index(near)], want)
            fit = fit_ring(rel, h['resi'], sat.resn, tgt, chain=chain, step=3.0,
                           wells=True, tip_target=tip_target)
            if fit is None:
                fit_ok = False
                break
            n_, ca_, c_ = (rel.atom(h['resi'], nm, chain) for nm in ('N', 'CA', 'C'))
            if all(v is not None for v in (n_, ca_, c_)):
                fit = dict(fit)
                fit['CB'] = _ideal_cb(n_, ca_, c_)
            muts.append(mutate(rel, h['resi'], sat.resn, fit))
            diag[f'{sat.resn}{h["resi"]}'] = dict(
                ring_rmsd=round(fit['_ring_rmsd'], 2),
                was=h['wt'], cb_dev=round(h['cb_dev'], 2),
                chi=[round(x, 1) for x in fit['_chi']],
                off_rotamer=fit.get('_off_rotamer'),
                tip_dist=fit.get('_tip_dist'),
                tip_dev=fit.get('_tip_dev'),
                segment_shift=shift_info.get(h['resi']))
        if not fit_ok:
            print(f'      attempt {attempt+1}/{n_try}: fit_ring failed')
            continue

        fin = apply_mutations(rel, muts, chain)
        worst = 0.0
        for tag, r_ in ([(f'{a.resn}{anchor}', anchor)]
                        + [(f'{[s for s in spec.satellites if s.resn==h["resn"]][0].resn}'
                            f'{h["resi"]}', h['resi']) for h in sol['satellites']]):
            ca, cb = fin.atom(r_, 'CA', chain), fin.atom(r_, 'CB', chain)
            d = float(np.linalg.norm(ca - cb))
            diag[tag]['CA_CB'] = round(d, 2)
            diag[tag]['strained'] = bool(abs(d - 1.53) > 0.15)
            worst = max(worst, abs(d - 1.53))
        geom_ok, geom_rows = validate_build(fin, cat_resis, chain)
        breaks = peptide_breaks(fin, chain)
        if breaks:
            geom_ok = False
        for g in geom_rows:
            tag_ = f"{g.get('resn', '?')}{g['resi']}"
            if tag_ in diag:
                diag[tag_]['geometry'] = {k: v for k, v in g.items()
                                          if k not in ('resi', 'resn')}
        all_r = [r for c, r in fin.protein_res if c == chain]
        _, all_rows = validate_build(fin, all_r, chain)
        n_bad = sum(1 for g in all_rows if g.get('problems'))
        bonds = backbone_bond_audit(fin, fin.xyz, chain)
        bb_strain = float(bonds.get('worst_dev', 99.0))
        cand = dict(struct=fin, sub=sub, relax=rr, rama=rama, diag=diag,
                    worst_ca_cb_dev=round(worst, 2),
                    residues_moved={str(k): round(v, 2) for k, v in per.items()},
                    tier=tier_of(mx),
                    protpardelle=pp_info, geometry_ok=bool(geom_ok),
                    geometry=[g for g in geom_rows],
                    peptide_breaks=breaks,
                    backbone_bonds=bonds,
                    bb_strain=round(bb_strain, 3),
                    n_residues_off_ideal=int(n_bad),
                    protpardelle_attempt=attempt + 1)
        last = cand
        print(f'      attempt {attempt+1}/{n_try}: bb_strain={bb_strain:.3f} A'
              f'  geom={"ok" if geom_ok else "FAIL"}')
        if geom_ok or not strict_geometry:
            if breaks:
                print('      WARNING stretched/broken peptide: '
                      + ', '.join(
                          f"{b['resi_i']}-{b['resi_j']} C-N={b['c_n']:.2f}"
                          + (f" CA-CA={b['ca_ca']:.2f}" if 'ca_ca' in b else '')
                          for b in breaks))
            if n_bad:
                print(f'      WARNING {n_bad}/{len(all_rows)} residues outside ideal geometry')
            if not geom_ok:
                if any(g.get('problems') for g in geom_rows):
                    print('      catalytic residues are NOT valid L-amino acids:')
                    print(format_validation(geom_rows, only_bad=True))
                if breaks:
                    print('      discontinuous backbone: '
                          + ', '.join(
                              f"{b['resi_i']}-{b['resi_j']} C-N={b['c_n']:.2f}"
                              + (f" CA-CA={b['ca_ca']:.2f}" if 'ca_ca' in b else '')
                              for b in breaks))
                if not strict_geometry:
                    print('      kept anyway (--allow-bad-geometry)')
            if geom_ok and (best_ok is None or bb_strain < best_ok['bb_strain']):
                best_ok = cand
                print(f'      -> best so far (bb_strain={bb_strain:.3f} A)')
            continue
        print(f'      attempt {attempt+1}/{n_try}: geometry FAILED')
        if any(g.get('problems') for g in geom_rows):
            print(format_validation(geom_rows, only_bad=True))
        if breaks:
            print('      discontinuous backbone: '
                  + ', '.join(
                      f"{b['resi_i']}-{b['resi_j']} C-N={b['c_n']:.2f}"
                      + (f" CA-CA={b['ca_ca']:.2f}" if 'ca_ca' in b else '')
                      for b in breaks))

    if best_ok is not None:
        return best_ok
    if last is not None and not strict_geometry:
        return last
    if last is not None:
        print('      REJECTED after all attempts (pass --allow-bad-geometry to keep it anyway)')
    return None


def resolve_satellite_hosts(st, chain, scaffold_path, barrel_shell_r=None,
                            satellite_positions='', pocket_pdb='',
                            suggest_shell=None):
    """Return (satpos or None, pocket_center info). Exit via SystemExit on --suggest-shell."""
    cen = cen_src = None
    if barrel_shell_r is not None or suggest_shell is not None:
        cen, cen_src = resolve_pocket_center(
            st, chain, pocket_pdb=(pocket_pdb or None), scaffold_path=scaffold_path)
        print(f'  pocket center: {cen_src}  '
              f'[{cen[0]:.2f}, {cen[1]:.2f}, {cen[2]:.2f}]')

    satpos = None
    if barrel_shell_r is not None:
        shell, segs = barrel_shell(st, cen, chain, radius=barrel_shell_r)
        satpos = [x['resi'] for x in shell]
        print(f'  barrel shell: {len(segs)} strands -> {len(satpos)} satellite positions '
              f'within {barrel_shell_r} A, loops excluded')
        for x in shell:
            print(f"     {x['wt']}{x['resi']:<5d} strand {x['strand']:>8s}  "
                  f"CB-pocket {x['cb_pocket']:5.1f}  toward {x['toward']:+.2f}")
    if satellite_positions:
        declared = parse_resi_list(satellite_positions)
        declared = [r for r in declared if (chain, r) in st.residues]
        if satpos is None:
            satpos = declared
            print(f'  satellite hosts restricted to {len(satpos)} declared positions: '
                  + ','.join(str(r) for r in satpos))
        else:
            extra = sorted(set(declared) - set(satpos))
            satpos = sorted(set(satpos) | set(declared))
            print(f'  satellite hosts: barrel shell ∪ {len(declared)} declared '
                  f'-> {len(satpos)} positions'
                  + (f' (added {",".join(str(r) for r in extra)})' if extra else ''))

    if suggest_shell is not None:
        rows = []
        for c, r in st.protein_res:
            if c != chain:
                continue
            ca, cb = st.atom(r, 'CA', c), st.atom(r, 'CB', c)
            if cb is None:
                continue
            d = float(np.linalg.norm(cb - cen))
            if d > suggest_shell:
                continue
            v1, v2 = cb - ca, cen - cb
            pin = float(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2))
            rows.append((d, int(r), str(st.resname(r, c)), pin))
        rows.sort()
        print(f'\n  {len(rows)} positions with CB within {suggest_shell} A of the pocket '
              f'({cen_src}):')
        for d, r, wt, pin in rows:
            mark = '  <-' if pin >= 0 else ''
            print(f'    {wt}{r:<5d} CB-pocket {d:5.1f}  CA->CB.CB->pocket {pin:+.2f}{mark}')
        inward = [r for _, r, _, pin in rows if pin >= 0]
        print('\n  --satellite-positions "' + ','.join(str(r) for r in inward) + '"')
        raise SystemExit(0)

    return satpos, (cen, cen_src)


def explore_and_filter(spec, st, chain, anchors, mob_flat, satpos,
                       chi_step, max_cb_dev, min_occ, max_occ, no_accommodate,
                       candidates_path=None):
    """Run Explorer, dedupe by residue assignment, filter by occlusion."""
    print(f'[1/4] exploring {len(anchors)} anchors x {int(360 / chi_step) ** 2} grafts')
    mode = 'covalent' if is_covalent(spec) else 'non-covalent'
    print(f'  mode: {mode}  (anchor is the frame residue for χ1/χ2 search)')
    ex = Explorer(spec, st, chain, mobile_resis=mob_flat, max_cb_dev=max_cb_dev,
                  satellite_positions=satpos)
    sols = ex.run(anchors, chi_step=chi_step)
    if not sols:
        print('  no placement satisfies the theozyme on these anchors.')
        return None
    prot = heavy_xyz(st, rec='ATOM')
    for s in sols:
        s['occ'] = occlusion(prot, s['sub']) / len(s['sub'])
        s['cb_dev'] = max([h['cb_dev'] for h in s['satellites']], default=0.0)
    accom_budget = 0.0 if no_accommodate else 1.2
    for s_ in sols:
        s_['cb_residual'] = max(0.0, s_['cb_dev'] - accom_budget)
        s_['needs_accommodation'] = bool(s_['cb_dev'] > 0.6)
        s_['pre_cst'] = float(s_['cb_dev'])
    sols.sort(key=lambda z: (z['pre_cst'], -z['occ']))
    print(f'  {len(sols)} complete solutions; best satellite CB deviation '
          f'{sols[0]["cb_dev"]:.2f} A')

    uniq = {}
    for s_ in sols:
        key = (s_['anchor'],
               tuple(sorted((h['resn'], h['resi']) for h in s_['satellites'])))
        if key not in uniq or s_['pre_cst'] < uniq[key]['pre_cst'] or (
                s_['pre_cst'] == uniq[key]['pre_cst'] and s_['occ'] > uniq[key]['occ']):
            uniq[key] = s_
    combos = sorted(uniq.values(), key=lambda z: (z['pre_cst'], -z['occ']))
    n_before = len(combos)
    if min_occ is not None:
        combos = [c for c in combos if c['occ'] >= min_occ]
    if max_occ is not None:
        combos = [c for c in combos if c['occ'] <= max_occ]
    if len(combos) != n_before:
        lo = min_occ if min_occ is not None else float('-inf')
        hi = max_occ if max_occ is not None else float('inf')
        print(f'  occlusion filter [{lo}, {hi}] kept {len(combos)}/{n_before} assignments')
    if not combos:
        print('  nothing survives the occlusion filter -- loosen --min-occ/--max-occ')
        return None
    print(f'  {len(combos)} distinct residue assignments')

    if candidates_path:
        os.makedirs(os.path.dirname(os.path.abspath(candidates_path)) or '.', exist_ok=True)
        json.dump([dict(
            rank=i,
            anchor=int(c['anchor']), anchor_resn=spec.anchor.resn,
            anchor_wt=c['anchor_wt'], chi1=c['chi1'], chi2=c['chi2'],
            satellites=[dict(resn=h['resn'], resi=int(h['resi']), was=h['wt'],
                             cb_dev=round(h['cb_dev'], 2),
                             cacbcg=round(h['cacbcg'], 1)) for h in c['satellites']],
            max_cb_dev=round(c['cb_dev'], 2),
            pre_cst=round(c['pre_cst'], 2),
            cb_residual_after_accommodation=round(c['cb_residual'], 2),
            needs_accommodation=c['needs_accommodation'],
            occlusion=round(c['occ'], 1),
            n_graft_solutions=sum(
                1 for s2 in sols
                if s2['anchor'] == c['anchor'] and
                tuple(sorted((h['resn'], h['resi']) for h in s2['satellites'])) ==
                tuple(sorted((h['resn'], h['resi']) for h in c['satellites']))),
            covalent=is_covalent(spec),
        ) for i, c in enumerate(combos)], open(candidates_path, 'w'), indent=2)
        print(f'  -> {candidates_path}')

    print(f'\n  candidates, ordered by pre-CST (max satellite CBdev), then occlusion.')
    print(f'  This is BUILD ORDER. Final rank uses built worstCST after --build-max.')
    print(f'  Occlusion floor: {min_occ}')
    for i, c in enumerate(combos[:12]):
        sat = ' '.join(f"{h['resn']}{h['resi']}(was {h['wt']})" for h in c['satellites'])
        flag = ' [needs accommodation]' if c['needs_accommodation'] else ''
        print(f"    {i:3d}. {spec.anchor.resn}{c['anchor']}(was {c['anchor_wt']}) + {sat}"
              f"  preCST {c['pre_cst']:.2f}  CBdev {c['cb_dev']:.2f}  "
              f"occ {c['occ']:.1f}{flag}")
    return combos


def build_ranked_pdbs(spec, st, chain, combos, mobile, out_dir, name,
                      build_max=25, top=None, accommodate=True, sat_window=3,
                      sat_mode='shift', max_shift=2.5, relaxer=None,
                      allow_bad_geometry=False, n_protpardelle_attempts=1,
                      target_occ=None, min_occ=None):
    """Build placements, write ranked PDBs under out_dir, return report list.

    ``top`` limits how many ranked files are kept (None = keep all built).
    """
    os.makedirs(out_dir, exist_ok=True)
    report = []
    built = 0
    for k, sol in enumerate(combos):
        if built >= build_max:
            break
        print(f'[2/4] building placement {k}: anchor {spec.anchor.resn}{sol["anchor"]} '
              f'+ ' + ' '.join(f'{h["resn"]}{h["resi"]}' for h in sol['satellites']))
        B = build_one(spec, st, sol, mobile, chain,
                      accommodate=accommodate, window=sat_window,
                      sat_mode=sat_mode, max_shift=max_shift,
                      relaxer=relaxer, strict_geometry=not allow_bad_geometry,
                      n_protpardelle_attempts=n_protpardelle_attempts)
        if B is None:
            continue
        fin = B['struct']
        sub = B['sub']
        tag = f'{name}_b{k}'
        pdb_path = os.path.join(out_dir, f'{tag}.pdb')
        lig_path = os.path.join(out_dir, f'{tag}_ligand.pdb')
        cat = write_placement_pdbs(
            spec, sol, fin, sub, chain, pdb_path, lig_path,
            remarks=[f'prepared by theozyme.placements | {name}'])

        fin2 = Structure(pdb_path)
        built_occ = occlusion(heavy_xyz(fin2, rec='ATOM'), sub) / len(sub)
        if target_occ is not None:
            occ = target_occ
            print(f'[3/4] occlusion: built {built_occ:.1f}, target {occ:.1f} '
                  f'({occ - built_occ:+.1f} to close by design)')
        else:
            occ = built_occ
            print(f'[3/4] occlusion: built {built_occ:.1f}')

        # Kabsch self-check (ligand theozyme_map)
        tmap = []
        P, Q = [], []
        names = list(spec.lig_atoms)
        for nm in names:
            tmap.append(f'{spec.lig_atoms[nm]-1}:{spec.lig_resn}:901:{nm}')
            P.append(spec.p(spec.lig_atoms[nm]))
            Q.append(sub[names.index(nm)])
        P, Q = np.array(P), np.array(Q)
        rmsd = kabsch_rmsd(P, Q)
        assert rmsd < 0.05, f'theozyme_map self-check failed: RMSD {rmsd:.3f} A'
        print(f'      theozyme_map RMSD = {rmsd:.4f} A')

        cst_built, worst_cst = measure_cst(spec, sol, fin2, sub, chain)
        print(f'[4/4] worstCST = {worst_cst:.2f} A  ({len(cst_built)} restraints)')

        report.append(dict(
            tag=tag, pre_rank=k, anchor=sol['anchor'],
            chi1=sol['chi1'], chi2=sol['chi2'],
            satellites=[{k_: v for k_, v in h.items()} for h in sol['satellites']],
            catalytic=cat, theozyme_map=tmap,
            covalent=is_covalent(spec),
            cst_built=cst_built, worst_cst_deviation=round(worst_cst, 2),
            target_occ=round(occ, 1), built_occ=round(built_occ, 1),
            map_rmsd=round(rmsd, 4), tier=B['tier'],
            residues_moved=B['residues_moved'], sidechains=B['diag'],
            worst_CA_CB_deviation=B['worst_ca_cb_dev'],
            geometry_ok=B.get('geometry_ok'), geometry=B.get('geometry'),
            peptide_breaks=B.get('peptide_breaks'),
            bb_strain=B.get('bb_strain'),
            backbone_bonds=B.get('backbone_bonds'),
            n_residues_off_ideal=B.get('n_residues_off_ideal'),
            protpardelle=B.get('protpardelle'),
            protpardelle_attempt=B.get('protpardelle_attempt'),
            relax={k_: round(v, 3) for k_, v in B['relax'].items() if k_ != 'xyz'},
            pdb=pdb_path, ligand=lig_path))
        built += 1

    report.sort(key=lambda r: (r['worst_cst_deviation'],
                               r.get('bb_strain', 99.0),
                               r['worst_CA_CB_deviation']))
    for i, r in enumerate(report):
        r['built_rank'] = i

    def _mv(src, dst):
        if os.path.exists(src) and src != dst:
            os.replace(src, dst)

    staged = []
    for i, r in enumerate(report):
        newtag = f'{name}_rank{i}'
        staged.append((r, newtag,
                       os.path.join(out_dir, f'{newtag}.pdb'),
                       os.path.join(out_dir, f'{newtag}_ligand.pdb')))
    for r, newtag, npdb, nlig in staged:
        _mv(r['pdb'], npdb + '.tmp')
        _mv(r['ligand'], nlig + '.tmp')
    keep_n = len(staged) if top is None else min(top, len(staged))
    for i, (r, newtag, npdb, nlig) in enumerate(staged):
        if i < keep_n:
            _mv(npdb + '.tmp', npdb)
            _mv(nlig + '.tmp', nlig)
            r.update(tag=newtag, pdb=npdb, ligand=nlig)
        else:
            for p in (npdb + '.tmp', nlig + '.tmp'):
                if os.path.exists(p):
                    os.remove(p)
            # remove pre-rank staging files if still present
            for p in (r.get('pdb'), r.get('ligand')):
                if p and os.path.exists(p) and '_b' in os.path.basename(p):
                    os.remove(p)
            r.update(tag=newtag, pdb=None, ligand=None)

    report_path = os.path.join(out_dir, f'{name}_report.json')
    # JSON-safe: drop huge nested floats already rounded; peptide_breaks ok
    json.dump(report, open(report_path, 'w'), indent=2, default=str)
    print(f'\n  report -> {report_path}')
    print(f'  final rank: worstCST, then strain, then CA-CB'
          + (f'  |  occ floor was {min_occ}' if min_occ is not None else ''))
    print(f"{'rank':>4s} {'pre':>4s} {'assignment':>22s} {'strain':>7s} "
          f"{'worstCST':>9s} {'CA-CB':>6s} {'occ':>6s} {'geom':>5s} {'pp':>3s}")
    for r in report:
        sat = ' '.join(f"{h['resn']}{h['resi']}" for h in r['satellites'])
        print(f"{r['built_rank']:>4d} {r.get('pre_rank', '-'):>4} "
              f"{spec.anchor.resn + str(r['anchor']) + '+' + sat:>22s} "
              f"{r.get('bb_strain', float('nan')):>7.3f} "
              f"{r['worst_cst_deviation']:>9.2f} {r['worst_CA_CB_deviation']:>6.2f} "
              f"{r['built_occ']:>6.1f} "
              f"{('ok' if r.get('geometry_ok') else 'BAD'):>5s} "
              f"{r.get('protpardelle_attempt', '-'):>3}")
    kept = [r for r in report if r.get('pdb')]
    print(f'\n{len(kept)} PDB(s) written under {out_dir}/{name}_rank<i>.pdb')
    return report
