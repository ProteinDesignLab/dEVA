#!/usr/bin/env python3
"""Prepare scaffold + theozyme placements for a dEVA run."""
import sys, os, json, argparse, time
_DEVA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _DEVA_ROOT)
import numpy as np
from scipy.spatial import cKDTree
from theozyme.structure import Structure, barrel_shell, pocket_center
from theozyme.spec import TheozymeSpec
from theozyme.explore import Explorer
from theozyme.loops import LoopSegment, displacement_demand, tier_of, rama_fraction
from theozyme.relax import RestrainedRelax, propagate_sidechains
from theozyme.sidechains import mutate, apply_mutations, fit_ring, TIP, solve_segment_shift
from theozyme.graft import (add_graft_restraints, validate_build, format_validation,
                            cb_backbone_shift, snap_cb_to_cone, peptide_breaks)
from theozyme.accommodate import backbone_bond_audit

R_OCC, SOFTNESS = 8.0, 1.0

def parse_resi_list(s):
    """Parse '51,83,110' or '50-60,180-190' into a list of ints."""
    out=[]
    for tok in filter(None, (t.strip() for t in s.split(','))):
        if '-' in tok:
            lo,hi=tok.split('-'); out += list(range(int(lo),int(hi)+1))
        else:
            out.append(int(tok))
    return out

def resolve_pocket_center(st, chain, pocket_pdb=None, scaffold_path=None):
    """Ligand centroid if available; else strand C-mouth; optional companion PDB."""
    cen, src = pocket_center(st, chain)
    if src.startswith('ligand'):
        return cen, src
    candidates=[]
    if pocket_pdb:
        candidates.append(pocket_pdb)
    if scaffold_path:
        base=os.path.abspath(scaffold_path)
        if base.endswith('_scaffold.pdb'):
            candidates.append(base[:-len('_scaffold.pdb')]+'.pdb')
        elif base.endswith('_scaffold.cif'):
            candidates.append(base[:-len('_scaffold.cif')]+'.cif')
    seen=set()
    for path in candidates:
        path=os.path.abspath(path)
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        h=Structure(path)
        c2, s2 = pocket_center(h, chain)
        if s2.startswith('ligand'):
            return c2, f'{s2}@{os.path.basename(path)}'
    return cen, src

def occlusion(protein_xyz, lig_xyz, r_occ=R_OCC, softness=SOFTNESS):
    """Reproduces models/pocket_shape.py::_occlusion so target_occ can be calibrated."""
    d = np.sqrt(((protein_xyz[:, None, :] - lig_xyz[None, :, :])**2).sum(-1))
    return float((1.0/(1.0+np.exp((d-r_occ)/softness))).sum())

def heavy_xyz(st, rec=None):
    return st.xyz[[i for i in range(len(st.xyz))
                   if str(st.elem[i]).upper() != 'H'
                   and (rec is None or str(st.rec[i]) == rec)]]

def kabsch_rmsd(P, Q):
    Pc,Qc=P.mean(0),Q.mean(0); H=(P-Pc).T@(Q-Qc); U,S,Vt=np.linalg.svd(H)
    R=Vt.T@np.diag([1,1,np.sign(np.linalg.det(Vt.T@U.T))])@U.T
    t=Qc-R@Pc
    return float(np.sqrt((((P@R.T)+t-Q)**2).sum(1).mean()))

def build_one(spec, st, sol, mobile, chain, seeds=3, seed=0,
              accommodate=True, window=3, sat_tol=0.30, sat_mode='shift',
              sat_wells=(25.0,35.0), max_shift=2.5,
              relaxer=None, strict_geometry=True, n_protpardelle_attempts=1):
    """Relax the backbone around the fixed substrate, then install the side chains."""
    X = sol['X']; sub = sol['sub']
    segs = [LoopSegment(st,a,b,chain) for a,b in mobile] if mobile else []
    mob = sorted({r for a,b in mobile for r in range(a,b+1)}) if mobile else []
    anchor_res = sol['anchor']
    sat_res = {h['resi'] for h in sol['satellites']}
    if accommodate:
        for r in sat_res:
            mob += [x for x in range(r-window, r+window+1)
                    if (chain, x) in st.residues]
    mob = sorted({r for r in mob if r != anchor_res})
    per,mx = displacement_demand(st, sub, mob, chain) if mob else ({},0.0)
    active=[s for s in segs if any(r in per for r in s.res)]
    rng=np.random.default_rng(seed)
    cands=[st.xyz.copy()]
    for _ in range(seeds):
        if not active: break
        t2=st.xyz.copy(); ok=True
        for s in active:
            t2=s.perturb(t2,rng,10.0); t2,_,cl=s.close(t2); ok &= cl
        if ok: cands.append(t2)
    bb=st.backbone_idx(); tsub=cKDTree(sub); best=None; shift_info={}
    anchor=sol['anchor']
    for c0 in cands:
        s2=Structure(); s2.__dict__.update(st.__dict__); s2.xyz=c0
        rx=RestrainedRelax(s2, mob, substrate_xyz=sub, chain=chain)
        for nm in ('N','CA','C'):
            p0=st.atom(anchor, nm, chain)
            if p0 is not None:
                rx.add_reach(anchor, p0, 0.0, 0.10, k=200.0, name=nm)
        for h in sol['satellites']:
            sat=[s for s in spec.satellites if s.resn==h['resn']][0]
            tgt_cb = X[sat.atoms['CB']-1]
            if sat_mode == 'shift':
                tip=TIP.get(sat.resn)
                sh=None
                if tip and tip in sat.atoms:
                    nearn=min(spec.lig_atoms, key=lambda nm: np.linalg.norm(
                        spec.p(sat.atoms[tip])-spec.p(spec.lig_atoms[nm])))
                    want=float(np.linalg.norm(spec.p(sat.atoms[tip])
                                              -spec.p(spec.lig_atoms[nearn])))
                    sh=solve_segment_shift(st, h['resi'], sat.resn, tip, None, chain=chain,
                                           well_tol=sat_wells, chi_step=3.0,
                                           partner=sub[list(spec.lig_atoms).index(nearn)],
                                           target_dist=want)
                if sh is not None and sh['magnitude'] <= max_shift:
                    for nm in ('N','CA','C'):
                        p0=st.atom(h['resi'], nm, chain)
                        if p0 is not None:
                            rx.add_reach(h['resi'], p0+sh['shift'], 0.0, 0.40,
                                         k=70.0, name=nm)
                    shift_info[h['resi']]=dict(magnitude=sh['magnitude'],
                        chi=sh['chi'], off_rotamer=sh['off_rotamer'],
                        vector=[round(float(v),3) for v in sh['shift']])
                else:
                    shift_info[h['resi']]=add_graft_restraints(
                        rx, st, h['resi'], tgt_cb, chain, tol=0.15, k=120.0,
                        max_shift=max_shift)
            elif sat_mode == 'cb':
                shift_info[h['resi']]=add_graft_restraints(
                    rx, st, h['resi'], tgt_cb, chain, tol=sat_tol, k=150.0,
                    max_shift=max_shift)
            elif sat_mode == 'ca':
                cb_nat = st.atom(h['resi'],'CB',chain); ca_nat = st.atom(h['resi'],'CA',chain)
                if cb_nat is not None and ca_nat is not None:
                    rx.add_reach(h['resi'], ca_nat+(tgt_cb-cb_nat), 0.0, sat_tol, k=150.0, name='CA')
            elif sat_mode == 'soft':
                shift_info[h['resi']]=add_graft_restraints(
                    rx, st, h['resi'], tgt_cb, chain, tol=0.40, k=25.0,
                    max_shift=max_shift)
        rr=rx.run_staged(target_clearance=3.20)
        d,_=tsub.query(rr['xyz'][bb])
        rama=float(np.mean([rama_fraction(st,s,rr['xyz']) for s in active])) if active else 1.0
        sc = float(d.min()) - 0.4*max(0, rr['max_disp_rigid']-0.30) - 0.3*(1-rama)
        if best is None or sc>best[0]: best=(sc, rr, rama)
    _, rr, rama = best

    rr_base = {**rr, 'xyz': snap_cb_to_cone(st, rr['xyz'], chain)}

    from theozyme.accommodate import propagate_sidechains_cb
    cat_resis=[anchor]+[h['resi'] for h in sol['satellites']]
    n_try = max(1, int(n_protpardelle_attempts if relaxer is not None else 1))
    last = None
    best_ok = None  # lowest-strain geometry-ok attempt across samples
    for attempt in range(n_try):
        rr = {**rr_base, 'xyz': rr_base['xyz'].copy()}
        pp_info=None
        if relaxer is not None:
            s3=Structure(); s3.__dict__.update(st.__dict__); s3.xyz=rr['xyz']
            try:
                rr={**rr}
                rr['xyz'], pp_info = relaxer.relax_structure(
                    s3, cat_resis, chain=chain, seed=seed + 17 * attempt)
                rr['xyz'] = snap_cb_to_cone(st, rr['xyz'], chain)
                if n_try > 1:
                    print(f'      protpardelle attempt {attempt+1}/{n_try}'
                          f'  bb_rmsd={pp_info.get("bb_rmsd")}')
            except Exception as e:
                print(f'      protpardelle skipped: {type(e).__name__}: {e}')

        Xf = propagate_sidechains_cb(st, st.xyz, rr['xyz'], chain)
        rel = Structure(); rel.__dict__.update(st.__dict__); rel.xyz = Xf
        muts=[]; diag={}
        a=spec.anchor
        acoord={k: X[v-1] for k,v in a.atoms.items()}
        for nm in ('N','CA','C'): acoord[nm]=rel.atom(anchor,nm,chain)
        from theozyme.rigid import ideal_cb as _ideal_cb
        acoord['CB'] = _ideal_cb(acoord['N'], acoord['CA'], acoord['C'])
        muts.append(mutate(rel, anchor, a.resn, acoord))
        diag[f'{a.resn}{anchor}'] = dict(source='QM coordinates verbatim')
        fit_ok = True
        for h in sol['satellites']:
            sat=[s for s in spec.satellites if s.resn==h['resn']][0]
            tgt={k: X[v-1] for k,v in sat.atoms.items() if k!='CB'}
            tip=TIP.get(sat.resn); tip_target=None
            if tip and tip in sat.atoms:
                near=min(spec.lig_atoms, key=lambda nm: np.linalg.norm(
                    spec.p(sat.atoms[tip])-spec.p(spec.lig_atoms[nm])))
                want=float(np.linalg.norm(spec.p(sat.atoms[tip])-spec.p(spec.lig_atoms[near])))
                tip_target=(tip, sub[list(spec.lig_atoms).index(near)], want)
            fit=fit_ring(rel, h['resi'], sat.resn, tgt, chain=chain, step=3.0,
                         wells=True, tip_target=tip_target)
            if fit is None:
                fit_ok = False
                break
            n_,ca_,c_ = (rel.atom(h['resi'], nm, chain) for nm in ('N','CA','C'))
            if all(v is not None for v in (n_, ca_, c_)):
                fit = dict(fit)
                fit['CB'] = _ideal_cb(n_, ca_, c_)
            muts.append(mutate(rel, h['resi'], sat.resn, fit))
            diag[f'{sat.resn}{h["resi"]}']=dict(ring_rmsd=round(fit['_ring_rmsd'],2),
                                                was=h['wt'], cb_dev=round(h['cb_dev'],2),
                                                chi=[round(x,1) for x in fit['_chi']],
                                                off_rotamer=fit.get('_off_rotamer'),
                                                tip_dist=fit.get('_tip_dist'),
                                                tip_dev=fit.get('_tip_dev'),
                                                segment_shift=shift_info.get(h['resi']))
        if not fit_ok:
            print(f'      attempt {attempt+1}/{n_try}: fit_ring failed')
            continue

        fin = apply_mutations(rel, muts, chain)
        worst=0.0
        for tag,r_ in [(f'{a.resn}{anchor}',anchor)]+[ (f'{[s for s in spec.satellites if s.resn==h["resn"]][0].resn}{h["resi"]}', h['resi']) for h in sol['satellites']]:
            ca,cb = fin.atom(r_,'CA',chain), fin.atom(r_,'CB',chain)
            d=float(np.linalg.norm(ca-cb))
            diag[tag]['CA_CB'] = round(d,2)
            diag[tag]['strained'] = bool(abs(d-1.53) > 0.15)
            worst=max(worst, abs(d-1.53))
        geom_ok, geom_rows = validate_build(fin, cat_resis, chain)
        breaks = peptide_breaks(fin, chain)
        if breaks:
            geom_ok = False
        for g in geom_rows:
            tag_=f"{g.get('resn','?')}{g['resi']}"
            if tag_ in diag:
                diag[tag_]['geometry']={k:v for k,v in g.items() if k not in ('resi','resn')}
        all_r=[r for c,r in fin.protein_res if c==chain]
        _, all_rows = validate_build(fin, all_r, chain)
        n_bad=sum(1 for g in all_rows if g.get('problems'))
        bonds = backbone_bond_audit(fin, fin.xyz, chain)
        bb_strain = float(bonds.get('worst_dev', 99.0))
        cand = dict(struct=fin, sub=sub, relax=rr, rama=rama, diag=diag,
                    worst_ca_cb_dev=round(worst,2),
                    residues_moved={str(k):round(v,2) for k,v in per.items()},
                    tier=tier_of(mx),
                    protpardelle=pp_info, geometry_ok=bool(geom_ok),
                    geometry=[g for g in geom_rows],
                    peptide_breaks=breaks,
                    backbone_bonds=bonds,
                    bb_strain=round(bb_strain, 3),
                    n_residues_off_ideal=int(n_bad),
                    protpardelle_attempt=attempt+1)
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
            # keep sampling remaining protpardelle draws for a lower-strain ok build
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

def main():
    ap=argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scaffold', required=True)
    ap.add_argument('--theozyme-spec', required=True)
    ap.add_argument('--anchors', required=True,
                    help='candidate anchor positions, e.g. 83,210,180 or 50-90')
    ap.add_argument('--name', required=True)
    ap.add_argument('--deva-root', default=_DEVA_ROOT,
                    help='dEVA tree that receives inputs/ and configs/ (default: parent of this script)')
    ap.add_argument('--chain', default=None)
    ap.add_argument('--mobile', default='', help='loop ranges, e.g. 52-66,180-190')
    ap.add_argument('--chi-step', type=float, default=4.0)
    ap.add_argument('--max-cb-dev', type=float, default=2.2,
                    help='satellite CB tolerance BEFORE relaxation. Guided accommodation '
                         'moves a satellite CB ~1.2 A, so anything below ~1.5 silently '
                         'excludes placements the pipeline could actually build. '
                         'Reference: 4A29->5AN7 moved the CB of residue 51 by 1.76 A.')
    ap.add_argument('--delta-mu', default='',
                    help='Debye vector from your QM output, e.g. "-0.3111,-4.3508,5.2337"')
    ap.add_argument('--reactant-xyz', default='')
    ap.add_argument('--n-generations', type=int, default=60)
    ap.add_argument('--n-individuals', type=int, default=30)
    ap.add_argument('--n-mutations', type=int, default=3)
    ap.add_argument('--top', type=int, default=1,
                    help='how many configs to WRITE (after building and ranking)')
    ap.add_argument('--build-max', type=int, default=25,
                    help='how many surviving candidates to BUILD before ranking. Pre-build '
                         'CB deviation does not predict built quality (Spearman 0.22 on RA95), '
                         'so the only honest ranking is to build and measure. ~8 s each.')
    ap.add_argument('--no-accommodate', action='store_true',
                    help='freeze satellite backbones instead of guiding them onto the theozyme')
    ap.add_argument('--target-occ', type=float, default=None,
                    help='setpoint written into the config. Default calibrates from the built '
                         'structure, which tells dEVA to HOLD current enclosure rather than '
                         'improve it. Reference: 4A29+3NK 84.9, 5AN7 (10^9 rate) 94.7')
    ap.add_argument('--min-occ', type=float, default=85.0,
                    help='reject placements whose occlusion is below this (dEVA units). '
                         'Default 85.')
    ap.add_argument('--max-occ', type=float, default=None,
                    help='reject placements above this; guards against sealing the site')
    ap.add_argument('--satellite-positions', default='',
                    help='restrict (or, with --barrel-shell, expand) satellite hosts to '
                         'these positions, e.g. "51,83,110" or ranges "50-60,180-190". '
                         'Alone: only these hosts. With --barrel-shell: union of the '
                         'auto shell and this list (force-include mouth residues the '
                         'toward filter might still drop).')
    ap.add_argument('--barrel-shell', type=float, nargs='?', const=13.0, default=None,
                    metavar='RADIUS',
                    help='auto-restrict satellites to beta-strand positions within RADIUS '
                         'of the pocket whose CA->CB faces it. Excludes loops. Pocket '
                         'center = ligand if present (or companion PDB / --pocket-pdb), '
                         'else TIM-barrel strand C-terminal mouth. Default radius 13 A.')
    ap.add_argument('--pocket-pdb', default='',
                    help='PDB with a ligand used only to locate the pocket center for '
                         '--barrel-shell / --suggest-shell. Default: if scaffold is '
                         'NAME_scaffold.pdb, try NAME.pdb automatically.')
    ap.add_argument('--suggest-shell', type=float, default=None,
                    help='print positions whose CB lies within this distance of the pocket '
                         'and exit (same center as --barrel-shell)')
    ap.add_argument('--sat-mode', default='shift',
                    choices=['shift','cb','soft','none'],
                    help="how the satellite backbone is driven. 'shift' solves for the "
                         "smallest rigid translation that lets an ON-ROTAMER side chain "
                         "satisfy the catalytic distance and restrains N/CA/C/CB to it. "
                         "'cb'/'soft' pull only CB, which bends the N-CA-CB frame.")
    ap.add_argument('--max-shift', type=float, default=2.5,
                    help='reject a solved segment translation larger than this (A)')
    ap.add_argument('--sat-window', type=int, default=3,
                    help='residues either side of a satellite made mobile for closure')
    ap.add_argument('--allow-bad-geometry', action='store_true',
                    help='keep builds whose catalytic residues are not valid L-amino '
                         'acids. They will still be reported. Off by default because '
                         'such a structure cannot be designed against: LigandMPNN '
                         'rebuilds the side chain correctly and it moves several A.')
    ap.add_argument('--protpardelle', action='store_true',
                    help='idealise the backbone with a light protpardelle pass after '
                         'accommodation, before side chains are installed. Fixes strain '
                         'left by loop closure (N-CA bonds 1.36-1.38 vs ideal 1.458). '
                         'Does NOT fix a CB dragged off its cone -- protpardelle is '
                         'backbone-only and emits no CB.')
    ap.add_argument('--protpardelle-repo', default=None,
                    help='protpardelle checkout (default: <deva-root>/../protpardelle-1c)')
    ap.add_argument('--noise-angstrom', type=float, default=1.5,
                    help='protpardelle partial-diffusion start noise (Å on the native '
                         'schedule). Graft healing usually needs ≥1 Å; 0.4 with a 24-step '
                         'cap was silently ~0.16 Å. --protpardelle-steps floors the rewind.')
    ap.add_argument('--protpardelle-steps', type=int, default=100,
                    help='minimum PD rewind steps (1c pd.num_steps). Default 100 matches '
                         'examples/sampling/01_partial_diffusion.yaml; noise-angstrom may '
                         'raise this further. Schedule length is 500.')
    ap.add_argument('--n-protpardelle-attempts', type=int, default=5,
                    help='independent protpardelle noise-and-denoise draws per placement; '
                         'keep the geometry-ok sample with lowest backbone bond strain')
    ap.add_argument('--device', default='cuda',
                    help="torch device for protpardelle ('cuda' or 'cpu')")
    a=ap.parse_args()
    if a.protpardelle_repo is None:
        a.protpardelle_repo = os.path.join(os.path.abspath(a.deva_root), '..', 'protpardelle-1c')
    a.protpardelle_repo = os.path.abspath(a.protpardelle_repo)

    t0=time.time()
    spec=TheozymeSpec(a.theozyme_spec)
    st=Structure(a.scaffold); chain=a.chain or str(st.chain[0])
    print(spec.summary()); print()
    anchors=parse_resi_list(a.anchors)
    anchors=[r for r in anchors if (chain,r) in st.residues
             and st.resname(r,chain) not in ('GLY','PRO')]
    mobile=[]
    for seg in filter(None, a.mobile.split(',')):
        lo,hi=seg.split('-'); mobile.append((int(lo),int(hi)))
    mob_flat=sorted({r for lo,hi in mobile for r in range(lo,hi+1)})

    cen=None; cen_src=None
    if a.barrel_shell is not None or a.suggest_shell is not None:
        cen, cen_src = resolve_pocket_center(
            st, chain, pocket_pdb=(a.pocket_pdb or None), scaffold_path=a.scaffold)
        print(f'  pocket center: {cen_src}  '
              f'[{cen[0]:.2f}, {cen[1]:.2f}, {cen[2]:.2f}]')

    satpos=None
    if a.barrel_shell is not None:
        shell,segs=barrel_shell(st, cen, chain, radius=a.barrel_shell)
        satpos=[x['resi'] for x in shell]
        print(f'  barrel shell: {len(segs)} strands -> {len(satpos)} satellite positions '
              f'within {a.barrel_shell} A, loops excluded')
        for x in shell:
            print(f"     {x['wt']}{x['resi']:<5d} strand {x['strand']:>8s}  "
                  f"CB-pocket {x['cb_pocket']:5.1f}  toward {x['toward']:+.2f}")
    if a.satellite_positions:
        declared=parse_resi_list(a.satellite_positions)
        declared=[r for r in declared if (chain,r) in st.residues]
        if satpos is None:
            satpos=declared
            print(f'  satellite hosts restricted to {len(satpos)} declared positions: '
                  + ','.join(str(r) for r in satpos))
        else:
            extra=sorted(set(declared) - set(satpos))
            satpos=sorted(set(satpos) | set(declared))
            print(f'  satellite hosts: barrel shell ∪ {len(declared)} declared '
                  f'-> {len(satpos)} positions'
                  + (f' (added {",".join(str(r) for r in extra)})' if extra else ''))

    if a.suggest_shell is not None:
        rows=[]
        for c,r in st.protein_res:
            if c!=chain: continue
            ca,cb=st.atom(r,'CA',c),st.atom(r,'CB',c)
            if cb is None: continue
            d=float(np.linalg.norm(cb-cen))
            if d>a.suggest_shell: continue
            v1,v2=cb-ca,cen-cb
            pin=float(np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2))
            rows.append((d,int(r),str(st.resname(r,c)),pin))
        rows.sort()
        print(f'\n  {len(rows)} positions with CB within {a.suggest_shell} A of the pocket '
              f'({cen_src}):')
        for d,r,wt,pin in rows:
            mark='  <-' if pin>=0 else ''
            print(f'    {wt}{r:<5d} CB-pocket {d:5.1f}  CA->CB.CB->pocket {pin:+.2f}{mark}')
        inward=[r for _,r,_,pin in rows if pin>=0]
        print('\n  --satellite-positions "' + ','.join(str(r) for r in inward) + '"')
        sys.exit(0)

    relaxer=None
    if a.protpardelle:
        from theozyme.protpardelle_bridge import ProtpardelleRelaxer
        relaxer=ProtpardelleRelaxer(repo_dir=a.protpardelle_repo, task='backbone',
                                    noise_angstrom=a.noise_angstrom,
                                    n_steps=a.protpardelle_steps,
                                    schedule_steps=500,
                                    device=a.device, verbose=True)
        probs=relaxer.preflight()
        if probs:
            print('  protpardelle unavailable, continuing without it:')
            for x in probs: print(f'    {x}')
            relaxer=None
        else:
            print(f'  protpardelle partial-diffusion ON '
                  f'(noise≈{a.noise_angstrom} A, min rewind {a.protpardelle_steps}, '
                  f'schedule 500, no motif cond)')

    print(f'[1/5] exploring {len(anchors)} anchors x {int(360/a.chi_step)**2} grafts')
    ex=Explorer(spec, st, chain, mobile_resis=mob_flat, max_cb_dev=a.max_cb_dev,
                satellite_positions=satpos)
    sols=ex.run(anchors, chi_step=a.chi_step)
    if not sols:
        print('  no placement satisfies the theozyme on these anchors.'); sys.exit(2)
    prot=heavy_xyz(st, rec='ATOM')
    for s in sols:
        s["occ"]=occlusion(prot, s["sub"]) / len(s["sub"])
        s['cb_dev']=max([h['cb_dev'] for h in s['satellites']], default=0.0)
    ACCOM_BUDGET = 0.0 if a.no_accommodate else 1.2
    for s_ in sols:
        s_['cb_residual'] = max(0.0, s_['cb_dev'] - ACCOM_BUDGET)
        s_['needs_accommodation'] = bool(s_['cb_dev'] > 0.6)
    # Pre-build "CST" proxy: max satellite CB deviation (covalent tip is exact in
    # the rigid graft). Built worstCST replaces this after --build-max.
    for s_ in sols:
        s_['pre_cst']=float(s_['cb_dev'])
    sols.sort(key=lambda z: (z['pre_cst'], -z['occ']))
    print(f'  {len(sols)} complete solutions; best satellite CB deviation {sols[0]["cb_dev"]:.2f} A')

    uniq={}
    for s_ in sols:
        key=(s_['anchor'], tuple(sorted((h['resn'], h['resi']) for h in s_['satellites'])))
        if key not in uniq or s_['pre_cst'] < uniq[key]['pre_cst'] or (
                s_['pre_cst'] == uniq[key]['pre_cst'] and s_['occ'] > uniq[key]['occ']):
            uniq[key]=s_
    combos=sorted(uniq.values(), key=lambda z: (z['pre_cst'], -z['occ']))
    n_before=len(combos)
    if a.min_occ is not None:
        combos=[c for c in combos if c['occ']>=a.min_occ]
    if a.max_occ is not None:
        combos=[c for c in combos if c['occ']<=a.max_occ]
    if len(combos)!=n_before:
        lo = a.min_occ if a.min_occ is not None else float('-inf')
        hi = a.max_occ if a.max_occ is not None else float('inf')
        print(f'  occlusion filter [{lo}, {hi}] kept {len(combos)}/{n_before} assignments')
    if not combos:
        print('  nothing survives the occlusion filter -- loosen --min-occ/--max-occ'); sys.exit(2)
    print(f'  {len(combos)} distinct residue assignments')
    cand_path=os.path.join(a.deva_root,'inputs/ra95',f'{a.name}_candidates.json')
    os.makedirs(os.path.dirname(cand_path), exist_ok=True)
    json.dump([dict(rank=i,
                    anchor=int(c['anchor']), anchor_resn=spec.anchor.resn,
                    anchor_wt=c['anchor_wt'], chi1=c['chi1'], chi2=c['chi2'],
                    satellites=[dict(resn=h['resn'], resi=int(h['resi']), was=h['wt'],
                                     cb_dev=round(h['cb_dev'],2),
                                     cacbcg=round(h['cacbcg'],1)) for h in c['satellites']],
                    max_cb_dev=round(c['cb_dev'],2),
                    pre_cst=round(c['pre_cst'],2),
                    cb_residual_after_accommodation=round(c['cb_residual'],2),
                    needs_accommodation=c['needs_accommodation'],
                    occlusion=round(c['occ'],1),
                    n_graft_solutions=sum(1 for s2 in sols
                        if s2['anchor']==c['anchor'] and
                        tuple(sorted((h['resn'],h['resi']) for h in s2['satellites']))==
                        tuple(sorted((h['resn'],h['resi']) for h in c['satellites']))))
               for i,c in enumerate(combos)],
              open(cand_path,'w'), indent=2)
    print(f'  -> {cand_path}')
    print(f'\n  candidates, ordered by pre-CST (max satellite CBdev), then occlusion.')
    print(f'  This is BUILD ORDER. Final rank uses built worstCST after --build-max.')
    print(f'  Occlusion floor: {a.min_occ}')
    for i,c in enumerate(combos[:12]):
        sat=' '.join(f"{h['resn']}{h['resi']}(was {h['wt']})" for h in c['satellites'])
        flag=' [needs accommodation]' if c['needs_accommodation'] else ''
        print(f"    {i:3d}. {spec.anchor.resn}{c['anchor']}(was {c['anchor_wt']}) + {sat}"
              f"  preCST {c['pre_cst']:.2f}  CBdev {c['cb_dev']:.2f}  "
              f"occ {c['occ']:.1f}{flag}")
    sols = combos

    root=a.deva_root
    os.makedirs(os.path.join(root,'inputs/ra95'), exist_ok=True)
    os.makedirs(os.path.join(root,'configs/ra95'), exist_ok=True)
    report=[]; built=0
    for k,sol in enumerate(sols):
        if built>=a.build_max: break
        print(f'[2/5] building placement {k}: anchor {spec.anchor.resn}{sol["anchor"]} '
              f'+ ' + ' '.join(f'{h["resn"]}{h["resi"]}' for h in sol['satellites']))
        B=build_one(spec, st, sol, mobile, chain,
                    accommodate=not a.no_accommodate, window=a.sat_window,
                    sat_mode=a.sat_mode, max_shift=a.max_shift,
                    relaxer=relaxer, strict_geometry=not a.allow_bad_geometry,
                    n_protpardelle_attempts=a.n_protpardelle_attempts)
        if B is None: continue
        fin=B['struct']; sub=B['sub']
        tag=f'{a.name}_b{k}'

        lig_extra=[dict(name=n, resn=spec.lig_resn, resi=901, xyz=q, elem=(n[0] if n[0] in 'CNOSP' else 'C'),
                        chain=chain) for n,q in zip(spec.lig_atoms, sub)]
        pdb_path=os.path.join(root,'inputs/ra95',f'{tag}.pdb')
        lig_path=os.path.join(root,'inputs/ra95',f'{tag}_ligand.pdb')

        fin.write(pdb_path, extra=lig_extra)
        S=fin.serial
        con={}
        for x,y in spec.lig_bonds+spec.partial_bonds:
            ia,ib=S[(chain,901,x)],S[(chain,901,y)]
            con.setdefault(ia,[]).append(ib); con.setdefault(ib,[]).append(ia)
        links=[]
        if spec.anchor.covalent:
            c=spec.anchor.covalent
            ia=S[(chain,sol['anchor'],c['atom'])]; ib=S[(chain,901,c['ligand_atom'])]
            con.setdefault(ia,[]).append(ib); con.setdefault(ib,[]).append(ia)
            d=float(np.linalg.norm(fin.atom(sol['anchor'],c['atom'],chain)-
                                   sub[list(spec.lig_atoms).index(c['ligand_atom'])]))
            links=[dict(a=dict(name=c['atom'],resn=spec.anchor.resn,chain=chain,resi=sol['anchor']),
                        b=dict(name=c['ligand_atom'],resn=spec.lig_resn,chain=chain,resi=901),dist=d)]
        cat=[f'{chain}{sol["anchor"]}']+[f'{chain}{h["resi"]}' for h in sol['satellites']]
        fin.write(pdb_path, extra=lig_extra, conect=con, links=links, remarks=[
            f'prepared by prepare_deva.py for dEVA | theozyme {spec.name}',
            f'anchor {spec.anchor.resn}{sol["anchor"]} chi1={sol["chi1"]:.0f} chi2={sol["chi2"]:.0f}',
            'catalytic residues: '+' '.join(cat),
            'LINK declares the covalent theozyme-ligand bond; CONECT gives ligand connectivity'])
        L=Structure(); L.rec=np.array(['HETATM']*len(sub)); L.name=np.array(list(spec.lig_atoms))
        L.resn=np.array([spec.lig_resn]*len(sub)); L.chain=np.array([chain]*len(sub))
        L.resi=np.array([901]*len(sub)); L.occ=np.ones(len(sub)); L.b=np.zeros(len(sub))
        L.elem=np.array([(n[0] if n[0] in 'CNOSP' else 'C') for n in spec.lig_atoms]); L.xyz=sub
        L._index(); L.write(lig_path)

        fin2=Structure(pdb_path)
        built_occ=occlusion(heavy_xyz(fin2, rec="ATOM"), sub) / len(sub)
        if a.target_occ is not None:
            occ=a.target_occ
            gap=occ-built_occ
            print(f'[3/5] occlusion: built {built_occ:.1f}, target {occ:.1f} '
                  f'({gap:+.1f} to close by design)')
        else:
            occ=built_occ
            print(f'[3/5] occlusion: built {built_occ:.1f}; target_occ auto-set to the same '
                  f'value\n      (objective will HOLD enclosure, not improve it -- '
                  f'pass --target-occ to aim higher)')

        tbond_pair=None; tbond=None
        if spec.partial_bonds:
            xx,yy=spec.partial_bonds[0]
            tbond=[spec.lig_atoms[xx]-1, spec.lig_atoms[yy]-1]
            tbond_pair=(spec.lig_atoms[xx], spec.lig_atoms[yy])

        print('[4/5] validating theozyme_map by Kabsch, exactly as dEVA will')
        tmap=[]; P=[]; Q=[]
        names=list(spec.lig_atoms)
        for nm in names:
            tmap.append(f'{spec.lig_atoms[nm]-1}:{spec.lig_resn}:901:{nm}')
            P.append(spec.p(spec.lig_atoms[nm])); Q.append(sub[names.index(nm)])
        P=np.array(P); Q=np.array(Q)
        rmsd=kabsch_rmsd(P, Q)
        assert rmsd < 0.05, f'theozyme_map self-check failed: RMSD {rmsd:.3f} A'
        print(f'      theozyme_map RMSD = {rmsd:.4f} A  ({len(tmap)} ligand atoms, no protein)')

        Pc,Qc=P.mean(0),Q.mean(0); Hm=(P-Pc).T@(Q-Qc)
        U_,_,Vt_=np.linalg.svd(Hm)
        Rm=Vt_.T@np.diag([1,1,np.sign(np.linalg.det(Vt_.T@U_.T))])@U_.T
        tvec=Qc-Rm@Pc
        axis_pdb=probe_pdb=dmu_pdb=None
        if tbond_pair is not None:
            i0,j0=tbond_pair
            v=spec.p(i0)-spec.p(j0)
            axis_pdb=Rm@(-v/np.linalg.norm(v))
            probe_pdb=Rm@(0.5*(spec.p(i0)+spec.p(j0)))+tvec
            print(f'      breaking bond {spec.partial_bonds[0][0]}->{spec.partial_bonds[0][1]}'
                  f'  |b|={np.linalg.norm(v):.3f} A at the TS')
            print(f'      axis  (pdb frame) [{axis_pdb[0]:+.4f}, {axis_pdb[1]:+.4f}, {axis_pdb[2]:+.4f}]')
            print(f'      probe (pdb frame) [{probe_pdb[0]:.3f}, {probe_pdb[1]:.3f}, {probe_pdb[2]:.3f}]')
        if a.delta_mu.strip():
            dmu_tz=np.array([float(x) for x in a.delta_mu.split(',')])
            dmu_pdb=Rm@dmu_tz
            print(f'      delta_mu (pdb frame) [{dmu_pdb[0]:+.4f}, {dmu_pdb[1]:+.4f}, '
                  f'{dmu_pdb[2]:+.4f}]  |dmu|={np.linalg.norm(dmu_pdb):.2f} D')

        cst_built={}
        if spec.anchor.covalent:
            cc=spec.anchor.covalent
            tgt=float(np.linalg.norm(spec.p(spec.anchor.atoms[cc['atom']])
                                     -spec.p(spec.lig_atoms[cc['ligand_atom']])))
            got=float(np.linalg.norm(fin2.atom(sol['anchor'],cc['atom'],chain)
                                     -sub[list(spec.lig_atoms).index(cc['ligand_atom'])]))
            cst_built['CST1']=dict(atoms=f"{cc['atom']}...{cc['ligand_atom']}",
                                   target=round(tgt,2), built=round(got,2),
                                   deviation=round(abs(got-tgt),2))
        for si,h in enumerate(sol['satellites'], start=2):
            sat=[x for x in spec.satellites if x.resn==h['resn']][0]
            tip=TIP.get(sat.resn)
            if tip is None or tip not in sat.atoms: continue
            near=min(spec.lig_atoms,
                     key=lambda nm: np.linalg.norm(spec.p(sat.atoms[tip])-spec.p(spec.lig_atoms[nm])))
            tgt=float(np.linalg.norm(spec.p(sat.atoms[tip])-spec.p(spec.lig_atoms[near])))
            got=float(np.linalg.norm(fin2.atom(h['resi'],tip,chain)
                                     -sub[list(spec.lig_atoms).index(near)]))
            cst_built[f'CST{si}']=dict(atoms=f"{sat.resn}{h['resi']}.{tip}...{near}",
                                       target=round(tgt,2), built=round(got,2),
                                       deviation=round(abs(got-tgt),2))
        worst_cst=max((v['deviation'] for v in cst_built.values()), default=99.0)

        print('[5/5] writing config')
        cfg=render_config(a, spec, tag, cat, tmap, tbond, occ, root,
                          built=built_occ, explicit=a.target_occ is not None,
                          probe=probe_pdb, axis=axis_pdb, dmu_vec=dmu_pdb)
        cfg_path=os.path.join(root,'configs/ra95',f'{tag}.yml')
        open(cfg_path,'w').write(cfg)
        report.append(dict(tag=tag, pre_rank=k, anchor=sol['anchor'], chi1=sol['chi1'], chi2=sol['chi2'],
            satellites=[{k:v for k,v in h.items()} for h in sol['satellites']],
            catalytic=cat, theozyme_map=tmap, theozyme_bond=tbond,
            cst_built=cst_built, worst_cst_deviation=round(worst_cst,2),
            target_occ=round(occ,1), built_occ=round(built_occ,1),
            map_rmsd=round(rmsd,4), tier=B['tier'],
            residues_moved=B['residues_moved'], sidechains=B['diag'],
            worst_CA_CB_deviation=B['worst_ca_cb_dev'],
            geometry_ok=B.get('geometry_ok'), geometry=B.get('geometry'),
            peptide_breaks=B.get('peptide_breaks'),
            bb_strain=B.get('bb_strain'),
            backbone_bonds=B.get('backbone_bonds'),
            n_residues_off_ideal=B.get('n_residues_off_ideal'),
            protpardelle=B.get('protpardelle'),
            protpardelle_attempt=B.get('protpardelle_attempt'),
            relax={k:round(v,3) for k,v in B['relax'].items() if k!='xyz'},
            pdb=os.path.relpath(pdb_path,root), ligand=os.path.relpath(lig_path,root),
            config=os.path.relpath(cfg_path,root)))
        built+=1
        print(f'      -> {cfg_path}')
    # Prefer catalytic tip fidelity (worstCST), then backbone strain, then CA-CB.
    report.sort(key=lambda r: (r['worst_cst_deviation'],
                               r.get('bb_strain', 99.0),
                               r['worst_CA_CB_deviation']))
    for i,r in enumerate(report):
        r['built_rank']=i
    def _mv(src, dst):
        ps, pd = os.path.join(root,src), os.path.join(root,dst)
        if os.path.exists(ps) and ps != pd: os.replace(ps, pd)
    staged=[]
    for i,r in enumerate(report):
        newtag=f'{a.name}_rank{i}'
        staged.append((r, newtag, f'inputs/ra95/{newtag}.pdb', f'inputs/ra95/{newtag}_ligand.pdb',
                       f'configs/ra95/{newtag}.yml'))
    for r,newtag,npdb,nlig,ncfg in staged:
        _mv(r['pdb'], npdb+'.tmp'); _mv(r['ligand'], nlig+'.tmp')
        p=os.path.join(root,r['config'])
        if os.path.exists(p): os.remove(p)
    for i,(r,newtag,npdb,nlig,ncfg) in enumerate(staged):
        _mv(npdb+'.tmp', npdb); _mv(nlig+'.tmp', nlig)
        r.update(tag=newtag, pdb=npdb, ligand=nlig)
        if i < a.top:
            open(os.path.join(root,ncfg),'w').write(
                render_config(a, spec, newtag, r['catalytic'], r['theozyme_map'],
                              r['theozyme_bond'], r['target_occ'], root,
                              built=r['built_occ'], explicit=a.target_occ is not None))
            r['config']=ncfg
        else:
            r['config']=None
    json.dump(report, open(os.path.join(root,'inputs/ra95',f'{a.name}_report.json'),'w'), indent=2)
    print(f'\ndone in {time.time()-t0:.0f}s\n')
    print(f'  final rank: worstCST, then strain, then CA-CB  |  occ floor was {a.min_occ}')
    print(f"{'rank':>4s} {'pre':>4s} {'assignment':>22s} {'strain':>7s} {'worstCST':>9s} {'CA-CB':>6s} {'occ':>6s} {'geom':>5s} {'pp':>3s}")
    for r in report:
        sat=' '.join(f"{h['resn']}{h['resi']}" for h in r['satellites'])
        print(f"{r['built_rank']:>4d} {r.get('pre_rank','-'):>4} "
              f"{spec.anchor.resn+str(r['anchor'])+'+'+sat:>22s} "
              f"{r.get('bb_strain', float('nan')):>7.3f} "
              f"{r['worst_cst_deviation']:>9.2f} {r['worst_CA_CB_deviation']:>6.2f} {r['built_occ']:>6.1f} "
              f"{('ok' if r.get('geometry_ok') else 'BAD'):>5s} "
              f"{r.get('protpardelle_attempt','-'):>3}")
    print(f'\n{len(report)} structures kept, one file each: inputs/ra95/{a.name}_rank<i>.pdb')
    print(f'configs written for the top {a.top}: ' +
          ', '.join(r['config'] for r in report[:a.top] if r['config']))
    for r in report:
        print(f"  {r['tag']}: {spec.anchor.resn}{r['anchor']} + " +
              ' '.join(f"{h['resn']}{h['resi']}" for h in r['satellites']) +
              f" | target_occ {r['target_occ']} | map RMSD {r['map_rmsd']}")
        for k,v in r['sidechains'].items(): print(f'      {k}: {v}')
    print('\nRUN IT:')
    print(f'  cd {root}')
    print(f'  python run.py -c {report[0]["config"]} '
          f'--models seq_model protpardelle_relax electric_field pocket_shape desolvation')

def render_config(a, spec, tag, cat, tmap, tbond, occ, root, built=None, explicit=False,
                  probe=None, axis=None, dmu_vec=None):
    dmu = a.delta_mu.strip()
    dmu_line = (f'    delta_mu_theozyme: [{dmu}]' if dmu else
                '    # delta_mu_theozyme: [x, y, z]  # Debye, from QM')
    react = a.reactant_xyz or '# set --reactant-xyz'
    if probe is not None and axis is not None:
        resolved = (
f"""    probe_xyz: [{probe[0]:.3f}, {probe[1]:.3f}, {probe[2]:.3f}]
    axis_xyz:  [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]""")
        if dmu_vec is not None:
            resolved += (f"\n    delta_mu:  [{dmu_vec[0]:.4f}, {dmu_vec[1]:.4f}, "
                         f"{dmu_vec[2]:.4f}]")
    else:
        resolved = None
    fixed=' '.join(cat)
    bond=f'    theozyme_bond: {tbond}' if tbond else '    # theozyme_bond: [i, j]'
    mapping='\n'.join(f'      - "{m}"' for m in tmap)
    ef_block = resolved if resolved is not None else (
        f'    theozyme_reactant: {react}\n'
        f'    theozyme_ts:       {spec.xyz_path}\n'
        f'    theozyme_map:\n{mapping}\n{bond}\n'
        f'    theozyme_max_rmsd: 0.75\n{dmu_line}')
    occ_note = (f'set via --target-occ (built: {built:.1f})'
                if explicit else f'calibrated from inputs/ra95/{tag}.pdb')
    theo_resis = [int(''.join(ch for ch in x if ch.isdigit())) for x in cat]
    pp_repo = getattr(a, 'protpardelle_repo', None) or '../protpardelle-1c'
    try:
        pp_repo = os.path.relpath(os.path.abspath(pp_repo), root)
    except ValueError:
        pp_repo = os.path.abspath(pp_repo)
    noise = float(getattr(a, 'noise_angstrom', 1.5) or 1.5)
    n_steps = int(getattr(a, 'protpardelle_steps', 100) or 100)
    n_pp = int(getattr(a, 'n_protpardelle_attempts', 3) or 3)
    device = str(getattr(a, 'device', 'cuda') or 'cuda')
    return f"""# {tag}.yml -- {spec.name}; catalytic: {' '.join(cat)}
# Run with:
#   python run.py -c configs/ra95/{tag}.yml \\
#     --models seq_model protpardelle_relax electric_field pocket_shape desolvation
# MODEL ORDER IS LOAD-BEARING: protpardelle_relax must be second so objectives
# score the relaxed structure.

general:
  cuda: true
  seed: 2
  outputs: /scratch/users/gelnesr/dEVA/ra95/{tag}
  save_outputs: true

input:
  pdb: inputs/ra95/{tag}.pdb

evolution:
  n_generations: {a.n_generations}
  n_individuals: {a.n_individuals}
  n_mutations: {a.n_mutations}

seq_model: ligandmpnn

models:

  protpardelle_relax:
    theozyme_residues: {theo_resis}
    ligand_resi: 901
    chain: A
    reference_pdb: inputs/ra95/{tag}.pdb
    mode: fixed
    backend: protpardelle
    repo_dir: {pp_repo}
    task: backbone
    # model_name: cc58
    # model_epoch: 416
    device: {device}
    noise_angstrom: {noise}
    n_steps: {n_steps}
    schedule_steps: 500
    step_scale: 1.0
    s_churn: 0.0
    align_window: 8
    n_protpardelle_attempts: {n_pp}
    require_protpardelle: false
    rescore_pmpnn_after: true
    clearance: 3.40
    shell: 15.0
    max_backbone_disp: 2.0
    relieve_sidechains: true
    closure_tol: 0.08
    k_closure: 60.0
    k_rep: 25.0
    maxiter: 300
    emit_objective: false
    w_strain: 0.5
    free_rmsd: 1.0
    fail_value: -10.0
    accept_tol: 0.05
    out_subdir: relaxed
    verbose: false

  desolvation:
    verbose: true

  electric_field:
{ef_block}
    exclude_residues: {json.dumps(cat)}
    include_hetatm: false
    fix_incomplete: true
    report_au: false
    verbose: false

  pocket_shape:
    ligand_pdb: inputs/ra95/{tag}_ligand.pdb
    target_occ: {occ:.1f}       # {occ_note}
    r_clash: 1.2
    w_clash: 1.0
    w_seal: 0.0
    verbose: false

  ligandmpnn:
    model_path: ./models/ligandmpnn/model_params/ligandmpnn_v_32_020_25.pt
    packer_path: ./models/ligandmpnn/model_params/ligandmpnn_sc_v_32_002_16.pt
    var_residues: ""
    fixed_residues: "{fixed}"
    bias_AA: ""
    bias_AA_per_residue: ""
    omit_AA: "C"
    omit_AA_per_residue: ""
    symmetry_residues: ""
    symmetry_weights: ""
    homo_oligomer: 0
    repack_everything: 0
    sc_num_denoising_steps: 8
    sc_num_samples: 5
    ligand_mpnn_cutoff_for_score: 20.0
    use_atom_context: 1
    temperature: 0.5
    zero_indexed: 0
    autoregressive_score: 0
    single_aa_score: 1
    chains_to_design: ""
    parse_these_chains_only: ""
    parse_atoms_with_zero_occupancy: 0
    verbose: false
    keep_ligand_in_packed: 1

  proteinmpnn:
    model_path: ./models/ligandmpnn/model_params/proteinmpnn_v_48_010.pt
    ligand_mpnn_cutoff_for_score: 20.0
    temperature: 0.5
    zero_indexed: 0
    autoregressive_score: 0
    single_aa_score: 1
    fixed_residues: "{fixed}"
    bias_AA: ""
    omit_AA: "C"
    symmetry_residues: ""
    symmetry_weights: ""
    homo_oligomer: 0
    chains_to_design: ""
    parse_these_chains_only: ""
    parse_atoms_with_zero_occupancy: 0
    verbose: false
"""

if __name__ == '__main__':
    main()
