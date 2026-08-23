#!/usr/bin/env python3
"""Prepare scaffold + theozyme placements as PDBs (any theozyme, not RA95-specific).

Works for covalent *and* non-covalent theozymes:

- Always pick one frame residue with ``role: "anchor"`` (χ1/χ2 search places the
  rigid theozyme XYZ onto the scaffold).
- **Non-covalent**: omit ``covalent_to_ligand`` in the JSON. No protein–ligand LINK.
  Anchor/satellites are CB + tip geometry constraints, not a covalent adduct.
- **Covalent**: keep ``covalent_to_ligand``; LINK is written into the complex PDB.

Theozyme input may be a JSON spec (+ XYZ) **or** a PDB containing catalytic
residues + ligand HETATM (``--theozyme-pdb`` / ``--theozyme-spec *.pdb`` with
``--catalytic``).

Writes (under ``--out-dir``):
  ``{name}_candidates.json``, ``{name}_rank{i}.pdb``, ``{name}_rank{i}_ligand.pdb``,
  ``{name}_report.json``.

Does *not* write dEVA campaign YMLs, ``delta_mu``, reactant maps, or electric-field
wiring — use ``project_retroaldolase/prepare_deva.py`` for the RA95 campaign path.

Example (non-covalent JSON)::

  python theozyme/prepare_placements.py \\
    --scaffold inputs/myscaffold.pdb \\
    --theozyme-spec path/to/theozyme_spec.json \\
    --name my_theozyme \\
    --anchors 40-60,80-100 \\
    --satellite-positions 45,90,120 \\
    --mobile 50-65 \\
    --out-dir inputs/my_theozyme \\
    --build-max 10 --top 5

Example (theozyme PDB — first catalytic residue is the frame/anchor)::

  python theozyme/prepare_placements.py \\
    --scaffold inputs/myscaffold.pdb \\
    --theozyme-pdb path/to/theozyme.pdb \\
    --catalytic 83,51 \\
    --name my_theozyme \\
    --anchors 40-60,80-100 \\
    --satellite-positions 45,90,120 \\
    --out-dir inputs/my_theozyme
"""
import argparse
import os
import sys
import time

_DEVA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DEVA_ROOT not in sys.path:
    sys.path.insert(0, _DEVA_ROOT)

from theozyme.spec import load_theozyme
from theozyme.structure import Structure
from theozyme.placements import (
    build_ranked_pdbs, explore_and_filter, parse_resi_list,
    resolve_satellite_hosts, is_covalent,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scaffold', required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--theozyme-spec', default=None,
                     help='theozyme JSON (+ xyz field), or a .pdb (needs --catalytic)')
    src.add_argument('--theozyme-pdb', default=None,
                     help='theozyme PDB with catalytic residues + ligand HETATM')
    ap.add_argument('--catalytic', default='',
                    help='for PDB theozymes: anchor[,satellite...] residue numbers '
                         '(e.g. 83,51 or A83,A51). First entry is the frame/anchor.')
    ap.add_argument('--theozyme-chain', default=None,
                    help='default chain ID inside the theozyme PDB (default: first chain)')
    ap.add_argument('--ligand-resi', type=int, default=None,
                    help='ligand residue number in the theozyme PDB '
                         '(default: largest non-water HETATM)')
    ap.add_argument('--covalent', default=None, metavar='PROT:LIG',
                    help='force covalent adduct, e.g. NZ:C13 (overrides LINK)')
    ap.add_argument('--no-covalent', action='store_true',
                    help='ignore PDB LINK records; treat as non-covalent')
    ap.add_argument('--partial-bonds', default='',
                    help='TS / partial ligand bonds as A:B,C:D (optional)')
    ap.add_argument('--write-spec', default=None,
                    help='when loading a PDB, also dump reusable JSON+XYZ here')
    ap.add_argument('--no-waters', action='store_true',
                    help='omit nearby HOH from a PDB theozyme')
    ap.add_argument('--anchors', required=True,
                    help='candidate frame/anchor positions, e.g. 83,210 or 50-90')
    ap.add_argument('--name', required=True)
    ap.add_argument('--out-dir', default=None,
                    help='directory for PDBs + JSON (default: inputs/<name>)')
    ap.add_argument('--deva-root', default=_DEVA_ROOT,
                    help='repo root used to resolve relative --out-dir defaults')
    ap.add_argument('--chain', default=None)
    ap.add_argument('--mobile', default='', help='loop ranges, e.g. 52-66,180-190')
    ap.add_argument('--chi-step', type=float, default=4.0,
                    help='χ1/χ2 grid step for frame-residue search (always used; '
                         'not covalent-specific)')
    ap.add_argument('--max-cb-dev', type=float, default=2.2,
                    help='satellite CB tolerance before accommodation')
    ap.add_argument('--top', type=int, default=None,
                    help='how many ranked PDBs to keep (default: all built)')
    ap.add_argument('--build-max', type=int, default=25,
                    help='how many distinct assignments to build before ranking')
    ap.add_argument('--no-accommodate', action='store_true')
    ap.add_argument('--min-occ', type=float, default=None,
                    help='reject placements below this occlusion (default: no floor)')
    ap.add_argument('--max-occ', type=float, default=None)
    ap.add_argument('--target-occ', type=float, default=None,
                    help='recorded in the report only (no YML written here)')
    ap.add_argument('--satellite-positions', default='',
                    help='restrict / expand satellite hosts (see prepare_deva)')
    ap.add_argument('--barrel-shell', type=float, nargs='?', const=13.0, default=None,
                    metavar='RADIUS',
                    help='auto β-strand shell around pocket (TIM-barrel helper; optional)')
    ap.add_argument('--pocket-pdb', default='',
                    help='companion PDB with ligand for pocket center')
    ap.add_argument('--suggest-shell', type=float, default=None,
                    help='print CB-near-pocket positions and exit')
    ap.add_argument('--sat-mode', default='shift',
                    choices=['shift', 'cb', 'soft', 'none'])
    ap.add_argument('--max-shift', type=float, default=2.5)
    ap.add_argument('--sat-window', type=int, default=3)
    ap.add_argument('--allow-bad-geometry', action='store_true')
    ap.add_argument('--protpardelle', action='store_true')
    ap.add_argument('--protpardelle-repo', default=None)
    ap.add_argument('--noise-angstrom', type=float, default=1.5)
    ap.add_argument('--protpardelle-steps', type=int, default=100)
    ap.add_argument('--n-protpardelle-attempts', type=int, default=5)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--disable-tqdm', action='store_true', default=True,
                    help='silence protpardelle "Sampling backbones" bar (default)')
    ap.add_argument('--tqdm', dest='disable_tqdm', action='store_false',
                    help='show protpardelle sampling progress bars')
    a = ap.parse_args()

    if a.out_dir is None:
        a.out_dir = os.path.join(a.deva_root, 'inputs', a.name)
    a.out_dir = os.path.abspath(a.out_dir)
    if a.protpardelle_repo is None:
        a.protpardelle_repo = os.path.join(os.path.abspath(a.deva_root),
                                          '..', 'protpardelle-1c')
    a.protpardelle_repo = os.path.abspath(a.protpardelle_repo)

    t0 = time.time()
    theo_path = a.theozyme_pdb or a.theozyme_spec
    is_pdb = (a.theozyme_pdb is not None
              or os.path.splitext(theo_path)[1].lower() == '.pdb')
    if is_pdb and not a.catalytic:
        ap.error('PDB theozyme requires --catalytic ANCHOR[,SAT...] '
                 '(first residue is the frame/anchor)')
    cov_pair = None
    if a.covalent:
        if ':' not in a.covalent:
            ap.error('--covalent must look like PROT:LIG (e.g. NZ:C13)')
        p, q = a.covalent.split(':', 1)
        cov_pair = (p.strip(), q.strip())
    partial = [x.strip() for x in a.partial_bonds.split(',') if x.strip()]
    pdb_kwargs = dict(
        name=a.name,
        chain=a.theozyme_chain,
        ligand_resi=a.ligand_resi,
        covalent=cov_pair,
        no_covalent=a.no_covalent,
        partial_bonds=partial or None,
        include_waters=not a.no_waters,
    )
    if is_pdb:
        spec = load_theozyme(theo_path, catalytic=a.catalytic, **pdb_kwargs)
        if a.write_spec:
            from theozyme.pdb_import import write_spec_bundle
            j, x = write_spec_bundle(spec.raw, a.write_spec)
            print(f'  wrote reusable spec: {j}\n  xyz: {x}')
    else:
        if a.catalytic or a.covalent or a.no_covalent or a.ligand_resi is not None:
            print('  note: --catalytic/--covalent/--ligand-resi apply only to PDB '
                  'theozymes; JSON path ignored those flags')
        spec = load_theozyme(theo_path)
    st = Structure(a.scaffold)
    chain = a.chain or str(st.chain[0])
    print(spec.summary())
    print()
    if is_covalent(spec):
        print('  placement mode: COVALENT (LINK will be written)')
    else:
        print('  placement mode: NON-COVALENT (no protein–ligand LINK; '
              'anchor is frame residue only)')
    print()

    anchors = parse_resi_list(a.anchors)
    anchors = [r for r in anchors if (chain, r) in st.residues
               and st.resname(r, chain) not in ('GLY', 'PRO')]
    mobile = []
    for seg in filter(None, a.mobile.split(',')):
        lo, hi = seg.split('-')
        mobile.append((int(lo), int(hi)))
    mob_flat = sorted({r for lo, hi in mobile for r in range(lo, hi + 1)})

    satpos, _ = resolve_satellite_hosts(
        st, chain, a.scaffold,
        barrel_shell_r=a.barrel_shell,
        satellite_positions=a.satellite_positions,
        pocket_pdb=a.pocket_pdb,
        suggest_shell=a.suggest_shell)

    relaxer = None
    if a.protpardelle:
        from theozyme.protpardelle_bridge import ProtpardelleRelaxer
        relaxer = ProtpardelleRelaxer(
            repo_dir=a.protpardelle_repo, task='backbone',
            noise_angstrom=a.noise_angstrom, n_steps=a.protpardelle_steps,
            schedule_steps=500, device=a.device, verbose=True,
            disable_tqdm=a.disable_tqdm)
        probs = relaxer.preflight()
        if probs:
            print('  protpardelle unavailable, continuing without it:')
            for x in probs:
                print(f'    {x}')
            relaxer = None
        else:
            print(f'  protpardelle partial-diffusion ON '
                  f'(noise≈{a.noise_angstrom} A, min rewind {a.protpardelle_steps})')

    os.makedirs(a.out_dir, exist_ok=True)
    cand_path = os.path.join(a.out_dir, f'{a.name}_candidates.json')
    combos = explore_and_filter(
        spec, st, chain, anchors, mob_flat, satpos,
        chi_step=a.chi_step, max_cb_dev=a.max_cb_dev,
        min_occ=a.min_occ, max_occ=a.max_occ,
        no_accommodate=a.no_accommodate,
        candidates_path=cand_path)
    if not combos:
        sys.exit(2)

    report = build_ranked_pdbs(
        spec, st, chain, combos, mobile, a.out_dir, a.name,
        build_max=a.build_max, top=a.top,
        accommodate=not a.no_accommodate, sat_window=a.sat_window,
        sat_mode=a.sat_mode, max_shift=a.max_shift, relaxer=relaxer,
        allow_bad_geometry=a.allow_bad_geometry,
        n_protpardelle_attempts=a.n_protpardelle_attempts,
        target_occ=a.target_occ, min_occ=a.min_occ)

    print(f'\ndone in {time.time() - t0:.0f}s')
    if not report:
        print('  no builds survived; try --allow-bad-geometry or loosen filters')
        sys.exit(2)
    for r in report:
        if not r.get('pdb'):
            continue
        print(f"  {r['tag']}: {spec.anchor.resn}{r['anchor']} + "
              + ' '.join(f"{h['resn']}{h['resi']}" for h in r['satellites'])
              + f" | occ {r['built_occ']} | map RMSD {r['map_rmsd']}")
        for k, v in r['sidechains'].items():
            print(f'      {k}: {v}')


if __name__ == '__main__':
    main()
