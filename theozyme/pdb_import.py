"""Build a TheozymeSpec-compatible description from a theozyme PDB.

A theozyme PDB is any structure that contains:

- one or more catalytic protein residues (``ATOM``), identified by residue number
- one ligand residue (``HETATM``, non-water)
- optional ``LINK`` (covalent protein–ligand bond)
- optional ``CONECT`` (ligand connectivity; otherwise bonds are distance-inferred)
- optional waters (``HOH`` / ``WAT``) near the ligand

Roles cannot be inferred from a PDB alone: the caller must name the **anchor**
(frame residue) and any **satellites**. Covalent vs non-covalent is taken from
``LINK`` (or an explicit override), never guessed by distance alone.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .sidechains import CHI_DEF
from .structure import _WATER
from .theozyme import COV

_BACKBONE = frozenset(('N', 'CA', 'C', 'O', 'OXT', 'H', 'HA', 'H1', 'H2', 'H3'))


def _is_heavy(elem, name):
    e = (elem or '').strip().upper() or (name[:1].upper() if name else 'X')
    return e != 'H'


def _sidechain_heavy_names(resn):
    """CB + chi-movable heavy atoms for a standard residue."""
    names = {'CB'}
    for _, mv in CHI_DEF.get(resn, []):
        names.update(mv)
    return names


def parse_pdb_extras(path):
    """Parse ATOM/HETATM plus LINK and CONECT (Structure ignores the latter)."""
    atoms = []  # dicts with serial, rec, name, resn, chain, resi, xyz, elem
    links = []
    conect = {}  # serial -> set of serials
    for line in open(path):
        if line.startswith(('ATOM', 'HETATM')):
            if len(line) > 16 and line[16] not in (' ', 'A'):
                continue
            serial = int(line[6:11])
            name = line[12:16].strip()
            resn = line[17:20].strip()
            chain = line[21]
            resi = int(line[22:26])
            xyz = np.array([float(line[30:38]), float(line[38:46]),
                            float(line[46:54])], float)
            elem = line[76:78].strip() if len(line) >= 78 else ''
            if not elem:
                elem = name[0] if name else 'X'
            atoms.append(dict(serial=serial, rec=line[:6].strip(), name=name,
                              resn=resn, chain=chain, resi=resi, xyz=xyz,
                              elem=elem))
        elif line.startswith('LINK'):
            # PDB v3 LINK columns (also matches Structure.write format).
            a = dict(name=line[12:16].strip(), resn=line[17:20].strip(),
                     chain=line[21], resi=int(line[22:26]))
            b = dict(name=line[42:46].strip(), resn=line[47:50].strip(),
                     chain=line[51], resi=int(line[52:56]))
            links.append((a, b))
        elif line.startswith('CONECT'):
            toks = [line[i:i + 5] for i in range(6, min(len(line.rstrip()), 31), 5)]
            nums = []
            for t in toks:
                t = t.strip()
                if t:
                    nums.append(int(t))
            if not nums:
                continue
            a = nums[0]
            conect.setdefault(a, set())
            for b in nums[1:]:
                conect[a].add(b)
                conect.setdefault(b, set()).add(a)
    return atoms, links, conect


def _parse_resi_token(tok, default_chain):
    """'83', 'A83', or 'A:83' -> (chain, resi)."""
    tok = tok.strip()
    if not tok:
        raise ValueError('empty residue token')
    if ':' in tok:
        ch, r = tok.split(':', 1)
        return ch.strip() or default_chain, int(r)
    if tok[0].isalpha() and len(tok) > 1 and tok[1:].lstrip('-').isdigit():
        return tok[0], int(tok[1:])
    return default_chain, int(tok)


def parse_catalytic_list(s, default_chain='A'):
    """'83,51,56' or 'A83,A51' -> [(chain, resi), ...]. First is the anchor."""
    out = []
    for tok in filter(None, (t.strip() for t in s.split(','))):
        out.append(_parse_resi_token(tok, default_chain))
    if not out:
        raise ValueError('catalytic list is empty')
    return out


def _residue_atoms(atoms, chain, resi):
    return [a for a in atoms if a['chain'] == chain and a['resi'] == resi
            and a['rec'] == 'ATOM']


def _het_atoms(atoms, chain, resi):
    return [a for a in atoms if a['chain'] == chain and a['resi'] == resi
            and a['rec'] == 'HETATM']


def _pick_ligand(atoms, ligand_resi=None, ligand_chain=None, default_chain='A'):
    """Return (chain, resi, resn, heavy_atom_list)."""
    # Group HETATM residues excluding water.
    groups = {}
    for a in atoms:
        if a['rec'] != 'HETATM' or a['resn'] in _WATER:
            continue
        key = (a['chain'], a['resi'])
        groups.setdefault(key, []).append(a)
    if not groups:
        raise ValueError('theozyme PDB has no non-water HETATM ligand residue')

    if ligand_resi is not None:
        ch = ligand_chain or default_chain
        key = (ch, int(ligand_resi))
        if key not in groups:
            # Allow matching by resi alone if unique.
            hits = [k for k in groups if k[1] == int(ligand_resi)]
            if len(hits) == 1:
                key = hits[0]
            else:
                raise ValueError(
                    f'ligand residue {ch}{ligand_resi} not found among '
                    f'{sorted(groups)}')
    else:
        # Largest heavy-atom count; prefer resi 901 if tied (placement convention).
        def score(k):
            heavies = sum(1 for a in groups[k] if _is_heavy(a['elem'], a['name']))
            prefer = 1 if k[1] == 901 else 0
            return (heavies, prefer)
        key = max(groups, key=score)

    heavies = [a for a in groups[key] if _is_heavy(a['elem'], a['name'])]
    if not heavies:
        raise ValueError(f'ligand {key[0]}{key[1]} has no heavy atoms')
    resn = heavies[0]['resn']
    return key[0], key[1], resn, heavies


def _infer_bonds(atom_list, conect, bond_factor=1.20):
    """Ligand bonds from CONECT if present, else covalent-radii distance graph."""
    by_serial = {a['serial']: a for a in atom_list}
    names = {a['serial']: a['name'] for a in atom_list}
    bonds = set()

    # Prefer CONECT among ligand serials.
    lig_serials = set(by_serial)
    for a, bs in conect.items():
        if a not in lig_serials:
            continue
        for b in bs:
            if b in lig_serials and a < b:
                bonds.add((names[a], names[b]))

    if bonds:
        return sorted(bonds)

    # Distance heuristic.
    for i, a in enumerate(atom_list):
        ra = COV.get(a['elem'], COV.get(a['elem'].title(), 0.77))
        for b in atom_list[i + 1:]:
            rb = COV.get(b['elem'], COV.get(b['elem'].title(), 0.77))
            d = float(np.linalg.norm(a['xyz'] - b['xyz']))
            if d <= bond_factor * (ra + rb):
                bonds.add(tuple(sorted((a['name'], b['name']))))
    return sorted(bonds)


def _find_covalent(links, anchor_chain, anchor_resi, lig_chain, lig_resi,
                   lig_atom_names, explicit=None, order='single'):
    """Return covalent_to_ligand dict or None."""
    if explicit:
        prot, lig = explicit
        return dict(atom=prot, ligand_atom=lig, order=order)

    for a, b in links:
        pairs = ((a, b), (b, a))
        for prot, lig in pairs:
            if (prot['chain'] == anchor_chain and prot['resi'] == anchor_resi
                    and lig['chain'] == lig_chain and lig['resi'] == lig_resi
                    and lig['name'] in lig_atom_names):
                return dict(atom=prot['name'], ligand_atom=lig['name'],
                            order=order)
    return None


def _nearby_waters(atoms, ref_xyz, radius=4.0):
    """HOH residues with O within radius of any reference point."""
    water_groups = {}
    for a in atoms:
        if a['rec'] != 'HETATM' or a['resn'] not in _WATER:
            continue
        if not _is_heavy(a['elem'], a['name']):
            continue
        key = (a['chain'], a['resi'])
        water_groups.setdefault(key, []).append(a)
    out = []
    for key, wa in sorted(water_groups):
        o = next((x for x in wa if x['name'] in ('O', 'OH2')), wa[0])
        dmin = min(float(np.linalg.norm(o['xyz'] - r)) for r in ref_xyz)
        if dmin <= radius:
            out.append(wa)
    return out


def pdb_to_theozyme_dict(path, catalytic, name=None, chain=None,
                         ligand_resi=None, ligand_chain=None,
                         covalent=None, no_covalent=False,
                         covalent_order='single',
                         partial_bonds=None, include_waters=True,
                         water_radius=4.0, bond_factor=1.20,
                         lig_resn=None):
    """Convert a theozyme PDB into a dict consumed by ``TheozymeSpec.from_dict``.

    Parameters
    ----------
    path : str
        PDB path.
    catalytic : sequence of (chain, resi) or parseable string
        First entry is the **anchor**; the rest are **satellites**.
    covalent : None or (prot_atom, lig_atom)
        Force a covalent adduct. If None and ``no_covalent`` is False, use a
        PDB ``LINK`` between the anchor and the ligand when present.
    """
    atoms, links, conect = parse_pdb_extras(path)
    if not atoms:
        raise ValueError(f'no ATOM/HETATM records in {path}')

    default_chain = chain or (atoms[0]['chain'] if atoms else 'A')
    if isinstance(catalytic, str):
        cat = parse_catalytic_list(catalytic, default_chain)
    else:
        cat = [(c or default_chain, int(r)) for c, r in catalytic]
    if not cat:
        raise ValueError('need at least one catalytic residue (the anchor)')

    lig_ch, lig_resi, lig_resn_found, lig_atoms = _pick_ligand(
        atoms, ligand_resi=ligand_resi, ligand_chain=ligand_chain,
        default_chain=default_chain)
    lig_resn = lig_resn or lig_resn_found

    # Build ordered atom list → 1-based XYZ indices.
    el, coords, claim = [], [], []  # claim entries describe JSON maps
    residues_out = []

    def add_atom(a, owner):
        el.append(a['elem'] if a['elem'] else a['name'][0])
        coords.append(a['xyz'])
        idx = len(el)  # 1-based
        claim.append((owner, a['name'], idx, a['serial']))
        return idx

    for i, (ch, resi) in enumerate(cat):
        rats = _residue_atoms(atoms, ch, resi)
        if not rats:
            raise ValueError(f'catalytic residue {ch}{resi} not found as ATOM')
        resn = rats[0]['resn']
        want = _sidechain_heavy_names(resn)
        by_name = {a['name']: a for a in rats
                   if _is_heavy(a['elem'], a['name']) and a['name'] not in _BACKBONE}
        # Prefer canonical side-chain names; fall back to all non-backbone heavies.
        picked = {}
        for nm in sorted(want):
            if nm in by_name:
                picked[nm] = by_name[nm]
        if 'CB' not in picked:
            raise ValueError(
                f'{resn} {ch}{resi}: missing CB (required for theozyme frame)')
        if len(picked) < 2 and resn not in ('ALA',):
            # Keep whatever non-backbone heavies exist (incomplete side chain).
            for nm, a in by_name.items():
                picked.setdefault(nm, a)
        atom_map = {}
        role = 'anchor' if i == 0 else 'satellite'
        for nm, a in picked.items():
            atom_map[nm] = add_atom(a, f'{role}:{resn}')
        entry = dict(role=role, resn=resn, atoms=atom_map)
        residues_out.append(entry)

    lig_map = {}
    # Preserve PDB atom-name order (serial order).
    for a in sorted(lig_atoms, key=lambda x: x['serial']):
        if a['name'] in lig_map:
            raise ValueError(f'duplicate ligand atom name {a["name"]}')
        lig_map[a['name']] = add_atom(a, 'ligand')

    # Waters (optional).
    waters_out = []
    if include_waters:
        ref = np.array(coords, float)
        for wa in _nearby_waters(atoms, ref, radius=water_radius):
            wmap = {}
            for a in sorted(wa, key=lambda x: x['serial']):
                # Map O / OH2 → "O"; keep other heavy names as-is.
                nm = 'O' if a['name'] in ('O', 'OH2') else a['name']
                wmap[nm] = add_atom(a, 'water')
            if wmap:
                waters_out.append(wmap)

    bonds = _infer_bonds(lig_atoms, conect, bond_factor=bond_factor)
    # Drop any bond that names an atom we did not keep (should not happen).
    bonds = [list(b) for b in bonds if b[0] in lig_map and b[1] in lig_map]

    partial = []
    for pair in (partial_bonds or []):
        if isinstance(pair, str):
            a, b = pair.split(':')
            partial.append([a.strip(), b.strip()])
        else:
            partial.append([pair[0], pair[1]])
        for nm in partial[-1]:
            if nm not in lig_map:
                raise ValueError(f'partial bond atom {nm} not in ligand')
    # Prefer partial classification when the user marks a TS bond.
    if partial:
        part_set = {frozenset(p) for p in partial}
        bonds = [b for b in bonds if frozenset(b) not in part_set]

    cov = None
    if not no_covalent:
        cov = _find_covalent(
            links, cat[0][0], cat[0][1], lig_ch, lig_resi, set(lig_map),
            explicit=covalent, order=covalent_order)
        if cov:
            if cov['atom'] not in residues_out[0]['atoms']:
                raise ValueError(
                    f'covalent protein atom {cov["atom"]} not in anchor side chain')
            residues_out[0]['covalent_to_ligand'] = cov

    X = np.asarray(coords, float)
    data = dict(
        name=name or os.path.splitext(os.path.basename(path))[0],
        residues=residues_out,
        ligand=dict(resn=lig_resn, atoms=lig_map, bonds=bonds,
                    partial_bonds=partial),
        waters=waters_out,
        # Inline geometry (no separate XYZ file required).
        elements=el,
        coords=X.tolist(),
        source_pdb=os.path.abspath(path),
        source_ligand=f'{lig_ch}{lig_resi}',
        source_catalytic=[f'{c}{r}' for c, r in cat],
    )
    return data


def write_spec_bundle(data, out_json, out_xyz=None):
    """Write a reusable JSON + XYZ pair from a pdb_to_theozyme_dict result."""
    out_json = os.path.abspath(out_json)
    if out_xyz is None:
        out_xyz = os.path.splitext(out_json)[0] + '.xyz'
    out_xyz = os.path.abspath(out_xyz)
    el = data['elements']
    X = np.asarray(data['coords'], float)
    lines = [str(len(el)), data.get('name', 'theozyme from PDB')]
    for e, (x, y, z) in zip(el, X):
        lines.append(f'{e:2s} {x:12.6f} {y:12.6f} {z:12.6f}')
    os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
    open(out_xyz, 'w').write('\n'.join(lines) + '\n')

    dump = {k: v for k, v in data.items()
            if k not in ('elements', 'coords')}
    dump['xyz'] = out_xyz
    # Keep provenance keys for debugging.
    open(out_json, 'w').write(json.dumps(dump, indent=2) + '\n')
    return out_json, out_xyz
