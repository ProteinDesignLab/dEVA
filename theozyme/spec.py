"""Declarative theozyme description.

A theozyme is: one FRAME residue (``role: "anchor"``), zero or more SATELLITE
residues (matched by CB onto scaffold positions), a LIGAND, and optional waters.

Sources
-------
- **JSON + XYZ** (canonical): atom indices into a QM geometry file.
- **PDB**: catalytic residues + ligand HETATM, converted via
  ``TheozymeSpec.from_pdb`` / ``theozyme.pdb_import`` (roles still required).

Covalent vs non-covalent
------------------------
- **Covalent** (e.g. Lys–Schiff base): set ``covalent_to_ligand`` on the
  anchor. Placement still uses χ1/χ2 about CB; the covalent tip locks ligand
  attachment and is written as a PDB LINK.
- **Non-covalent**: omit ``covalent_to_ligand``. The anchor is only the *frame*
  residue for rigid-body search (CB/CG define the graft). Catalytic geometry is
  tip⋯ligand distances (and satellite CBs), not a bond. No LINK is written.

Both modes share the same Explorer / build path (``theozyme.placements`` /
``prepare_placements.py``). Campaign-specific EF wiring stays in prepare_deva.
"""
import json
import os

import numpy as np

from .geometry import dihedral
from .theozyme import read_xyz


class ResidueSpec:
    def __init__(self, d):
        self.role = d['role']; self.resn = d['resn']
        self.atoms = {k: int(v) for k, v in d['atoms'].items()}   # 1-based XYZ indices
        # Optional. Present => covalent adduct; absent => non-covalent placement.
        self.covalent = d.get('covalent_to_ligand')
        for req in ('CB',):
            if req not in self.atoms:
                raise ValueError(f'{self.resn}: theozyme residue must declare {req}')
        self.cg = d.get('cg_atom') or self._guess_cg()

    def _guess_cg(self):
        for c in ('CG', 'CG1', 'OG', 'OG1', 'SG', 'CB'):
            if c in self.atoms:
                return c
        raise ValueError(f'{self.resn}: cannot identify the CG-equivalent atom; set "cg_atom"')


class TheozymeSpec:
    def __init__(self, path=None, *, data=None):
        """Load from a JSON path, a ``.pdb`` path, or an in-memory ``data`` dict.

        PDB paths require catalytic roles to be present in ``data`` already, or
        use ``TheozymeSpec.from_pdb(...)`` / pass kwargs through ``load_theozyme``.
        """
        if data is None:
            if path is None:
                raise ValueError('TheozymeSpec needs path or data=')
            path = os.path.abspath(path)
            ext = os.path.splitext(path)[1].lower()
            if ext == '.pdb':
                raise ValueError(
                    f'{path} looks like a PDB. Use TheozymeSpec.from_pdb(...) '
                    f'or prepare_placements --theozyme-pdb with --catalytic')
            data = json.load(open(path))
            self.spec_path = path
        else:
            self.spec_path = path
        self._init_from_dict(data)

    def _init_from_dict(self, d):
        self.raw = d
        self.name = d.get('name', 'theozyme')
        if 'coords' in d and 'elements' in d:
            self.el = list(d['elements'])
            self.X = np.asarray(d['coords'], float)
            if self.X.ndim != 2 or self.X.shape[1] != 3:
                raise ValueError('coords must be an (N, 3) array')
            if len(self.el) != len(self.X):
                raise ValueError('elements / coords length mismatch')
            self.xyz_path = d.get('source_pdb') or d.get('xyz') or '(inline)'
            self.title = d.get('name', 'inline theozyme')
        else:
            self.xyz_path = d['xyz']
            self.el, self.X, self.title = read_xyz(d['xyz'])
        self.residues = [ResidueSpec(r) for r in d['residues']]
        anch = [r for r in self.residues if r.role == 'anchor']
        if len(anch) != 1:
            raise ValueError('exactly one residue must have role "anchor"')
        self.anchor = anch[0]
        self.satellites = [r for r in self.residues if r.role == 'satellite']
        L = d['ligand']
        self.lig_resn = L.get('resn', 'LIG')
        self.lig_atoms = {k: int(v) for k, v in L['atoms'].items()}
        self.lig_bonds = [tuple(b) for b in L.get('bonds', [])]
        self.partial_bonds = [tuple(b) for b in L.get('partial_bonds', [])]
        self.waters = [{k: int(v) for k, v in w.items()} for w in d.get('waters', [])]
        self.validate()

    @classmethod
    def from_dict(cls, data):
        return cls(data=data)

    @classmethod
    def from_pdb(cls, path, catalytic, **kwargs):
        """Build a spec from a theozyme PDB (see ``theozyme.pdb_import``)."""
        from .pdb_import import pdb_to_theozyme_dict
        data = pdb_to_theozyme_dict(path, catalytic, **kwargs)
        return cls(data=data)

    def p(self, i):
        return self.X[i - 1]

    def validate(self):
        n = len(self.X)
        seen = {}
        for r in self.residues:
            for nm, i in r.atoms.items():
                if not 1 <= i <= n:
                    raise ValueError(f'{r.resn}.{nm}: index {i} out of range 1..{n}')
                if i in seen:
                    raise ValueError(f'atom {i} claimed by both {seen[i]} and {r.resn}.{nm}')
                seen[i] = f'{r.resn}.{nm}'
        for nm, i in self.lig_atoms.items():
            if i in seen:
                raise ValueError(f'ligand atom {nm} ({i}) also claimed by {seen[i]}')
            seen[i] = f'LIG.{nm}'
        if self.anchor.covalent:
            a = self.anchor.covalent['atom']
            b = self.anchor.covalent['ligand_atom']
            if a not in self.anchor.atoms:
                raise ValueError(f'covalent atom {a} not declared')
            if b not in self.lig_atoms:
                raise ValueError(f'ligand atom {b} not declared')

    def locked_torsions(self):
        """Torsions of the anchor side chain fixed by the QM (chi3 onward).

        chi1 and chi2 need CA, which model compounds often lack, so they stay free
        for the scaffold search. Meaningful for covalent adducts with a long chain
        (e.g. Lys); may be empty for short non-covalent side chains.
        """
        a = self.anchor
        order = self._chain_order(a)
        out = {}
        for k in range(len(order) - 3):
            nm = f'chi{k + 3}'
            out[nm] = float(dihedral(*[self.p(a.atoms[order[k + j]]) for j in range(4)]))
        return out

    def _chain_order(self, r):
        pref = ['CB', 'CG', 'CG1', 'OG', 'OG1', 'SG', 'CD', 'CD1', 'SD', 'NE',
                'CE', 'OE1', 'NZ', 'CZ', 'OH', 'NH1']
        return [a for a in pref if a in r.atoms]

    def summary(self):
        L = [f'theozyme: {self.name}   ({len(self.X)} atoms, {self.title})',
             f'  source   : {self.xyz_path}',
             f'  anchor    : {self.anchor.resn}  atoms {sorted(self.anchor.atoms)}']
        if self.anchor.covalent:
            c = self.anchor.covalent
            d = np.linalg.norm(self.p(self.anchor.atoms[c['atom']])
                               - self.p(self.lig_atoms[c['ligand_atom']]))
            L.append(f'              covalent {c["atom"]}-{c["ligand_atom"]} = {d:.3f} A')
        else:
            L.append('              non-covalent (no covalent_to_ligand; frame residue only)')
        for s in self.satellites:
            d = np.linalg.norm(self.p(self.anchor.atoms['CB']) - self.p(s.atoms['CB']))
            L.append(f'  satellite : {s.resn}  CB-CB from anchor = {d:.2f} A')
        L.append(f'  ligand    : {self.lig_resn}, {len(self.lig_atoms)} atoms, '
                 f'{len(self.lig_bonds)} bonds, {len(self.partial_bonds)} partial')
        locked = self.locked_torsions()
        if locked:
            for k, v in locked.items():
                L.append(f'  locked    : {k} = {v:+.2f} deg')
        else:
            L.append('  locked    : (none — short side chain / no χ3+)')
        L.append('  free      : chi1, chi2 of the anchor  ->  2-parameter search')
        return '\n'.join(L)


def load_theozyme(path, catalytic=None, **pdb_kwargs):
    """Load JSON or PDB. For PDB, ``catalytic`` is required (anchor[,satellites])."""
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdb':
        if not catalytic:
            raise ValueError(
                'PDB theozyme requires catalytic residues '
                '(e.g. catalytic="83,51" — first is the anchor)')
        return TheozymeSpec.from_pdb(path, catalytic, **pdb_kwargs)
    return TheozymeSpec(path)
