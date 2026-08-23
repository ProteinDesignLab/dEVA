# Theozymes

A theozyme is a rigid QM geometry (XYZ) plus a JSON map that names catalytic
residues and the ligand. Placement grafts that geometry onto a scaffold PDB.

**Canonical public path:** a theozyme JSON + scaffold PDB →
`theozyme/prepare_placements.py` → ranked complex / ligand PDBs.

Shared explore → build → write logic lives in `theozyme/placements.py`.
Project-specific wrappers may exist for special design campaigns (extra YML
emission, electric-field wiring, etc.); for general use, stay on
`prepare_placements.py`.

---

## Theozyme JSON spec

Required files:

| File | Role |
|---|---|
| Spec JSON | Anchor / satellite / ligand atom map |
| TS XYZ | Same frame as the 1-based indices in the JSON |
| Scaffold PDB | Protein to graft onto |

### Top-level fields

```json
{
  "name": "my_theozyme",
  "xyz": "path/to/Transition_State.xyz",
  "residues": [ ... ],
  "ligand": { ... },
  "waters": [ { "O": 48 } ]
}
```

- **`xyz`** — path to the QM XYZ (resolved relative to the working directory when
  you run the CLI).
- **`residues`** — exactly one `role: "anchor"`, zero or more `role: "satellite"`.
- **`ligand`** — `resn`, atom name → XYZ index map, covalent `bonds`, optional
  `partial_bonds` (TS bonds drawn in the PDB).
- **`waters`** — optional list of `{ "O": <index> }` (and other atoms if present).

Atom indices are **1-based** into the XYZ. Each heavy atom may be claimed by only
one residue or the ligand (`TheozymeSpec` validates this).

### Residue roles

| Role | Purpose |
|---|---|
| **anchor** | Frame residue. χ1/χ2 about CB place the rigid theozyme onto the scaffold. Required: declare at least `CB`; usually also CG-equivalent. |
| **satellite** | Matched by CB onto scaffold hosts. Tip atoms (e.g. Tyr OH) become geometry / CST constraints. |
| **ligand** | Non-protein atoms written as residue 901. |

Optional on a residue: `"cg_atom": "CG"` if the CG-equivalent is not guessed from
the usual names (`CG`, `CG1`, `OG`, …).

### Illustrative specs

**Non-covalent** — omit `covalent_to_ligand`. The anchor is only the frame
residue (CB/CG define the graft). Catalytic geometry is tip⋯ligand distances
(and satellite CBs). No protein–ligand LINK.

```json
{
  "name": "my_theozyme",
  "xyz": "path/to/Transition_State.xyz",
  "residues": [
    {
      "role": "anchor",
      "resn": "SER",
      "atoms": { "CB": 1, "OG": 2 }
    },
    {
      "role": "satellite",
      "resn": "HIS",
      "atoms": {
        "CB": 10, "CG": 11, "ND1": 12, "CD2": 13,
        "CE1": 14, "NE2": 15
      }
    }
  ],
  "ligand": {
    "resn": "LIG",
    "atoms": { "C1": 20, "O1": 21, "C2": 22 },
    "bonds": [["C1", "O1"], ["C1", "C2"]],
    "partial_bonds": [["C2", "O1"]]
  }
}
```

**Covalent** — set `covalent_to_ligand` on the anchor. Placement still uses
χ1/χ2; the covalent tip locks ligand attachment and is written as a PDB LINK.
χ3+ torsions are taken from the QM adduct.

```json
{
  "role": "anchor",
  "resn": "LYS",
  "atoms": { "CB": 1, "CG": 2, "CD": 3, "CE": 4, "NZ": 5 },
  "covalent_to_ligand": {
    "atom": "NZ",
    "ligand_atom": "C1",
    "order": "double"
  }
}
```

Both modes share the same Explorer / build path.

---

## Authoring a new theozyme

1. Orient your TS in one XYZ frame; keep atom order stable.
2. Write a JSON (start from a snippet above) and point `"xyz"` at that file.
3. Map catalytic side-chain heavy atoms and ligand atoms to 1-based indices.
4. Declare bonds / partial_bonds for the ligand CONECT records.
5. Add `covalent_to_ligand` only if the chemistry is a covalent adduct.
6. Sanity-check:

```bash
python -c "from theozyme.spec import TheozymeSpec; print(TheozymeSpec('my_theozyme.json').summary())"
```

---

## What `prepare_placements.py` does

1. **Load** `TheozymeSpec` + scaffold; resolve satellite hosts
   (`--satellite-positions` and/or `--barrel-shell`).
2. **Explore** grafts: for each `--anchors` position × χ1/χ2 grid (`--chi-step`),
   place the rigid theozyme; match satellite CBs within `--max-cb-dev`.
3. **Filter** by occlusion (`--min-occ` / `--max-occ`); write
   `{name}_candidates.json`.
4. **Build** up to `--build-max` distinct assignments: mutate catalytic residues,
   optionally **accommodate** satellite backbone (`--sat-mode`, default `shift`),
   optional **protpardelle** partial diffusion, validate geometry.
5. **Rank** by built catalytic geometry / occlusion; keep `--top` (default: all
   built); write ranked PDBs + `{name}_report.json`.

Does **not** write dEVA campaign YMLs, reactant maps, `delta_mu`, or
electric-field wiring. After you have PDBs, point your own evolution configs at
them (see repo `configs/` examples).

---

## How to run

From the dEVA repo root:

### Non-covalent

```bash
python theozyme/prepare_placements.py \
  --scaffold inputs/scaffold.pdb \
  --theozyme-spec my_theozyme.json \
  --name my_theozyme \
  --anchors 40-60,80-100 \
  --satellite-positions 45,90,120 \
  --mobile 50-65 \
  --out-dir inputs/my_theozyme \
  --build-max 10 --top 5
```

### Covalent

```bash
python theozyme/prepare_placements.py \
  --scaffold inputs/scaffold.pdb \
  --theozyme-spec my_theozyme.json \
  --name my_theozyme_cov \
  --anchors 80-100 \
  --satellite-positions 50,120 \
  --out-dir inputs/my_theozyme_cov
```

Same CLI either way — covalent vs non-covalent is decided by the JSON
(`covalent_to_ligand` present or absent).

### Important flags

| Flag | Meaning |
|---|---|
| `--scaffold` | Protein PDB |
| `--theozyme-spec` | JSON above |
| `--anchors` | Candidate frame positions (`83` or `50-90,180-210`); GLY/PRO skipped |
| `--name` | Run / output basename |
| `--out-dir` | Output directory (default `inputs/<name>`) |
| `--chain` | Scaffold chain ID (default: first chain) |
| `--satellite-positions` | Restrict (or with `--barrel-shell`, expand) satellite hosts |
| `--barrel-shell [R]` | Auto β-strand shell around pocket (default R=13 Å if flag given with no value); optional TIM-barrel helper |
| `--mobile` | Loop ranges made mobile during build (`52-66,180-190`) |
| `--build-max` | How many assignments to build before ranking (default 25) |
| `--top` | How many ranked PDBs to keep (default: all built) |
| `--min-occ` / `--max-occ` | Occlusion floor / ceiling (default: no floor) |
| `--target-occ` | Recorded in the report only |
| `--chi-step` | χ1/χ2 grid step in degrees (default 4) |
| `--max-cb-dev` | Satellite CB tolerance before accommodation (default 2.2 Å) |
| `--no-accommodate` | Freeze satellite backbones |
| `--allow-bad-geometry` | Keep builds that fail L-amino validation |
| `--protpardelle` | Partial-diffusion backbone assist during build |
| `--n-protpardelle-attempts` | Attempts per build (default 5) |
| `--device` | e.g. `cuda` (default) |
| `--deva-root` | Repo root for resolving default `--out-dir` |

Pocket / shell helpers: `--pocket-pdb`, `--suggest-shell` (print CB-near-pocket
positions and exit). Satellite drive modes: `--sat-mode {shift,cb,soft,none}`
(plus `--max-shift`, `--sat-window`). Protpardelle extras:
`--protpardelle-repo`, `--noise-angstrom`, `--protpardelle-steps`.

See also [`models/relax.py`](../models/relax.py) for the
evolution-time relaxer (separate from `--protpardelle` during placement).

---

## Outputs

Under `--out-dir` (default `inputs/<name>/`):

| Artifact | Contents |
|---|---|
| `{name}_rank{i}.pdb` | Protein + ligand (resi 901); LINK if covalent |
| `{name}_rank{i}_ligand.pdb` | Ligand-only |
| `{name}_candidates.json` | Pre-build explore hits |
| `{name}_report.json` | Built ranks, maps, occlusion, sidechain metrics |

Then evolve with your own configs, for example:

```bash
python run.py -c configs/.../my_run.yml \
  --models seq_model ...
```

If you use `relax` in a campaign, list it **second** (right after
`seq_model`).
