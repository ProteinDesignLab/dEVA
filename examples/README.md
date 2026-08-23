# Examples

Worked examples of adding objectives to dEVA to demonstrate the depth and breadth of possible objective functions. 

We note that none of the objectives need be differentiable! Any method that returns a numerical score can be used as an objective. Methods can use either the sequence or structural information, or both!

Anything else that returns a number can be added the same way. That includes hydrophobicity calculations, Rosetta binding energies, geometric counts of a local motif, an AlphaFold confidence or interface score, or an ML property such as pKa or expressibility in certain contexts. These scores should be tailored to specific properties of interest for the design campaign.

**[Add your own objective](add_your_own.md)** — drop a file in `models/`, name it, put that name on `--models`.

We provide a few examples of functionalities that may be of interest for designing enzymes:

**[Placing a theozyme](example_theozyme_placement.md)** — how `prepare_placements.py` seats a QM theozyme in a scaffold (JSON spec, search, ranked PDBs). Worked example: RA95 in a barrel, which the physics example then evolves.

**[Optional backbone relax](example_relax.md)** — `relax` is not a score. It can nudge backbone at placement and/or between generations. Skip it for a fixed backbone; if you include it in `--models`, keep it second.

**[A theozyme and physics-based terms](example_physics.md)** — Using a simple theozyme placed into the RA95 barrel (`configs/ra95_example.yml`). Electric field and desolvation scores with no training data behind them, plus sequence likelihood and pocket shape. 

**[Metal3D / Metal3D-Cat](example_metal3d.md)** — p(metal) as an extra objective. This walks through the objective used in the manuscript.

**[A substrate-aware score](example_substrate.md)** — pocket shape on a designed metalloenzyme (`configs/substrate_example.yml`), alongside sequence likelihood and catalytic-metal probability.

**[An catalytic pKa score](example_pka.md)** — extra objective for the catalytic lysine’s pKa (PROPKA + a local window). 