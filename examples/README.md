# Examples

Worked examples of adding objectives to dEVA to demonstrate the depth and breadth of possible objective functions. 

We note that none of the objectives need be differentiable! Any method that returns a numerical score can be used as an objective. Methods can use either the sequence or structural information, or both!

Other objectives include hydrophobicity calculations, Rosetta binding energies, geometric counts of a local motif, an AlphaFold confidence or interface score, or an ML property such as pKa or expressibility in certain contexts. These scores should be tailored to specific properties of interest for your design campaign.

**[Add your own objective](add_your_own.md)** — after generating a your custom scoring function, add the file to `models/`, register the model, and simply add it to your run's flags `--models`.

We provide a few examples of functionalities that may be of interest for designing enzymes:

**[Placing a theozyme](example_theozyme_placement.md)** — how `prepare_placements.py` seats a theozyme or motif into a scaffold. 

**[Optional backbone relax](example_relax.md)** — flexible backbone relax or partial diffusion. In this case, `relax` is not a score. However, it can be used to adjust the backbone during theozyme placement or during design generations. Skip it for a fixed backbone; if you include it in `--models`, keep it second.

**[A theozyme and physics-based terms](example_physics.md)** — an example of physics-based objectibes that can be used in conjunction with a theozyme or other objectives. 

**[A substrate-aware score](example_substrate.md)** — a geometric objective for the pocket shape of a modeled substrate in catalytic contexts.

**[An catalytic pKa score](example_pka.md)** —  a predictive score for the catalytic lysine’s pKa (PROPKA-3). 

**[Metal3D / Metal3D-Cat](example_metal3d.md)** — p(metal) as an extra objective. This walks through the objective used in the manuscript.
