# Examples

Worked examples of adding objectives to dEVA to demonstrate the depth and breadth of possible objective functions. 

We note that none of the objectives need be differentiable! Any method that returns a numerical score can be used as an objective. Methods can use either the sequence or structural information, or both!

These are all prototypes — neither is benchmarked or experimentally validated. Success is dependent on the biochemical
or biophysical intution that goes into the objectives. 

**[Physics-based terms](example_physics.md)** — electric field and desolvation objectives physics-based objectives
applied to the de novo retro-aldolase RA95. Two objectives with no training data behind
them: a Coulomb sum closed with a QM expansion, and a Born term. 

**[A theozyme and a substrate](example_substrate.md)** — pocket shape against a
theozyme on a designed metalloenzyme, run alongside sequence likelihood and catalytic-metal probability. 
A geometric objective on a traditionally-generated design to improve pocket accessibility and shape complementarity.