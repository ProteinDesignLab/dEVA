# dEVA: design by EVolutionary Algorithm

![image](assets/dEVA.png)

> [!NOTE]
> An updated version of dEVA is actively maintained at [gelnesr/dEVA](https://github.com/gelnesr/dEVA). If you have any issues, please report it there and we will get back to you ASAP.

dEVA was introduced in [Zero-shot design of a de novo metalloenzyme](https://www.biorxiv.org/content/10.64898/2026.04.23.720277v1) to design metalloproteins and metalloenzymes. This repository contains the code and examples from the paper, as well as additional examples and functionalities of how dEVA can be used as a general-purpose protein design method.

A dEVA design proposal proposes sequence mutations, then scores each design on more than one objective at once via a genetic algorithm. After the run, the top designs on the Pareto front are returned. 

In the paper those objectives were:

- **p(seq)** — does this sequence belong on this backbone? *(LigandMPNN)*
- **p(catalytic metal)** — is there still a catalytic metal at the site? *(Metal3D-Cat)*

```bash
python run.py -c configs/your_run.yml --models seq_model metal3d_model
```

Metal3D and Metal3D-Cat use the same wrapper; to toggle between them, swap the checkpoint. See [`examples/example_metal3d.md`](examples/example_metal3d.md).

---

## dEVA as a general-purpose design method

**If a method returns a number from a sequence or a structure, it can be an objective.** Nothing needs to be differentiable, just swap the scores or add new ones.

Each score is a small Python class (a **model**). It looks at the current design and returns one or more numbers. Higher is better. You name it and list that name on `--models`.

```bash
python run.py -c configs/your_run.yml --models seq_model <optional relax> <your scores>
```

`seq_model` is always first. Everything after that is an objective, in the order they run.

---

## Examples

A full list of examples can be found at [`examples/README.md`](examples/README.md).

---

## Additional functionalities

These are other possible additions to the dEVA design loop.

| example functionality | what it is | where |
|----------------------|------------|-------|
| **Theozyme placement** | Seat a QM geometry in a scaffold before you evolve | [`examples/example_theozyme_placement.md`](examples/example_theozyme_placement.md) |
| **Flexible backbone** | Move nearby backbone during design. | [`examples/example_relax.md`](examples/example_relax.md) |
| **Physics terms** | Ex. electric field, desolvation, etc. | [`examples/example_physics.md`](examples/example_physics.md) |
| **Geometric terms** | Ex. geometric enclosure of a fixed ligand pose | [`examples/example_substrate.md`](examples/example_substrate.md) |
| **Predictive scores** | Ex. apo pKa of one lysine (PROPKA3) | [`examples/example_pka.md`](examples/example_pka.md) |

---

## How to add a your own score

Write a file in `models/`, register a name, add a yaml block, put that name on `--models`.

Step-by-step template: [`examples/add_your_own.md`](examples/add_your_own.md).

---

## Citation

If you are using our code, datasets, or model, please use the following citation:

```bibtex
@article {ElNesr-2026,
    author = {El Nesr, Gina and Duerr, Simon L. and Mathews, Irimpan I. and Wen, Qi and Zhao, Kewei and Sarangi, Ritimukta and Roethlisberger, Ursula and Sunden, Fanny and Huang, Possu},
    title = {Zero-shot design of a de novo metalloenzyme},
    year = {2026},
    doi = {10.64898/2026.04.23.720277},
    journal = {bioRxiv}
}
```
