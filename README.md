# dEVA: design by EVolutionary Algorithm

![image](assets/dEVA.png)

> [!NOTE]
> An updated version of dEVA is actively maintained at [gelnesr/dEVA](https://github.com/gelnesr/dEVA). If you have any issues, please report it there and we will get back to you ASAP.

dEVA designs **metalloproteins and metalloenzymes** by evolution. It was introduced in [Zero-shot design of a de novo metalloenzyme](https://www.biorxiv.org/content/10.64898/2026.04.23.720277v1).

A run proposes sequences, then scores each design on more than one number at once. In the paper those numbers were:

- **p(seq)** — does this sequence belong on this backbone? *(LigandMPNN)*
- **p(catalytic metal)** — is there still a catalytic metal at the site? *(Metal3D-Cat)*

After the run, the top designs on the Pareto front are returned. 

```bash
python run.py -c configs/your_run.yml \
  --models seq_model metal3d_model
```

Metal3D and Metal3D-Cat use the same wrapper; to toggle between them, swap the checkpoint. See [`examples/example_metal3d.md`](examples/example_metal3d.md).

---

## dEVA as a general-purpose design method

**If a method returns a number from a sequence or a structure, it can be an objective.** Nothing needs to be differentiable, just swap the scores or add new ones.

Each score is a small Python class (a **model**). It looks at the current design and returns one or more numbers. Higher is better. You name it and list that name on `--models`.

```bash
python run.py -c configs/your_run.yml \
  --models seq_model <optional relax> <your scores>
```

`seq_model` is always first. Everything after that is an objective, in the order they run.

---

## Examples

Full list: [`examples/README.md`](examples/README.md).

**From the paper**

- [Metal3D / Metal3D-Cat](examples/example_metal3d.md) — p(metal), catalytic or general
- [A substrate-aware score](examples/example_substrate.md) — pocket shape next to sequence and Metal3D-Cat

**Other examples of objectives**

- [Place a theozyme](examples/example_theozyme_placement.md)
- [Physics terms on a posed theozyme](examples/example_physics.md)
- [A catalytic pKa score](examples/example_pka.md)

---

## Additional functionalities

These are other possible additions to the dEVA design loop.

| example functionality | what it is | where |
|----------------------|------------|-------|
| **Theozyme placement** | Seat a QM geometry in a scaffold before you evolve | [`examples/example_theozyme_placement.md`](examples/example_theozyme_placement.md) |
| **relax** | Move nearby backbone. Not a score. If you use it, keep it second in `--models` | [`examples/example_relax.md`](examples/example_relax.md) |
| **Physics terms** | Ex. electric field and desolvation as ordinary objectives | [`examples/example_physics.md`](examples/example_physics.md) |
| **Geometric terms** | Ex. geometric enclosure of a fixed ligand pose | [`examples/example_substrate.md`](examples/example_substrate.md) |
| **Predictive scores** | Apo pKa of one lysine (PROPKA + a window) | [`examples/example_pka.md`](examples/example_pka.md) |

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
