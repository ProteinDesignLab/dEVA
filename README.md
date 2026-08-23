# dEVA: design by EVolutionary Algorithm

![image](assets/dEVA.png)

> [!NOTE]
> An updated version of dEVA is actively maintained at [gelnesr/dEVA](https://github.com/gelnesr/dEVA). If you have any issues, please report it there and we will get back to you ASAP.

dEVA was first introduced in [Zero-shot design of a de novo metalloenzyme](https://www.biorxiv.org/content/10.64898/2026.04.23.720277v1). The paper used sequence likelihood and catalytic-metal probability. The code is not limited to that pair.

**If a method returns a number from a sequence or a structure, it can be an objective.** Nothing needs to be differentiable. For example, metal sites, theozymes, physics terms, geometric filters, or property predictors all plug in the same way.

---

## How a run is put together

Each score is a small Python class we call a **model**. It looks at the current design (PDB and/or sequence) and returns one or more numbers. Higher is better. You give the model a name and list that name on `--models`.

```bash
python run.py -c configs/your_run.yml \
  --models seq_model <optional relax> <your scores>
```

### Functionalities available in dEVA
- `seq_model` is always first (LigandMPNN or ProteinMPNN).
- `relax` is optional and **not** a score. If you use it, keep it second so later terms see the moved backbone. See [`examples/example_relax.md`](examples/example_relax.md).
- Everything after that is an objective. Order is the order they run.

Worked campaigns live in [`examples/`](examples/). The paper’s metal score is documented there as [`examples/example_metal3d.md`](examples/example_metal3d.md) — Metal3D and Metal3D-Cat share one wrapper; you swap the checkpoint.

---

## The dEVA design process can be applied to any protein design problem

The manuscript designed a catalytic metal site. The design process can be applied to any protein design problem. For example, it can be used to design a theozyme, a substrate analog, or a single ionizable residue. Swap the scores, not the engine. Anything that returns a number can sit on `--models` next to `seq_model`.

---

To add a new score, see [`examples/add_your_own.md`](examples/add_your_own.md).

---

## Examples

See [`examples/README.md`](examples/README.md) for the full list. Short version:

- [Place a theozyme](examples/example_theozyme_placement.md)
- [Optional relax](examples/example_relax.md)
- [Physics terms on a posed theozyme](examples/example_physics.md)
- [Metal3D / Metal3D-Cat](examples/example_metal3d.md)
- [Substrate / pocket](examples/example_substrate.md)
- [Catalytic pKa](examples/example_pka.md)
- [Add your own objective](examples/add_your_own.md)

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
