# A catalytic pKa score

One aspct of enzyme design is to optimize the pKa of the catalytic site. If a mechanism cares about the protonation of one side chain, you can score that and add it
to `--models` like anything else. Here we use [PROPKA3](https://doi.org/10.1021/ct100578z) (Olsson, Søndergaard, Rostkowski & Jensen, *J. Chem. Theory Comput.* 2011, 7, 525–537) to score the pKa of the catalytic lysine.

A lysine in water sits near pKa 10.5. Nucleophilic chemistry usually wants the amine protonated to a low enough pKa to attack, but not so low that the next protonation step is dead. `lys_pka` asks PROPKA for the apo pKa of one lysine, then rewards environments that look like they belong in that window.

---

## What it scores

1. **Apo pKa** of the catalytic lysine (PROPKA; ligand HETATM stripped
   so you score the resting amine, not the adduct).
2. **A gate** that is 1.0 inside the window (default **7.0–8.5**) and
   falls to 0 outside 6–10.
3. **Context terms** (only if the gate is on): no extra ionizable side chains
   next to NZ, optional amide contacts, a little credit for a buried shift off 10.5.

Outside the window the geometry terms are zeroed. That stops the search from inventing a pretty H-bond network on a lysine that is still pKa 12. The window is defined by `pka_lo` and `pka_hi` in the config file.

The catalytic lysine is **not** hard-coded. It is the first residue in
`fixed_residues` (or set `lys_resi` yourself).

---

## How to use it

Install PROPKA once (`pip install propka`). Add a `lys_pka` block and
put the model in `--models` flag. Keep the lysine fixed by setting `fixed_from` to `ligandmpnn` and `lys_index` to 0.

```yaml
models:
  lys_pka:
    fixed_from: ligandmpnn
    lys_index: 0          # first fixed residue is the Lys
    chain: A
    ignore_ligand: true
    pka_lo: 7.0
    pka_hi: 8.5
```

```bash
python run.py -c configs/your_run.yml \
  --models seq_model lys_pka <other_models>
```

---

## Files

| file | role |
|---|---|
| [`models/lys_pka.py`](../models/lys_pka.py) | the score |
