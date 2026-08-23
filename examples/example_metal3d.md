# Metal3D and Metal3D-Cat

The manuscript scores **P(catalytic metal)** with Metal3D-Cat. That is just another `--models` entry. The same wrapper runs **Metal3D** (any metal site) or **Metal3D-Cat** (catalytic metal site). 

`metal3d_model` scans through the current PDB and predicts based on the local chemical enviornments the probability and position of a metal site. It writes `pmetal` (higher is better). 

---

## Which weights

Same yaml block. Only `model_path` changes.

| weights | use when |
|---|---|
| `metal3d_cat.pth` | Metal3D-Cat for predicting a *catalytic* metal (the paper) |
| `metal3d_clean.pth` | Metal3D-Clean for predicting a generic metal site|

```yaml
models:
  metal3d:
    model_path: ./models/metal3d/weights/metal3d_cat.pth
    max_metal_p: 0.2
    threshold: 5
```

The CLI name is `metal3d_model`. The config section is `metal3d`.

```bash
python run.py -c configs/your_run.yml \
  --models seq_model metal3d_model
```

Add pocket shape, physics, pKa, or anything else on the same line. A demo that combines both Metal3D-Cat and pocket shape is available in [`example_substrate.md`](example_substrate.md).

---

## Files

| file | role |
|---|---|
| [`models/metal3d_model.py`](../models/metal3d_model.py) | the dEVA wrapper |
| `models/metal3d/weights/metal3d_cat.pth` | Metal3D-Cat |
| `models/metal3d/weights/metal3d_clean.pth` | Metal3D-Clean |
| `models/metal3d/weights/metal_0.5A_v3_d0.2_16Abox.pth` | Metal3D OG |

