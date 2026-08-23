# Add your own objective

Three steps. Drop a file in `models/`, give it a name, list that name on `--models`.

**If a method returns a number from a sequence or a structure, it can be an objective.** Nothing needs to be differentiable.

---

## 1. Create `models/my_objective.py`

```python
from typing import Dict
from core.interfaces import BaseModel
from core.registry import register_model
from evolve.individual import Individual

@register_model("my_objective")
class MyObjective(BaseModel):
    def __init__(self):
        pass

    def setup(self, config: Dict, device: str = "cpu") -> None:
        self.config = config
        self.cfg = config.models.my_objective

    def score(self, individual: Individual):
        pdb = individual.get_name()
        value = 0.0  # your number, higher = better
        individual.add_fitness({"my_objective": float(value)})
```

`models/__init__.py` imports every file in that folder. The string in `@register_model(...)` is the name you type on the command line.

The individual gives you:

- `get_name()` — path to the current PDB
- `get_gen()` / `get_index()` — generation and index
- `add_fitness({...})` — one dict of name → float
- `update_name(path)` — only if you wrote a new PDB

---

## 2. Add a yaml block

Same name, under `models:`:

```yaml
models:
  my_objective:
    any_param: 1.0
```

Read it in `setup()` as `config.models.my_objective`.

---

## 3. Run it

```bash
python run.py -c configs/your_run.yml \
  --models seq_model my_objective
```

`seq_model` stays first. Put `relax` second if you use it. You can add several scores on that line. Each key you pass to `add_fitness()` becomes its own axis on the Pareto front — emit one key unless you really want more objectives.

---

## Copy from these

| file | what it is |
|---|---|
| [`models/lys_pka.py`](../models/lys_pka.py) | property score (PROPKA) |
| [`models/pocket_shape.py`](../models/pocket_shape.py) | geometry on a PDB |
| [`models/metal3d_model.py`](../models/metal3d_model.py) | loaded network weights |

How those look in a campaign: [`README.md`](README.md).
