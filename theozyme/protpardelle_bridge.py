"""Protpardelle-1c backbone relaxer for dEVA.

Imports ``protpardelle`` from a ``protpardelle-1c`` checkout
(``src/protpardelle/…``, Zenodo ``model_params/``) and calls ``model.sample``
in-process.

Relaxation matches 1c ``sampling_partial_diffusion``: rewind the full structure
on the native noise schedule, then denoise. Motif crop-cond / replacement is
**not** used here — that path fought PD and shredded peptide geometry. Theozyme
pinning is enforced after sampling by ``accommodate`` / ``RigidTheozyme``.
"""
import os
import sys
import tempfile
from contextlib import contextmanager, nullcontext

import numpy as np

from .geometry import kabsch
from .structure import BACKBONE


class _NoOpTqdm:
    """Drop-in for ``tqdm`` that never draws a bar (manual or iterable use)."""

    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = range(0) if iterable is None else iterable
        self.n = 0
        self.total = kwargs.get('total')

    def __iter__(self):
        return iter(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, n=1):
        self.n += n

    def close(self):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def refresh(self):
        pass

    def clear(self, *args, **kwargs):
        pass

    def reset(self, total=None):
        if total is not None:
            self.total = total
        self.n = 0


@contextmanager
def _silence_tqdm():
    """Silence 1c sample noise (tqdm bar + per-draw INFO).

    protpardelle-1c does ``from tqdm.auto import tqdm`` and builds the bar
    directly — it does **not** honor ``tqdm_pbar=``. ``TQDM_DISABLE=1`` only
    works on tqdm>=4.66. Monkeypatching the already-bound ``tqdm`` name in
    imported protpardelle modules works across versions without editing 1c.

    Also raises ``protpardelle.core.models`` to WARNING so each PD draw does
    not spam ``Partial diffusion, going back to step …`` (no 1c env knob).
    """
    import logging

    prev = os.environ.get('TQDM_DISABLE')
    os.environ['TQDM_DISABLE'] = '1'
    patches = []  # (module, attr, original)
    pp_log = logging.getLogger('protpardelle.core.models')
    prev_level = pp_log.level

    def _patch(mod, attr, replacement):
        if mod is None or not hasattr(mod, attr):
            return
        cur = getattr(mod, attr)
        if cur is replacement:
            return
        patches.append((mod, attr, cur))
        setattr(mod, attr, replacement)

    try:
        import tqdm as tqdm_mod
        _patch(tqdm_mod, 'tqdm', _NoOpTqdm)
        try:
            import tqdm.auto as tqdm_auto
            _patch(tqdm_auto, 'tqdm', _NoOpTqdm)
        except ImportError:
            pass
    except ImportError:
        pass

    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if name != 'protpardelle' and not name.startswith('protpardelle.'):
            continue
        obj = getattr(mod, 'tqdm', None)
        if obj is None or obj is _NoOpTqdm:
            continue
        # Only replace names that look like the real tqdm constructor.
        obj_mod = getattr(obj, '__module__', '') or ''
        if 'tqdm' in obj_mod or getattr(obj, '__name__', '') == 'tqdm':
            _patch(mod, 'tqdm', _NoOpTqdm)

    pp_log.setLevel(logging.WARNING)
    try:
        yield
    finally:
        pp_log.setLevel(prev_level)
        for mod, attr, original in reversed(patches):
            try:
                setattr(mod, attr, original)
            except Exception:
                pass
        if prev is None:
            os.environ.pop('TQDM_DISABLE', None)
        else:
            os.environ['TQDM_DISABLE'] = prev

# Duplicated from protpardelle so this module imports without the repo present.
ATOM37 = ["N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD",
          "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
          "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2",
          "CZ3", "NZ", "OXT"]
ATOM37_INDEX = {a: i for i, a in enumerate(ATOM37)}
BB_IDXS = [ATOM37_INDEX[a] for a in ("N", "CA", "C", "O")]      # [0, 1, 2, 4]

RESTYPES = list("ARNDCQEGHILKMFPSTWYV")
RESTYPE_INDEX = {a: i for i, a in enumerate(RESTYPES)}
AA3to1 = dict(ALA='A', ARG='R', ASN='N', ASP='D', CYS='C', GLN='Q', GLU='E', GLY='G',
              HIS='H', ILE='I', LEU='L', LYS='K', MET='M', PHE='F', PRO='P', SER='S',
              THR='T', TRP='W', TYR='Y', VAL='V')
AA1to3 = {v: k for k, v in AA3to1.items()}

SIGMA_DATA_DEFAULT = 10.01

# Defaults from examples/sampling/01_partial_diffusion.yaml
_1C_DEFAULTS = {
    'backbone': dict(model_name='cc58', model_epoch=416),
    'allatom': dict(model_name='cc89', model_epoch=415),
}

_REQUIRED_MODULES = [
    ('torch', 'torch'),
    ('einops', 'einops'),
    ('jaxtyping', 'jaxtyping'),
    ('Bio', 'biopython'),
    ('omegaconf', 'omegaconf'),
    ('hydra', 'hydra-core'),
    ('ml_collections', 'ml_collections'),
    ('tqdm', 'tqdm'),
    ('yaml', 'pyyaml'),
]

# Official PD configs set conditional_cfg.enabled: false.
_PD_COND_CFG = {
    'enabled': False,
    'discontiguous_motif_assignment': {
        'enabled': False, 'strategy': 'fixed', 'fixed_motif_pos': [],
    },
    'num_recurrence_steps': 1,
    'crop_conditional_guidance': {
        'enabled': False, 'start': 0.0, 'end': 2.0, 'freq': 1,
        'freq_start': 0.0, 'freq_end': 0.0, 'strategy': 'backbone-sidechain',
    },
    'reconstruction_guidance': {
        'enabled': False, 'start': 0.0, 'end': 2.0, 'schedule': 'custom',
        'max_scale': 10.0, 'loss_weights': {'motif': 1.0},
    },
    'replacement_guidance': {'enabled': False, 'start': 0.0, 'end': 0.0},
}


class ProtpardelleUnavailable(RuntimeError):
    """Raised when the protpardelle-1c repo or weights are not usable."""


def structure_to_atom37(st, chain=None):
    """dEVA Structure -> (coords, mask, aatype, resis)."""
    chain = chain or str(st.chain[0])
    resis = [int(r) for c, r in st.protein_res if c == chain]
    n = len(resis)
    coords = np.zeros((n, 37, 3), float)
    mask = np.zeros((n, 37), float)
    aatype = np.zeros(n, np.int64)
    for k, r in enumerate(resis):
        idx = st.residues[(chain, r)]
        aatype[k] = RESTYPE_INDEX.get(AA3to1.get(str(st.resn[idx[0]]), 'G'), 7)
        for i in idx:
            nm = str(st.name[i])
            j = ATOM37_INDEX.get(nm)
            if j is None:
                continue
            coords[k, j] = st.xyz[i]
            mask[k, j] = 1.0
    return coords, mask, aatype, np.asarray(resis, int)


def atom37_backbone_into(st, xyz, coords37, resis, chain=None, names=BACKBONE):
    """Write N/CA/C/O from atom37 back into a full-structure coordinate array."""
    chain = chain or str(st.chain[0])
    out = np.asarray(xyz, float).copy()
    pos = {int(r): k for k, r in enumerate(resis)}
    for c, r in st.protein_res:
        if c != chain or int(r) not in pos:
            continue
        k = pos[int(r)]
        for i in st.residues[(c, r)]:
            nm = str(st.name[i])
            if nm in names and nm in ATOM37_INDEX:
                out[i] = coords37[k, ATOM37_INDEX[nm]]
    return out


def alignment_indices(resis, motif_resis, window=8, min_atoms=12):
    """Residue slots used to superpose a relaxed proposal back into the lab frame."""
    motif = set(int(r) for r in motif_resis)
    if not motif:
        return np.arange(len(resis))
    keep = [k for k, r in enumerate(resis)
            if any(abs(int(r) - m) <= window for m in motif)]
    if len(keep) * 3 < min_atoms:
        return np.arange(len(resis))
    return np.asarray(keep, int)


def superpose_backbone(moved37, ref37, mask37, slots):
    """Kabsch the proposal back onto the reference using N/CA/C of `slots`."""
    sel = [(k, j) for k in slots for j in BB_IDXS[:3] if mask37[k, j] > 0]
    if len(sel) < 4:
        raise ValueError('too few backbone atoms to superpose on')
    P = np.array([moved37[k, j] for k, j in sel])
    Q = np.array([ref37[k, j] for k, j in sel])
    R, t = kabsch(P, Q)
    rms = float(np.sqrt((((R @ P.T).T + t - Q) ** 2).sum(1).mean()))
    out = np.einsum('ij,naj->nai', R, moved37) + t
    return out, rms


def _write_atom37_pdb(path, coords37, mask37, aatype, residue_index=None):
    """ATOM-only PDB for 1c load_feats_from_pdb / partial_diffusion."""
    if residue_index is None:
        residue_index = np.arange(1, len(aatype) + 1)
    lines = []
    serial = 1
    for k, aa in enumerate(aatype):
        resn = AA1to3.get(RESTYPES[int(aa)] if int(aa) < len(RESTYPES) else 'G', 'GLY')
        resi = int(residue_index[k])
        for j, name in enumerate(ATOM37):
            if mask37[k, j] <= 0:
                continue
            x, y, z = coords37[k, j]
            nm = f'{name:>3s}' if len(name) < 4 else name[:4]
            lines.append(
                f'ATOM  {serial:5d} {nm:4s} {resn:3s} A{resi:4d}    '
                f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]:>2s}'
            )
            serial += 1
    lines.append('END')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


class ProtpardelleRelaxer:
    """Wraps protpardelle-1c ``sample`` as propose → coords (plain partial diffusion)."""

    def __init__(self, repo_dir=None, checkpoint=None, config=None, task='backbone',
                 device='cuda', noise_angstrom=1.5, n_steps=100, schedule_steps=500,
                 step_scale=1.0, s_churn=0.0, align_window=8,
                 minimpnn_checkpoint=None, denoiser=None, verbose=False,
                 model_name=None, model_epoch=None, model_params_dir=None,
                 disable_tqdm=True,
                 **_ignored):
        # **_ignored swallows removed knobs (use_replacement, use_crop_cond, …)
        # so old YMLs / callers do not crash.
        self.repo_dir = repo_dir
        self.checkpoint = checkpoint
        self.config = config
        self.task = task
        self.device = device
        self.noise_angstrom = float(noise_angstrom)
        self.n_steps = int(n_steps)
        self.schedule_steps = int(schedule_steps)
        self.step_scale = float(step_scale)
        self.s_churn = float(s_churn)
        self.align_window = int(align_window)
        self.minimpnn_checkpoint = minimpnn_checkpoint
        self.verbose = verbose
        # Silence 1c's hardcoded "Sampling backbones" bar (monkeypatch + TQDM_DISABLE).
        # No protpardelle-1c edit needed. Default True.
        self.disable_tqdm = bool(disable_tqdm)
        self.model_name = model_name
        self.model_epoch = None if model_epoch is None else int(model_epoch)
        self.model_params_dir = model_params_dir
        self._model = None
        self._denoiser = denoiser
        self.sigma_data = SIGMA_DATA_DEFAULT

    def _defaults(self):
        return _1C_DEFAULTS.get(self.task, _1C_DEFAULTS['backbone'])

    def _params_root(self):
        if self.model_params_dir:
            return self.model_params_dir
        env = os.environ.get('PROTPARDELLE_MODEL_PARAMS')
        if env:
            return env
        return os.path.join(self.repo_dir or '', 'model_params')

    def default_checkpoint(self):
        d = self._defaults()
        name = self.model_name or d['model_name']
        epoch = self.model_epoch or d['model_epoch']
        return os.path.join(
            self._params_root(), 'weights', f'{name}_epoch{epoch}.pth')

    def default_config(self):
        d = self._defaults()
        name = self.model_name or d['model_name']
        primary = os.path.join(self._params_root(), 'configs', f'{name}.yaml')
        if os.path.isfile(primary):
            return primary
        return os.path.join(
            self.repo_dir or '', 'examples', 'training', f'{name}.yaml')

    def missing_dependencies(self):
        import importlib.util
        out = []
        for mod, pip_name in _REQUIRED_MODULES:
            try:
                if importlib.util.find_spec(mod) is None:
                    out.append(pip_name)
            except (ImportError, ValueError):
                out.append(pip_name)
        return out

    def preflight(self):
        """Check everything the real model needs."""
        problems = []
        if self._denoiser is not None:
            return problems
        if not self.repo_dir or not os.path.isdir(self.repo_dir):
            problems.append(f'protpardelle-1c repo not found at {self.repo_dir!r}')
            return problems
        src = os.path.join(self.repo_dir, 'src', 'protpardelle')
        if not os.path.isdir(src):
            problems.append(
                f'{src} not found — repo_dir must point at a protpardelle-1c checkout')
            return problems
        missing = self.missing_dependencies()
        if missing:
            problems.append('missing python packages: ' + ', '.join(missing)
                            + f'  (pip install {" ".join(missing)})')
        params = self._params_root()
        if not os.path.isdir(params):
            problems.append(
                f'model_params missing at {params}. '
                f'Run:  cd {self.repo_dir} && bash download_model_params.sh')
        ckpt = self.checkpoint or self.default_checkpoint()
        cfg = self.config or self.default_config()
        if not os.path.isfile(ckpt):
            problems.append(
                f'checkpoint missing: {ckpt}. '
                f'Run:  cd {self.repo_dir} && bash download_model_params.sh')
        elif os.path.getsize(ckpt) < 100_000:
            with open(ckpt, 'rb') as f:
                head = f.read(64)
            if head.startswith(b'version https://git-lfs'):
                problems.append(
                    f'{ckpt} is a git-lfs pointer, not the weights. '
                    f'Run:  cd {self.repo_dir} && git lfs install && git lfs pull')
            else:
                problems.append(f'checkpoint suspiciously small: {ckpt}')
        if not os.path.isfile(cfg):
            problems.append(f'config missing: {cfg}')
        return problems

    def available(self):
        return not self.preflight()

    def _import_protpardelle(self, repo_dir=None):
        """Import protpardelle-1c from ``repo_dir/src`` (no top-level ``core`` clash)."""
        repo_dir = repo_dir or self.repo_dir
        params = self._params_root()
        os.environ.setdefault('PROJECT_ROOT_DIR', os.path.abspath(repo_dir))
        os.environ.setdefault('PROTPARDELLE_MODEL_PARAMS', os.path.abspath(params))

        src = os.path.join(os.path.abspath(repo_dir), 'src')
        if src not in sys.path:
            sys.path.insert(0, src)

        for key in [k for k in list(sys.modules)
                    if k == 'protpardelle' or k.startswith('protpardelle.')]:
            mod = sys.modules.get(key)
            f = getattr(mod, '__file__', None) if mod is not None else None
            if f is None or src not in os.path.abspath(f):
                del sys.modules[key]

        try:
            from protpardelle.common import residue_constants as rc
            from protpardelle.core.models import load_model
        except Exception as e:
            raise ProtpardelleUnavailable(
                f'cannot import protpardelle-1c from {src}: {e}') from e
        got = os.path.abspath(getattr(rc, '__file__', ''))
        if src not in got:
            raise ProtpardelleUnavailable(
                f'protpardelle.common.residue_constants resolved to {got}, '
                f'expected under {src}')
        return load_model, rc

    def _check_constants(self, residue_constants):
        if list(residue_constants.atom_types) != ATOM37:
            raise ProtpardelleUnavailable(
                'protpardelle atom37 ordering differs from the copy in '
                'protpardelle_bridge.ATOM37; coordinates would be scrambled')
        if list(residue_constants.restypes) != RESTYPES:
            raise ProtpardelleUnavailable('protpardelle restype ordering differs')

    def _load(self):
        if self._denoiser is not None or self._model is not None:
            return
        problems = self.preflight()
        if problems:
            raise ProtpardelleUnavailable('; '.join(problems))
        import torch
        load_model, residue_constants = self._import_protpardelle(self.repo_dir)
        self._check_constants(residue_constants)
        cfg_path = self.config or self.default_config()
        ckpt = self.checkpoint or self.default_checkpoint()
        dev = self.device
        if dev.startswith('cuda') and not torch.cuda.is_available():
            dev = 'cpu'
        model = load_model(cfg_path, ckpt, device=dev)
        if self.minimpnn_checkpoint:
            try:
                model.load_minimpnn(self.minimpnn_checkpoint)
            except Exception as e:
                raise ProtpardelleUnavailable(
                    f'failed to load minimpnn from {self.minimpnn_checkpoint}: {e}'
                ) from e
        self._model = model
        self._torch = torch
        self.sigma_data = float(model.sigma_data)
        if self.verbose:
            d = self._defaults()
            name = self.model_name or d['model_name']
            print(f'[protpardelle] 1c/{name} on {dev}, '
                  f'sigma_data={self.sigma_data:.2f}')

    def _rewind_steps(self, model):
        """Map ``noise_angstrom`` → partial-diffusion rewind on the native schedule.

        1c: ``pd_step = schedule_steps - rewind``, start noise =
        ``noise_schedule(rewind / schedule_steps)``. The INFO line
        ``T=pd_step/N`` is *not* that schedule argument.

        ``n_steps`` floors the rewind (deeper pass); it does not cap below the
        noise target.
        """
        import torch
        ns = model.sampling_noise_schedule_default
        target = float(self.noise_angstrom)
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            val = float(ns(torch.tensor(mid)))
            if val > target:
                hi = mid
            else:
                lo = mid
        t = 0.5 * (lo + hi)
        N = max(2, int(self.schedule_steps))
        rewind = int(round(t * N))
        rewind = max(rewind, int(self.n_steps))
        rewind = max(1, min(rewind, N - 1))
        start_noise = float(ns(torch.tensor(rewind / float(N))))
        return rewind, N, start_noise

    @staticmethod
    def _pad_atom37(out, n_res):
        """Ensure atom dim is 37 (backbone models may return fewer atoms)."""
        if out.shape[-2] == 37:
            return out
        full = np.zeros((n_res, 37, 3), float)
        for j, idx in enumerate(BB_IDXS[:out.shape[-2]]):
            full[:, idx] = out[:, j]
        return full

    def propose(self, coords37, mask37, aatype, motif_slots=None, residue_index=None,
                seed=None, n_samples=1):
        """Plain partial-diffusion rewind (no motif conditioning).

        ``motif_slots`` is accepted for API compatibility with test denoisers;
        the live model path ignores it (theozyme is restored after sampling).

        ``n_samples`` > 1 expands the 1c batch dim (``seq_mask`` shape ``B L``).
        Partial diffusion replicates the same PDB path across the batch and draws
        independent noise per sample in one ``model.sample`` call — one tqdm bar
        (or none if ``disable_tqdm``), not B sequential calls.
        Returns ``(L, 37, 3)`` when ``n_samples==1``, else ``(B, L, 37, 3)``.
        """
        n_samples = max(1, int(n_samples))
        if self._denoiser is not None:
            outs = [
                np.asarray(
                    self._denoiser(coords37, mask37, aatype, motif_slots or []),
                    float)
                for _ in range(n_samples)
            ]
            return outs[0] if n_samples == 1 else np.stack(outs, axis=0)
        self._load()
        torch = self._torch
        model = self._model

        n = coords37.shape[0]
        if seed is not None:
            torch.manual_seed(int(seed))

        if residue_index is None:
            residue_index = np.arange(n)
        residue_index = np.asarray(residue_index)
        seq_mask = torch.ones(n_samples, n, dtype=torch.float32, device=model.device)
        ri = torch.as_tensor(
            residue_index, dtype=torch.long, device=model.device
        )[None].expand(n_samples, -1).contiguous()
        chain_index = torch.zeros_like(ri)
        gt_aatype = torch.as_tensor(
            np.asarray(aatype), dtype=torch.long, device=model.device
        )[None].expand(n_samples, -1).contiguous()

        # 1c PD CA-centers the loaded PDB; restore that shift after sampling.
        ca_mean = coords37[:, ATOM37_INDEX['CA']].mean(0)

        rewind, schedule_steps, start_noise = self._rewind_steps(model)
        sidechain = (self.task != 'backbone')

        full_path = None
        try:
            full_tmp = tempfile.NamedTemporaryFile(
                suffix='_structure.pdb', delete=False)
            full_tmp.close()
            full_path = full_tmp.name
            _write_atom37_pdb(
                full_path, coords37, mask37, aatype, residue_index=residue_index)

            if self.verbose:
                pd_step = schedule_steps - rewind
                print(f'[protpardelle] partial diffusion rewind={rewind}/'
                      f'{schedule_steps}  start_noise≈{start_noise:.2f} A  '
                      f'n_samples={n_samples}  '
                      f'(1c logs step {pd_step}, T={pd_step/schedule_steps:.2f}; '
                      f'real t={rewind/schedule_steps:.3f}; no motif cond)',
                      flush=True)

            # 1c hardcodes tqdm + INFO "Partial diffusion, going back to step…".
            # Monkeypatch tqdm, TQDM_DISABLE, and bump that logger to WARNING.
            _ctx = _silence_tqdm() if self.disable_tqdm else nullcontext()
            with _ctx:
                aux = model.sample(
                    seq_mask=seq_mask,
                    residue_index=ri,
                    chain_index=chain_index,
                    gt_aatype=gt_aatype if sidechain else None,
                    num_steps=schedule_steps,
                    step_scale=self.step_scale,
                    s_churn=self.s_churn,
                    noise_schedule=None,
                    sidechain_mode=sidechain,
                    skip_mpnn_proportion=1.0,
                    anneal_seq_resampling_rate=None,
                    use_fullmpnn=False,
                    use_fullmpnn_for_final=False,
                    jump_steps=False,
                    uniform_steps=bool(sidechain),
                    motif_file_path=None,
                    conditional_cfg=dict(_PD_COND_CFG),
                    partial_diffusion={
                        'enabled': True,
                        'pdb_file_path': full_path,
                        'num_steps': rewind,
                        'repack': False,
                        'seq': None,
                    },
                    tqdm_pbar=(lambda x: x),
                )
        finally:
            if full_path and os.path.isfile(full_path):
                try:
                    os.unlink(full_path)
                except OSError:
                    pass

        batch = aux['x'].detach().cpu().numpy().astype(float)
        outs = []
        for b in range(n_samples):
            out = self._pad_atom37(batch[b], n) + ca_mean
            outs.append(out)
        return outs[0] if n_samples == 1 else np.stack(outs, axis=0)

    def relax_structure(self, st, motif_resis, chain=None, seed=None, n_samples=1):
        """Propose relaxed backbone(s) for `st`, returned in the input lab frame.

        ``n_samples==1`` (default): ``(xyz, info)`` as before.
        ``n_samples>1``: ``(list[xyz], list[info])`` from one batched PD call.
        """
        n_samples = max(1, int(n_samples))
        chain = chain or str(st.chain[0])
        coords, mask, aatype, resis = structure_to_atom37(st, chain)
        pos = {int(r): k for k, r in enumerate(resis)}
        motif_slots = [pos[int(r)] for r in motif_resis if int(r) in pos]

        proposed = self.propose(coords, mask, aatype, motif_slots,
                                residue_index=resis, seed=seed,
                                n_samples=n_samples)
        batch = proposed if n_samples > 1 else proposed[None]

        slots = alignment_indices(resis, motif_resis, window=self.align_window)
        xyzs, infos = [], []
        for b in range(n_samples):
            out, fit_rms = superpose_backbone(batch[b], coords, mask, slots)
            moved = np.linalg.norm(out[:, BB_IDXS] - coords[:, BB_IDXS], axis=-1)
            m = mask[:, BB_IDXS] > 0
            info = dict(
                superpose_rmsd=round(fit_rms, 3),
                n_align_residues=int(len(slots)),
                bb_rmsd=round(float(np.sqrt((moved[m] ** 2).mean())), 3),
                bb_max_disp=round(float(moved[m].max()), 3),
                noise_angstrom=self.noise_angstrom, n_steps=self.n_steps,
                schedule_steps=self.schedule_steps,
                n_motif=len(motif_slots),
                motif_conditioned=False,
                n_samples=n_samples,
                sample_index=b)
            xyzs.append(atom37_backbone_into(st, st.xyz, out, resis, chain))
            infos.append(info)
        if n_samples == 1:
            return xyzs[0], infos[0]
        return xyzs, infos


def identity_denoiser(coords37, mask37, aatype, motif_slots):
    """Return the input untouched."""
    return coords37.copy()


def jitter_denoiser(sigma=0.4, pin_motif=True, rotate=True, seed=0):
    """Stand-in denoiser for tests without a GPU."""
    rng = np.random.default_rng(seed)

    def f(coords37, mask37, aatype, motif_slots):
        out = coords37 + rng.normal(0, sigma, coords37.shape) * (mask37 > 0)[..., None]
        if pin_motif and motif_slots is not None and len(motif_slots):
            sl = np.asarray(motif_slots, int)
            out[sl] = coords37[sl]
        if rotate:
            a = rng.normal(size=3)
            a /= np.linalg.norm(a)
            th = rng.uniform(0.05, 0.3)
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
            out = np.einsum('ij,naj->nai', R, out) + rng.normal(0, 3, 3)
        return out
    return f
