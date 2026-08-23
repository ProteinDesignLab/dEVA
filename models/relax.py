"""dEVA plugin: relax backbone between iterations without moving the theozyme."""
import gc
import os
import sys
import json
import time
import logging
import traceback
from typing import Dict

import numpy as np

from common.utils import ensure_dir, create_file
from core.interfaces import BaseModel
from core.registry import register_model
from evolve.individual import Individual

from theozyme.pipeline import TheozymeRelaxPipeline, summarise
from theozyme.protpardelle_bridge import ProtpardelleRelaxer

logger = logging.getLogger("evolution")
logger.setLevel(logging.DEBUG)


@register_model("relax")
class RelaxModel(BaseModel):

    def __init__(self):
        self.pipeline = None
        self.enabled = True

    def setup(self, config: Dict, device: str = 'cpu') -> None:
        self.config = config
        models = config.models
        cfg = models.get('relax')
        if cfg is None:
            cfg = models.get('protpardelle_relax')
        if cfg is None:
            raise KeyError("config.models.relax is required (legacy key: protpardelle_relax)")
        self.cfg = cfg
        self.verbose = bool(cfg.get('verbose', True))
        self.seed = config.general.seed
        self.chain = str(cfg.get('chain', 'A'))

        out = config.general.outputs
        self.out_dir = os.path.join(out, cfg.get('out_subdir', 'relaxed'))
        ensure_dir(self.out_dir)
        self.log_path = os.path.join(self.out_dir, 'relax_log.jsonl')

        reference = cfg.get('reference_pdb', None) or config.input.pdb
        theozyme_resis = [int(r) for r in cfg.theozyme_residues]
        ligand_resi = cfg.get('ligand_resi', None)
        ligand_resi = None if ligand_resi is None else int(ligand_resi)

        backend = str(cfg.get('backend', 'protpardelle'))
        relaxer = None
        if backend == 'protpardelle':
            dev = str(cfg.get('device', '')) or (
                'cuda' if str(device).startswith('cuda') else 'cpu')
            relaxer = ProtpardelleRelaxer(
                repo_dir=cfg.get('repo_dir', '../protpardelle-1c'),
                checkpoint=cfg.get('checkpoint', None),
                config=cfg.get('model_config', None),
                task=str(cfg.get('task', 'backbone')),
                device=dev,
                noise_angstrom=float(cfg.get('noise_angstrom', 1.5)),
                n_steps=int(cfg.get('n_steps', 100)),
                schedule_steps=int(cfg.get('schedule_steps', 500)),
                step_scale=float(cfg.get('step_scale', 1.0)),
                s_churn=float(cfg.get('s_churn', 0.0)),
                align_window=int(cfg.get('align_window', 8)),
                minimpnn_checkpoint=cfg.get('minimpnn_checkpoint', None),
                model_name=cfg.get('model_name', None),
                model_epoch=cfg.get('model_epoch', None),
                model_params_dir=cfg.get('model_params_dir', None),
                verbose=self.verbose,
                disable_tqdm=bool(cfg.get('disable_tqdm', True)))
            problems = relaxer.preflight()
            if problems:
                msg = ('[relax] protpardelle is not usable: '
                       + '; '.join(problems))
                if bool(cfg.get('require_protpardelle', False)):
                    raise RuntimeError(msg)
                print(msg + '\n[relax] falling back to '
                            "backend='geometric' for this run", file=sys.stderr)
                backend = 'geometric'
                relaxer = None

        acc_kw = {}
        for k in ('enm_cutoff', 'k_rigid', 'k_mobile', 'k_pos_rigid', 'k_pos_mobile',
                  'rep_dmin', 'k_rep', 'closure_tol', 'k_closure', 'maxiter',
                  'relieve_sidechains', 'idealise_backbone', 'k_bond',
                  'gradient', 'grad_r0', 'grad_r1', 'grad_floor'):
            if k in cfg:
                acc_kw[k] = cfg[k]

        self.pipeline = TheozymeRelaxPipeline(
            reference_pdb=reference,
            theozyme_resis=theozyme_resis,
            ligand_resi=ligand_resi,
            chain=self.chain,
            backend=backend,
            relaxer=relaxer,
            mode=str(cfg.get('mode', 'fixed')),
            clearance=float(cfg.get('clearance', 3.40)),
            shell=float(cfg.get('shell', 10.0)),
            mobile_resis=cfg.get('mobile_residues', None),
            accept_tol=float(cfg.get('accept_tol', 0.05)),
            max_backbone_disp=cfg.get('max_backbone_disp', None),
            bond_tol=float(cfg.get('bond_tol', 0.02)),
            n_protpardelle_attempts=int(cfg.get('n_protpardelle_attempts', 1)),
            select_by_sc_clashes=bool(cfg.get('select_by_sc_clashes', True)),
            accommodate_kw=acc_kw,
            verbose=self.verbose)

        self.emit = bool(cfg.get('emit_objective', True))
        self.target = float(cfg.get('clearance', 3.40))
        self.w_strain = float(cfg.get('w_strain', 0.5))
        self.free_rmsd = float(cfg.get('free_rmsd', 1.0))
        self.fail_value = float(cfg.get('fail_value', -10.0))
        # After rewriting individual.name, Sampler re-scores LigandMPNN on the
        # relaxed PDB (same sequence) so fitness['pmpnn'] matches the backbone
        # EF/pocket/desolv see. Absolute pmpnn typically drops; Pareto becomes
        # "seq fit to final BB". Set false to keep the pre-relax designer score.
        self.rescore_pmpnn_after = bool(cfg.get('rescore_pmpnn_after', True))

        logger.info(f'[relax] backend={backend} '
                    f'mode={self.pipeline.mode} '
                    f'theozyme={self.pipeline.theozyme_resis} '
                    f'ligand={self.pipeline.rigid.lig_key} '
                    f'rescore_pmpnn_after={self.rescore_pmpnn_after}')
        if self.verbose:
            print(self.pipeline.rigid.summary(), flush=True)
        self._check_seq_model_agrees(config, theozyme_resis)

    def _check_seq_model_agrees(self, config, theozyme_resis):
        """Ensure theozyme residues are in ligandmpnn.fixed_residues."""
        which = str(config.get('seq_model', 'ligandmpnn'))
        sec = config.models.get(which)
        if sec is None:
            return
        fixed = str(sec.get('fixed_residues', '') or '')
        var = str(sec.get('var_residues', '') or '')
        if not fixed and not var:
            print(f'[relax] WARNING: {which}.fixed_residues is empty, so '
                  f'the designer may mutate the theozyme residues '
                  f'{theozyme_resis}. Set fixed_residues to pin them.', file=sys.stderr)
            return
        if fixed:
            pinned = {tok.lstrip(''.join(c for c in tok if c.isalpha()))
                      for tok in fixed.split()}
            pinned = {int(p) for p in pinned if p.isdigit()}
            missing = [r for r in theozyme_resis if r not in pinned]
            if missing:
                raise ValueError(
                    f'theozyme residues {missing} are not in {which}.fixed_residues '
                    f'({fixed!r}). The designer would be free to mutate them out from '
                    f'under the frozen geometry. Add them, using the chain prefix, '
                    f'e.g. "{self.chain}{missing[0]}".')

    def _objective(self, report):
        acc = report.get('accommodate') or {}
        ca = acc.get('clearance_after')
        if ca is None:
            return 0.0
        value = min(float(ca['min_dist']), self.target)
        strain = float(acc.get('rmsd_mobile', 0.0))
        value -= self.w_strain * max(0.0, strain - self.free_rmsd)
        if not report.get('accepted', True):
            value -= 0.25
        return value

    def score(self, individual: Individual):
        pdb_in = individual.get_name()
        gen, index = individual.get_gen(), individual.get_index()
        out = create_file(self.out_dir, os.path.splitext(os.path.basename(pdb_in))[0],
                          gen, index, self.seed).replace('.pdb', '_relaxed.pdb')
        t0 = time.time()
        try:
            _, report = self.pipeline.run(pdb_in, out, seed=self.seed + 1000 * gen + index)
            report['seconds'] = round(time.time() - t0, 2)
            report['generation'] = gen
            report['index'] = index

            if not report.get('ok', False):
                raise RuntimeError('theozyme verification failed: '
                                   + json.dumps(report['accommodate']['theozyme'],
                                                default=str))
            individual.update_name(name=out)
            value = self._objective(report)
            if self.verbose:
                print(summarise(report) + f"\n  {report['seconds']}s -> {out}",
                      flush=True)
            self._log(report)
        except Exception as e:
            # Print the exception message on its own line so a preceding
            # "final sequence: ..." log line is not mistaken for the cause.
            print(f'[relax] relaxation failed for {pdb_in}; '
                  f'keeping the unrelaxed structure', file=sys.stderr)
            print(f'[relax] {type(e).__name__}: {e}', file=sys.stderr)
            traceback.print_exc()
            value = self.fail_value
            self._log(dict(input=pdb_in, generation=gen, index=index,
                           error=f'{type(e).__name__}: {e}',
                           traceback=traceback.format_exc(limit=5)))

        if self.emit:
            individual.add_fitness({'relax': float(value)})
        gc.collect()

    def _log(self, report):
        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(report, default=_jsonable) + '\n')
        except Exception:
            pass


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
