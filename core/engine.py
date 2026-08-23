import os
import time
import torch
import pickle
import random
import logging
import numpy as np
from typing import Dict
from omegaconf import OmegaConf

import sampler
from common.utils import ensure_dir
from evolve.problem import Problem
from evolve.evolution import Evolution
from evolve.recovery import recover_run, needs_recovery

logger = logging.getLogger("evolution")
logger.setLevel(logging.DEBUG)

class EvolutionEngine:
    def __init__(self, config: str='configs/evolution.yml', pdb: str=None, out_folder: str=None):
        with open(config, 'r') as f:
            self.config = OmegaConf.load(f)
        self.seed = self.config.general.seed
        self.evolution = self.config.evolution
        self.set_seed()
        self.device = torch.device('cpu')
        if self.config.general.cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if out_folder is not None:
            print(f"Warning: ignoring output folder in yml. Using {out_folder} instead.")
            self.config.general.outputs = out_folder
        self.out_folder = self.config.general.outputs
        ensure_dir(self.out_folder)
        if pdb is not None:
            print(f"Warning: ignoring pdb in yml. Using {pdb} instead.")
            self.config.input.pdb = pdb
        self.pdb, self.pdb_name = self.check_pdb(self.config.input.pdb)
        self.models = None
        self.seq_model = self.config.seq_model

    def update_models(self, models):
        self.models = models

    def get_device(self):
        ''' Return device '''
        return self.device

    def get_outputs(self):
        ''' Return output folder '''
        return self.out_folder
        
    def get_config(self):
        ''' Return config '''
        return self.config
    
    def get_pdb_name(self):
        ''' Get device '''
        return self.pdb_name

    def get_pdb(self):
        ''' Return path to original pdb '''
        return self.pdb

    def parse_pdb_name(self, pdb):
        return os.path.split(pdb)[-1].split('.')[0]

    def check_pdb(self, pdb):
        if not os.path.isfile(pdb):
            logger.warning("File not found. Downloading pdb to current directory...")
            pdb_name = self.parse_pdb_name(pdb)
            os.system(f"wget -O {pdb} https://files.rcsb.org/download/{pdb_name.upper()}.pdb")
            assert os.path.isfile(pdb) and os.path.getsize(pdb) > 0, 'pdb not found'
        else:
            pdb_name = self.parse_pdb_name(pdb)
            assert os.path.isfile(pdb) and os.path.getsize(pdb) > 0, 'pdb not found'
        return pdb, pdb_name

    def save_statistics(self, evo_out: dict):
        """Pickle best/fronts/statistics, recovering any empty entry from checkpoints.

        evolve() builds these in memory and returns them only at the end, so a
        run that never entered the generation loop (e.g. resumed from a
        checkpoint already at n_generations) returns empty lists. Rather than
        writing those out -- and clobbering good data from an earlier run --
        rebuild them from the pareto_front_gen*.pkl checkpoints on disk.
        """
        save = os.path.join(self.out_folder, f'{self.pdb_name}_s{self.seed}.pkl')

        n_gen = getattr(self.evolution, 'n_generations', None) if hasattr(self, 'evolution') else None
        if needs_recovery(self.out_folder, evo_out, n_generations=n_gen):
            bad = [k for k, v in evo_out.items() if not v] or ['(partial)']
            logger.warning('incomplete evolution outputs %s; attempting recovery from '
                           'checkpoints in %s', bad, self.out_folder)
            evo_out, notes = recover_run(self.out_folder, evo_out, n_generations=n_gen)
            for n in notes:
                logger.warning('  %s', n)

        for key, value in evo_out.items():
            f_name = save.replace('.pkl', f'_{key}.pkl')
            if not value:
                if os.path.exists(f_name) and os.path.getsize(f_name) > 16:
                    logger.warning('refusing to overwrite %s with empty %s',
                                   os.path.basename(f_name), key)
                    continue
                logger.warning('%s is empty and could not be recovered', key)
            with open(f_name,  "wb") as f:
                pickle.dump(value, f)

        logger.info('dEVA outputs saved!')

    def set_seed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup(self):
        for k, m in self.models.items():
            m.setup(config=self.config, device=self.device)
        logger.info('Models loaded!')

    def run(self):
        design_sampler = sampler.Sampler(self.models)
        problem = Problem(sampler=design_sampler, seq_model=self.seq_model)
        logger.info("Starting evolution...")
        time_start = time.time()
        evo = Evolution(
            problem,
            num_generations=self.evolution.n_generations,
            num_individuals=self.evolution.n_individuals,
            num_mutations=self.evolution.n_mutations,
            sampler=design_sampler,
            seed=self.seed,
            checkpoint_dir=self.out_folder,
        )
        evo_out = evo.evolve()
        self.save_statistics(evo_out)
        logger.info(f"Evolution took {time.time() - time_start} seconds")