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

    def save_statistics(self, evo_out: tuple):
        save = os.path.join(self.out_folder, f'{self.pdb_name}_s{self.seed}.pkl')
        
        for key, value in evo_out.items():
            f_name = save.replace('.pkl', f'_{key}.pkl')
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
