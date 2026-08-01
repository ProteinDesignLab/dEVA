import os
import time
import logging
import argparse

from core.engine import EvolutionEngine
from core.interfaces import build_models

logger = logging.getLogger("evolution")
logger.setLevel(logging.DEBUG)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["PYTORCH_KERNEL_CACHE_PATH"] = "/scratch/users/gelnesr/.cache/pytorch_kernels"

_p = logging.getLogger("propka")
_p.setLevel(logging.CRITICAL)
_p.propagate = False
_p.handlers = [logging.NullHandler()]

def main(args):
    engine = EvolutionEngine(config=args.config, pdb=args.pdb, out_folder=args.out_folder)
    config = engine.get_config()

    # import all models here
    import models.mpnn_model
    
    pdb = engine.get_pdb_name()
    logger.info(f'Designing protein from {pdb}')
    
    logger.info(f'Loading models specified.')
    models = build_models(specs=args.models)
    engine.update_models(models)
    engine.setup()
    engine.run()

if __name__ == "__main__": 
    p = argparse.ArgumentParser(description="EVOLVE - a modulear multi-objective protein design platform)")
    p.add_argument('--config', default="configs/evolution.yml", type=str) 
    p.add_argument("--models", nargs="+", default=["seq_model", "metal3d_model"], help="List of registered model specs. Sequence model first, the remaining models/objectives are evaluated in the order provided")
    p.add_argument("--pdb", type=str, default=None)
    p.add_argument("--out_folder", type=str, default=None)
    args = p.parse_args()

    if 'seq_model' not in args.models:
        raise ValueError('Must define a seq_model to perform sequenc design. Aborting...')
    main(args)
