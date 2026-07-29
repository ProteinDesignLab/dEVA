#!/usr/bin/env python3
"""Write (and optionally submit) a Slurm array over a folder of dEVA configs.

Scratch layout is shared across campaigns:
  /scratch/users/gelnesr/nsga_sbatch/{sbatch,out,err}

Examples:
  # existing kim-design2 campaign (defaults)
  python make_sbatch.py
  python make_sbatch.py --submit

  # ra95 barrel ranks 0-9
  python make_sbatch.py --campaign ra95 --tag barrel \\
      --glob 'ra95_barrel_rank*.yml' \\
      --models seq_model protpardelle_relax electric_field pocket_shape desolvation
  python project_retroaldolase/make_sbatch.py --campaign ra95 --tag barrel \
      --glob 'ra95_barrel_rank*.yml' \
      --models seq_model protpardelle_relax electric_field pocket_shape desolvation \
      --submit
"""
import os
import glob
import argparse
import subprocess

# ---------------------------------------------------------------- defaults
SEED        = 8
CAMPAIGN    = "kim-design2"
OUT_ROOT    = None          # default: /scratch/users/gelnesr/dEVA/{campaign}
SBATCH_DIR  = "/scratch/users/gelnesr/nsga_sbatch/sbatch"
STDOUT_DIR  = "/scratch/users/gelnesr/nsga_sbatch/out"
STDERR_DIR  = "/scratch/users/gelnesr/nsga_sbatch/err"
EVOLVE_ROOT = "/home/users/gelnesr/dEVA"
MODELS      = "seq_model metal3d_model pocket_shape"
TAG         = None          # default: campaign name (shortened)

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--campaign', default=CAMPAIGN,
               help='config folder under configs/ and run folder under scratch '
                    f'(default: {CAMPAIGN})')
p.add_argument('--config-dir', default=None,
               help='override configs path relative to EVOLVE_ROOT '
                    '(default: configs/{campaign})')
p.add_argument('--glob', dest='pattern', default='*.yml',
               help="which ymls to include, e.g. 'ra95_barrel_rank*.yml' "
                    "(default: *.yml)")
p.add_argument('--models', nargs='+', default=MODELS.split(),
               help='passed to run.py --models')
p.add_argument('--tag', default=None,
               help='job-name / log prefix (default: campaign)')
p.add_argument('--seed', type=int, default=SEED)
p.add_argument('--out-root', default=None,
               help='per-task output root '
                    '(default: /scratch/users/gelnesr/dEVA/{campaign})')
p.add_argument('--evolve-root', default=EVOLVE_ROOT,
               help=f'dEVA checkout on the cluster (default: {EVOLVE_ROOT})')
p.add_argument('--sbatch-dir', default=SBATCH_DIR)
p.add_argument('--stdout-dir', default=STDOUT_DIR)
p.add_argument('--stderr-dir', default=STDERR_DIR)
p.add_argument('--time', default='1-00:00:00')
p.add_argument('--partition', default='possu')
p.add_argument('--mem', default='40G')
p.add_argument('--cpus', type=int, default=8)
p.add_argument('--submit', action='store_true',
               help='actually sbatch; without this the script is written and listed only')
p.add_argument('--only', nargs='*', default=None, help='restrict to these file stems')
p.add_argument('--skip', nargs='*', default=None,
               help='stems to leave out')
p.add_argument('--max-concurrent', type=int, default=20,
               help='throttle: max array tasks running at once (0 = unlimited)')
args = p.parse_args()

campaign = args.campaign
config_dir = args.config_dir or f'configs/{campaign}'
out_root = args.out_root or f'/scratch/users/gelnesr/dEVA/{campaign}'
tag = args.tag or campaign.replace('/', '_')
seed = args.seed
evolve_root = args.evolve_root
sbatch_dir, stdout_dir, stderr_dir = args.sbatch_dir, args.stdout_dir, args.stderr_dir
models = ' '.join(args.models)

# kim-design2 historical skips, only when using that campaign's defaults
skip = args.skip
if skip is None:
    skip = (['design_campaign2__G12', 'design_campaign2__H5']
            if campaign == 'kim-design2' and args.pattern == '*.yml'
            else [])

for d in (sbatch_dir, stdout_dir, stderr_dir):
    os.makedirs(d, exist_ok=True)

cfg_glob = os.path.join(evolve_root, config_dir, args.pattern)
stems = sorted(os.path.basename(f)[:-4] for f in glob.glob(cfg_glob))
if not stems:
    # also allow generating the script from a local checkout path
    local_glob = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              config_dir, args.pattern)
    stems = sorted(os.path.basename(f)[:-4] for f in glob.glob(local_glob))
    if stems:
        print(f'note: no ymls under {cfg_glob}; using local {local_glob}')
    else:
        raise SystemExit(f'no ymls matching {cfg_glob} (or local {local_glob})')
if args.only:
    stems = [s for s in stems if s in set(args.only)]
stems = [s for s in stems if s not in set(skip)]
if not stems:
    raise SystemExit('no stems left after --only/--skip')

# ------------------------------------------------------- manifest + script
manifest = f'{sbatch_dir}/{tag}_s{seed}.manifest'
with open(manifest, 'w') as fh:
    fh.write('\n'.join(stems) + '\n')

array_spec = f'0-{len(stems) - 1}'
if args.max_concurrent and args.max_concurrent > 0:
    array_spec += f'%{args.max_concurrent}'

script = f'{sbatch_dir}/{tag}_s{seed}.sbatch'
body = f"""#!/bin/bash
#SBATCH --time={args.time}
#SBATCH -p {args.partition}
#SBATCH --gres=gpu:1
#SBATCH --mem={args.mem}
#SBATCH -c {args.cpus}
#SBATCH --job-name={tag}_s{seed}
#SBATCH --array={array_spec}
#SBATCH --output={stdout_dir}/{tag}_%A_%a.out
#SBATCH --error={stderr_dir}/{tag}_%A_%a.err
#SBATCH --gpu_cmode=shared

set -euo pipefail

STEM=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {manifest})
if [ -z "${{STEM}}" ]; then
    echo "no stem on line $((SLURM_ARRAY_TASK_ID + 1)) of {manifest}" >&2
    exit 1
fi
echo "task ${{SLURM_ARRAY_TASK_ID}} -> ${{STEM}}"

ml chemistry
ml openbabel
ml gcc
cd {evolve_root}

python run.py --config {config_dir}/${{STEM}}.yml \\
    --models {models} \\
    --out_folder {out_root}/${{STEM}} \\
    > {stdout_dir}/{tag}_${{STEM}}_s{seed}.out \\
    2> {stderr_dir}/{tag}_${{STEM}}_s{seed}.err
"""

with open(script, 'w') as fh:
    fh.write(body)

print(f'campaign={campaign}  config_dir={config_dir}  tag={tag}  seed={seed}')
print(f'models={models}')
print(f'out_root={out_root}')
print(f'logs: {stdout_dir}/  {stderr_dir}/')
for i, s in enumerate(stems):
    print(f'{i:>4}  {s}')

if args.submit:
    subprocess.run(['sbatch', script], check=True)
    print(f'\nsubmitted array of {len(stems)} tasks from {script}')
else:
    print(f'\nwrote {script} and {manifest} '
          f'({len(stems)} tasks, re-run with --submit)')
