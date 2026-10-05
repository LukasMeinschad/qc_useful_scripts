"""
Create one Molpro job per conformer from a CREST/GOAT multi-XYZ ensemble.

The template is a complete Molpro input (memory, basis, method, ...). Only the
geometry={...} block is replaced by each conformer's coordinates.

Usage:
    python generate_crest_goat_reopt_job.py ensemble.xyz template.inp -n 5
    python generate_crest_goat_reopt_job.py ensemble.xyz template.inp -n 5 --ntasks 16 --mem 120G --submit
"""

import argparse
import re  
from pathlib import Path


SJOB = """#!/bin/bash
#SBATCH --partition={partition}
#SBATCH -J {jobname}
#SBATCH -N 1
#SBATCH --ntasks-per-node={ntasks}
#SBATCH --cpus-per-task=1
#SBATCH --mem={mem}
source /usr/local/_tci_software_environment_.sh
module purge
module load rune/singularity/4.3.2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p /scratch/$USER/tmpdir_$SLURM_JOBID
singularity exec --cleanenv --bind /scratch,/media /media/storage_5/containers/molpro_2025.4.1_mpp_expire-2027-06-07.simg molpro -n $SLURM_NTASKS_PER_NODE -d /scratch/$USER/tmpdir_$SLURM_JOBID -I $PWD -W $PWD -a {inp}
rm -rf /scratch/$USER/tmpdir_$SLURM_JOBID
"""


def read_xyz(path):
    """Return list of (comment line, [atom lines]) for every structure."""
    lines = Path(path).read_text().splitlines()
    structures, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i])
        structures.append((lines[i + 1].strip(), lines[i + 2:i + 2 + n]))
        i += n + 2
    return structures


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ensemble", help="multi-XYZ file (crest_conformers.xyz, *.finalensemble.xyz)")
    p.add_argument("template", help="complete Molpro input with a geometry={...} block")
    p.add_argument("-n", "--nconf", type=int, required=True, help="use conformers 1..N")
    p.add_argument("--outdir", default=".", help="where conf1, conf2, ... are created")
    p.add_argument("--ntasks", type=int, default=10)
    p.add_argument("--mem", default="100G")
    p.add_argument("--partition", default="CPU_rune")
    args = p.parse_args()

    template = Path(args.template).read_text()
    geom_re = re.compile(r"geometry\s*=\s*\{.*?\}", re.IGNORECASE | re.DOTALL)
    if not geom_re.search(template):
        raise SystemExit("No geometry={...} block found in template")

    name = Path(args.template).stem
    structures = read_xyz(args.ensemble)
    n = min(args.nconf, len(structures))
    print(f"{len(structures)} conformers in ensemble, writing 1..{n}")

    for c in range(1, n + 1):
        comment, atoms = structures[c - 1]
        geometry = "geometry={\n" + "\n".join(a.strip() for a in atoms) + "\n}"
        folder = Path(args.outdir) / f"conf{c}"
        folder.mkdir(parents=True, exist_ok=True)

        (folder / f"{name}.inp").write_text(geom_re.sub(lambda m: geometry, template, count=1))
        (folder / f"{name}.sjob").write_text(SJOB.format(
            partition=args.partition, jobname=f"{name}_conf{c}", ntasks=args.ntasks,
            mem=args.mem, inp=f"{name}.inp"))
        print(f"  conf{c:<4} {comment}")

    print(f"\nSubmit with:\n  cd {args.outdir} && "
          f"for d in conf{{1..{n}}}; do (cd $d && sbatch {name}.sjob); done")


if __name__ == "__main__":
    main()