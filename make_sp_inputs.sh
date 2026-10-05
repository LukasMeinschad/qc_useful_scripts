#!/bin/bash
# make_sp_inputs.sh - generate Molpro single-point inputs + slurm job files
#
# Usage:
#   make_sp_inputs.sh -x struct.xyz -m "mp2-f12 ccsd(t)-f12b" -b "vdz-f12 vtz-f12" [options]
#
# Options:
#   -x  xyz file (required)
#   -m  Molpro method keywords, space separated, quoted (required)
#   -b  Molpro basis keywords, space separated (required)
#   -o  output directory            (default: ./sp_<xyzname>)
#   -w  Molpro memory in MW         (default: 1000; 1 MW = 8 MB)
#   -n  MPI processes               (default: 1)
#   -p  slurm partition             (default: CPU_rune)
#   -N  slurm node                  (default: rune05)
#   -e  request the node exclusively (cleaner timings, longer queue)
#   -s  submit jobs with sbatch after generating
#   -h  help

set -euo pipefail

CONTAINER=/media/storage_5/containers/molpro_2025.4.1_mpp_expire-2027-06-07.simg

xyz="" methods="" bases="" outdir=""
mem_mw=1000
nproc=1
partition=CPU_rune
node=rune05
exclusive=false
submit=false

usage() { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while getopts "x:m:b:o:w:n:p:N:esh" opt; do
    case $opt in
        x) xyz=$OPTARG ;;
        m) methods=$OPTARG ;;
        b) bases=$OPTARG ;;
        o) outdir=$OPTARG ;;
        w) mem_mw=$OPTARG ;;
        n) nproc=$OPTARG ;;
        p) partition=$OPTARG ;;
        N) node=$OPTARG ;;
        e) exclusive=true ;;
        s) submit=true ;;
        h) usage 0 ;;
        *) usage 1 ;;
    esac
done

[[ -z $xyz || -z $methods || -z $bases ]] && usage 1
[[ -f $xyz ]] || { echo "xyz file '$xyz' not found" >&2; exit 1; }

name=$(basename "$xyz" .xyz)
outdir=${outdir:-sp_$name}
xyz_abs=$(realpath "$xyz")

# slurm memory: Molpro memory is per process, in MW (8 MB),
# plus ~25% headroom for the program itself
slurm_mem_gb=$(( (mem_mw * 8 * nproc * 125 / 100 + 999) / 1000 ))

exclusive_line=""
$exclusive && exclusive_line="#SBATCH --exclusive"

# folder-safe method name: ccsd(t)-f12a -> ccsdt-f12a
short_method() {
    echo "$1" | tr 'A-Z' 'a-z' | sed -E 's/[()]//g; s/[^a-z0-9_-]/_/g'
}

# short basis name: vdz-f12 -> vdz, cc-pvtz-f12 -> vtz
short_basis() {
    echo "$1" | tr 'A-Z' 'a-z' | sed -E 's/-f12$//; s/^cc-p//; s/[^a-z0-9_-]/_/g'
}

mkdir -p "$outdir"

for method in $methods; do
    for basis in $bases; do
        dir="$outdir/$(short_method "$method")_$(short_basis "$basis")"
        mkdir -p "$dir"

        # ---------- Molpro input ----------
        cat > "$dir/$name.inp" <<EOF
memory,$mem_mw,m
GTHRESH,OPTGRAD=1.D-7,TWOINT=1.D-14,PREFAC=1.D-16,ENERGY=1.D-8,
ORIENT,MASS

geometry={
$(cat "$xyz_abs")
}
mass,iso
basis=$basis

hf
{$method}
EOF

        # ---------- slurm job ----------
        cat > "$dir/$name.sjob" <<EOF
#!/bin/bash
#SBATCH --partition=$partition
#SBATCH --nodelist=$node
$exclusive_line
#SBATCH -J ${name}_$(basename "$dir")
#SBATCH -N 1
#SBATCH --ntasks-per-node=$nproc
#SBATCH --cpus-per-task=1
#SBATCH --mem=${slurm_mem_gb}G
source /usr/local/_tci_software_environment_.sh
module purge
module load rune/singularity/4.3.2
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK

mkdir -p /scratch/\$USER/tmpdir_\$SLURM_JOBID
singularity exec --cleanenv --bind /scratch,/media $CONTAINER molpro -n \$SLURM_NTASKS_PER_NODE -d /scratch/\$USER/tmpdir_\$SLURM_JOBID -I \$PWD -W \$PWD -a $name.inp
rm -rf /scratch/\$USER/tmpdir_\$SLURM_JOBID
EOF

        echo "created $dir"

        if $submit; then
            (cd "$dir" && sbatch "$name.sjob")
        fi
    done
done
