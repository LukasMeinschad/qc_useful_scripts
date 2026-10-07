"""
Compare two conformer ensembles generated with different global optimization
methods using three criteria for every pair of structures (A_i, B_j):

1. RMSD after optimal atom correspondence (permutation invariant, Hungarian + Kabsch)
2. Rotational constants from the mass-weighted inertia tensor (max. relative deviation)
3. Energy differences (relative to each ensemble's minimum, or absolute)

Convention: the energy is the first number on the XYZ comment line, in Hartree.

Example:
    python compare_ensembles.py crest/crest_conformers.xyz goat/h2o_n2/h2o_n2.finalensemble.xyz \
        --rmsd-thr 0.125 --rot-thr 0.01 --csv pairs.csv
"""
import argparse
import csv
import itertools
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import linear_sum_assignment


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
MASSES = {  # standard atomic weights, amu
    "H": 1.00794, "He": 4.002602, "Li": 6.941, "Be": 9.012182, "B": 10.811,
    "C": 12.0107, "N": 14.0067, "O": 15.9994, "F": 18.9984032, "Ne": 20.1797,
    "Na": 22.98976928, "Mg": 24.3050, "Al": 26.9815386, "Si": 28.0855,
    "P": 30.973762, "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.0983,
    "Ca": 40.078, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961, "Mn": 54.938045,
    "Fe": 55.845, "Co": 58.933195, "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.64, "As": 74.92160, "Se": 78.96, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.4678, "Sr": 87.62, "Ag": 107.8682, "Cd": 112.411,
    "Sn": 118.710, "Sb": 121.760, "Te": 127.60, "I": 126.90447, "Xe": 131.293,
    "Cs": 132.9054519, "Ba": 137.327, "Pt": 195.084, "Au": 196.966569,
    "Hg": 200.59, "Pb": 207.2,
}
HARTREE_TO_KCAL = 627.509474
ROT_MHZ = 505379.07  # h / (8 pi^2) in MHz * amu * Angstrom^2
FLOAT_RE = r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?"  # first float on the comment line


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def read_multixyz(path, energy_regex=FLOAT_RE):
    """Read a multi-xyz file -> list of dicts with symbols, coords and energy (Hartree)."""
    pat = re.compile(energy_regex)
    with open(path) as f:
        lines = f.read().splitlines()
    out, i = [], 0
    while i < len(lines):
        if not lines[i].strip():  # tolerate blank lines between frames
            i += 1
            continue
        n = int(lines[i].split()[0])
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        m = pat.search(comment)
        energy = float(m.group()) if m else None
        sym, xyz = [], []
        for line in lines[i + 2:i + 2 + n]:
            p = line.split()
            sym.append(re.sub(r"[^A-Za-z]", "", p[0]).capitalize())  # "C1" -> "C"
            xyz.append([float(v) for v in p[1:4]])
        out.append({"symbols": np.array(sym), "coords": np.array(xyz), "energy": energy})
        i += 2 + n
    if not out:
        sys.exit(f"No structures found in {path}")
    return out


# ----------------------------------------------------------------------------
# Rotational constants
# ----------------------------------------------------------------------------
def rotational_constants(symbols, coords):
    """Rotational constants A >= B >= C in MHz; near-zero moments are clamped (linear molecules)."""
    try:
        m = np.array([MASSES[s] for s in symbols])
    except KeyError as e:
        sys.exit(f"Unknown element {e.args[0]} in symbols: {symbols}")
    X = coords - (m[:, None] * coords).sum(0) / m.sum()  # shift to center of mass
    I = np.zeros((3, 3))
    for mi, r in zip(m, X):
        I += mi * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    moments = np.clip(np.linalg.eigvalsh(I), 1e-6, None)  # ascending -> constants descending
    return ROT_MHZ / moments


def rot_deviation(ra, rb):
    """Maximum relative deviation of the three rotational constants."""
    return float(np.max(np.abs(ra - rb) / np.maximum(ra, rb)))


# ----------------------------------------------------------------------------
# Permutation invariant RMSD
# ----------------------------------------------------------------------------
def principal_frame(X):
    """Center X and rotate it into its (geometric) principal axis frame."""
    X = X - X.mean(axis=0)
    _, v = np.linalg.eigh(X.T @ X)
    if np.linalg.det(v) < 0:  # keep a proper rotation
        v[:, 2] *= -1
    return X @ v


def kabsch(P, Q, allow_reflection=False):
    """Return P optimally rotated onto Q (both already centered).

    SVD of the covariance P^T Q = U S V^T gives the rotation U V^T;
    if det < 0 and reflections are not allowed, flip the smallest axis.
    """
    U, _, Vt = np.linalg.svd(P.T @ Q)
    if not allow_reflection and np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    return P @ (U @ Vt)


def hungarian_reorder(sym, P, Q):
    """Permutation of P's rows (only within the same element) that best matches Q."""
    perm = np.arange(len(sym))
    for el in np.unique(sym):
        idx = np.where(sym == el)[0]
        cost = ((P[idx][:, None, :] - Q[idx][None, :, :]) ** 2).sum(-1)
        row, col = linear_sum_assignment(cost)
        perm[idx[col]] = idx[row]
    return perm


def random_rotations(n, seed=0):
    """n uniformly distributed random rotation matrices (from random unit quaternions)."""
    q = np.random.default_rng(seed).normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1)[:, None]
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], 1)


def best_rmsd(symA, A, symB, B, reflections=False, nstarts=50, max_iter=30, early=1e-3):
    """Minimum RMSD over rotations (optionally reflections) and atom permutations.

    Starting from the principal axis frame plus random orientations, alternate
    Hungarian reassignment and Kabsch alignment until the RMSD stops improving.
    """
    if len(symA) != len(symB) or sorted(symA) != sorted(symB):
        return np.inf  # different composition -> not comparable
    oa, ob = np.argsort(symA, kind="stable"), np.argsort(symB, kind="stable")
    sym = symA[oa]
    P0, Q = principal_frame(A[oa]), principal_frame(B[ob])

    # starting orientations: axis flips of the principal frame + random rotations
    starts = [np.diag(s) for s in itertools.product([1, -1], repeat=3)]
    starts += list(random_rotations(nstarts))
    if reflections:
        starts += [-R for R in starts]
    else:
        starts = [R for R in starts if np.linalg.det(R) > 0]

    best = np.inf
    for R in starts:
        P, prev = P0 @ R.T, np.inf
        for _ in range(max_iter):
            P = P[hungarian_reorder(sym, P, Q)]
            P = kabsch(P, Q, allow_reflection=reflections)
            r = np.sqrt(((P - Q) ** 2).sum(1).mean())
            if prev - r < 1e-7:  # converged
                break
            prev = r
        best = min(best, r)
        if best < early:  # identical structures, no need to search further
            break
    return float(best)


# ----------------------------------------------------------------------------
# Ensemble comparison
# ----------------------------------------------------------------------------
def _rmsd_row(args):
    """Worker: RMSD of one structure of A against all structures of B."""
    a, ens_b, kw = args
    return [best_rmsd(a["symbols"], a["coords"], b["symbols"], b["coords"], **kw) for b in ens_b]


def rmsd_matrix(ens_a, ens_b, nprocs=1, **kw):
    """N_A x N_B RMSD matrix, rows computed in parallel."""
    jobs = [(a, ens_b, kw) for a in ens_a]
    if nprocs > 1:
        with ProcessPoolExecutor(nprocs) as ex:
            rows = list(ex.map(_rmsd_row, jobs))
    else:
        rows = [_rmsd_row(j) for j in jobs]
    return np.array(rows)


def rot_matrix(ens_a, ens_b):
    """N_A x N_B matrix of max. relative rotational constant deviations."""
    ra = [rotational_constants(s["symbols"], s["coords"]) for s in ens_a]
    rb = [rotational_constants(s["symbols"], s["coords"]) for s in ens_b]
    return np.array([[rot_deviation(x, y) for y in rb] for x in ra])


def energies_kcal(ens, absolute=False):
    """Energies in kcal/mol, relative to the ensemble minimum unless absolute=True."""
    e = np.array([s["energy"] if s["energy"] is not None else np.nan for s in ens], float)
    if not absolute and np.isfinite(e).any():
        e = e - np.nanmin(e)
    return e * HARTREE_TO_KCAL


def match_mask(rmsd, rot, de, args):
    """Boolean N_A x N_B matrix: True where a pair counts as the same structure."""
    m = (rmsd <= args.rmsd_thr) & (rot <= args.rot_thr)
    if args.e_thr is not None:  # energy criterion is optional
        m &= np.abs(de) <= args.e_thr
    return m


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def print_best_matches(label_a, label_b, ea, rmsd, rot, de, match):
    """For every structure of A: closest structure of B (by RMSD) and whether it matches."""
    print(f"\nBest match in {label_b} for every structure of {label_a}")
    print(f"{'i':>4} {'E_' + label_a:>10} {'j':>4} {'RMSD/A':>8} {'dRot':>8} {'dE':>8}  match")
    for i in range(rmsd.shape[0]):
        j = int(np.argmin(rmsd[i]))
        print(f"{i + 1:4d} {ea[i]:10.3f} {j + 1:4d} {rmsd[i, j]:8.4f} {rot[i, j]:8.4f} "
              f"{de[i, j]:8.3f}  {'yes' if match[i].any() else 'NO'}")


def write_csv(path, ea, eb, rmsd, rot, de, match):
    """All pairs in long format (1-based indices, energies in kcal/mol)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i_A", "j_B", "rmsd_A", "rot_dev", "E_A", "E_B", "dE", "match"])
        for i, j in itertools.product(range(rmsd.shape[0]), range(rmsd.shape[1])):
            w.writerow([i + 1, j + 1, f"{rmsd[i, j]:.5f}", f"{rot[i, j]:.6f}",
                        f"{ea[i]:.4f}", f"{eb[j]:.4f}", f"{de[i, j]:.4f}", int(match[i, j])])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ensemble_a", help="multi-xyz file of method A")
    p.add_argument("ensemble_b", help="multi-xyz file of method B")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--rmsd-thr", type=float, default=0.125, help="RMSD threshold in Angstrom")
    p.add_argument("--rot-thr", type=float, default=0.01, help="max. relative rot. constant deviation")
    p.add_argument("--e-thr", type=float, default=None,
                   help="optional energy threshold in kcal/mol (off by default, as methods may differ)")
    p.add_argument("--absolute-energy", action="store_true",
                   help="compare absolute energies instead of energies relative to each ensemble minimum")
    p.add_argument("--energy-regex", default=FLOAT_RE, help="regex for the energy on the comment line")
    p.add_argument("--reflections", action="store_true", help="treat mirror images as identical")
    p.add_argument("--nstarts", type=int, default=50, help="random starting orientations for the RMSD")
    p.add_argument("--nprocs", type=int, default=1, help="parallel processes for the RMSD matrix")
    p.add_argument("--csv", help="write all pairwise results to this CSV file")
    args = p.parse_args()

    # read both ensembles
    ens_a = read_multixyz(args.ensemble_a, args.energy_regex)
    ens_b = read_multixyz(args.ensemble_b, args.energy_regex)
    la, lb = args.label_a, args.label_b
    print(f"{la}: {len(ens_a)} structures from {args.ensemble_a}")
    print(f"{lb}: {len(ens_b)} structures from {args.ensemble_b}")

    # the three criteria as N_A x N_B matrices
    rmsd = rmsd_matrix(ens_a, ens_b, nprocs=args.nprocs,
                       reflections=args.reflections, nstarts=args.nstarts)
    rot = rot_matrix(ens_a, ens_b)
    ea = energies_kcal(ens_a, args.absolute_energy)
    eb = energies_kcal(ens_b, args.absolute_energy)
    de = ea[:, None] - eb[None, :]  # kcal/mol
    match = match_mask(rmsd, rot, de, args)

    # per-structure tables in both directions
    print_best_matches(la, lb, ea, rmsd, rot, de, match)
    print_best_matches(lb, la, eb, rmsd.T, rot.T, -de.T, match.T)

    # summary
    found_a, found_b = match.any(1), match.any(0)
    lowest = lambda e: int(np.nanargmin(e)) if np.isfinite(e).any() else 0  # first frame if no energies
    ia, ib = lowest(ea), lowest(eb)
    print("\nSummary")
    crit = f"RMSD <= {args.rmsd_thr} A, dRot <= {args.rot_thr}"
    crit += f", |dE| <= {args.e_thr} kcal/mol" if args.e_thr is not None else ""
    print(f"  match criteria             : {crit}")
    print(f"  {la} structures found in {lb}  : {found_a.sum()} / {len(ens_a)}")
    print(f"  {lb} structures found in {la}  : {found_b.sum()} / {len(ens_b)}")
    print(f"  unique to {la}               : {[i + 1 for i in np.where(~found_a)[0]]}")
    print(f"  unique to {lb}               : {[j + 1 for j in np.where(~found_b)[0]]}")
    print(f"  lowest-energy {la} #{ia + 1} vs {lb} #{ib + 1}: RMSD = {rmsd[ia, ib]:.4f} A, "
          f"dRot = {rot[ia, ib]:.4f}, same = {bool(match[ia, ib])}")

    if args.csv:
        write_csv(args.csv, ea, eb, rmsd, rot, de, match)
        print(f"\nPairwise results written to {args.csv}")


if __name__ == "__main__":
    main()
