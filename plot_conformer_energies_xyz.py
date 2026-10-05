#!/usr/bin/env python3
"""
Plot conformer energies from multi-structure XYZ files (CREST / GOAT ensembles)
and flag large energy jumps between consecutive conformers (e.g. dissociation).

Usage:
    python plot_conformer_energies.py crest_conformers.xyz h2o_n2.finalensemble.xyz \
        --labels CREST GOAT --threshold 1.0 -o energies.png
"""
import argparse
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

HARTREE_TO_KCAL = 627.509474
HARTREE_TO_KJ = 2625.499639
FLOAT_RE = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?")


def read_energies(path):
    """Return the energies (Hartree) of all structures in a multi-XYZ file.

    The energy is taken as the first floating-point number on the comment line,
    which works for CREST ('  -29.54244416') and ORCA GOAT ensembles.
    """
    energies = []
    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        n_atoms = int(line.split()[0])
        comment = lines[i + 1]
        match = FLOAT_RE.search(comment)
        if match is None:
            sys.exit(f"{path}: no energy found in comment line {i + 2}: {comment!r}")
        energies.append(float(match.group()))
        i += n_atoms + 2
    return np.array(energies)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="multi-structure XYZ files")
    parser.add_argument("--labels", nargs="+", help="legend labels (default: file names)")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="flag consecutive energy jumps larger than this (in --unit), default 1.0")
    parser.add_argument("--unit", choices=["kcal", "kj", "hartree"], default="kcal")
    parser.add_argument("--no-sort", action="store_true",
                        help="keep file order instead of sorting by energy")
    parser.add_argument("--per-file-ref", action="store_true",
                        help="reference each file to its own minimum instead of the global minimum")
    parser.add_argument("-o", "--output", help="save figure to this file instead of showing it")
    args = parser.parse_args()

    labels = args.labels or args.files
    if len(labels) != len(args.files):
        sys.exit("Number of --labels must match number of files")

    factor = {"kcal": HARTREE_TO_KCAL, "kj": HARTREE_TO_KJ, "hartree": 1.0}[args.unit]
    unit_label = {"kcal": "kcal/mol", "kj": "kJ/mol", "hartree": "Eh"}[args.unit]

    data = {}
    for path, label in zip(args.files, labels):
        e = read_energies(path)
        if not args.no_sort:
            e = np.sort(e)
        data[label] = e

    global_min = min(e.min() for e in data.values())

    fig, (ax_e, ax_d) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})

    for label, e in data.items():
        ref = e.min() if args.per_file_ref else global_min
        rel = (e - ref) * factor
        delta = np.diff(rel)
        idx = np.arange(1, len(e) + 1)

        line, = ax_e.plot(idx, rel, "o-", ms=4, label=f"{label} ({len(e)} conf.)")
        color = line.get_color()
        ax_d.plot(idx[1:], delta, "o-", ms=3, color=color)

        jumps = np.where(np.abs(delta) > args.threshold)[0]
        print(f"\n{label}: {len(e)} conformers, E_min = {e.min():.8f} Eh, "
              f"span = {rel.max() - rel.min():.3f} {unit_label}")
        if len(jumps) == 0:
            print(f"  no jumps > {args.threshold} {unit_label}")
        for j in jumps:
            print(f"  jump between conformer {j + 1} -> {j + 2}: "
                  f"{delta[j]:+.3f} {unit_label}  (E_rel {rel[j]:.3f} -> {rel[j + 1]:.3f})")
            ax_e.axvline(j + 1.5, color=color, ls=":", lw=1)
            ax_d.plot(j + 2, delta[j], "x", color="red", ms=9, mew=2)

    ax_d.axhline(args.threshold, color="red", ls="--", lw=1, label=f"threshold {args.threshold} {unit_label}")
    ax_d.axhline(-args.threshold, color="red", ls="--", lw=1)

    ax_e.set_ylabel(f"E$_{{rel}}$ / {unit_label}")
    ax_e.set_title("Conformer energies")
    ax_e.legend()
    ax_e.grid(alpha=0.3)

    ax_d.set_xlabel("Conformer index" + ("" if args.no_sort else " (sorted by energy)"))
    ax_d.set_ylabel(f"ΔE (i − (i−1)) / {unit_label}")
    ax_d.legend()
    ax_d.grid(alpha=0.3)

    fig.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=300)
        print(f"\nSaved figure to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()