"""  
Extract Molpro Single Point Energies, wall times and plot them
"""
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
HARTREE_TO_KJMOL = 2625.49962  # Hartree to kJ/mol

# Energy RegEx
#DCSD-F12B/cc-pVDZ-F12 energy=   -553.331824280196
ENERGY_RE = re.compile(r"^\s*(\S+)/(\S+)\s+energy=\s*(-?\d+\.\d+)", re.M)


# CPU Times RegEx
# PROGRAMS   *        TOTAL DCSD-F12B    HF-SCF       INT
# CPU TIMES  *      3053.45   2923.40     83.71     46.16
PROG_CPU_RE = re.compile(
    r"^\s*PROGRAMS\s+\*\s+(.+)\n\s*CPU TIMES\s+\*\s+(.+)$", re.M
)


def method_cpu_time(text, method):
    """  
    CPU times of the program step matching the method
    """
    blocks = PROG_CPU_RE.findall(text)
    if not blocks:
        return float("nan")
    names_line, times_line = blocks[-1] # Final summary block
    names = names_line.split()
    times = [float(t) for t in times_line.split()]
    # Molpro always list the most recent step
    # so the first match is the one producing the final energy
    for name, t in zip(names, times):
        if name.upper() == method.upper():
            return t
    # Fallback if molpro truncated the name in the header
    for name, t in zip(names, times):
        if method.upper().startswith(name.upper()):
            return t
    print(f"Warning: CPU time for {method} not found in {names_line}", file=sys.stderr)
    return float("nan")

def parse_output(path,root):
    """ 
    Parse the Molpro output file and 
    extract the energy and CPU time for each method
    """
    text = path.read_text(errors="replace")
    if "Molpro calculation terminated" not in text:
        print(f"Warning: {path} did not terminate normally", file=sys.stderr)
        return None
    energies = ENERGY_RE.findall(text)
    if not energies:
        print(f"Warning: No energies found in {path}", file=sys.stderr)
        return None
    method, basis, energy = energies[-1]  # Final energy
    # label is the subfolder path relative to the root
    rel = path.parent.relative_to(root)
    label = str(rel) if str(rel) != "." else path.stem

    return {
        "label": label,
        "method": method,
        "basis": basis,
        "energy": float(energy),
        "cpu_s": method_cpu_time(text, method),
    }

def main(root):
    root = Path(root).resolve()
    results = []
    for f in sorted(root.rglob("*.out")):
        if f.name.startswith("slurm"):
            continue
        r = parse_output(f, root)
        if r is not None:
            results.append(r)

    if not results:
        sys.exit("No results found")

    # table to stdout
    print(f"{'calc':<20}{'method':<14}{'basis':<16}{'E / Eh':>20}{'CPU / s':>12}")
    for r in results:
        print(f"{r['label']:<20}{r['method']:<14}{r['basis']:<16}"
              f"{r['energy']:>20.10f}{r['cpu_s']:>12.2f}")

    # CSV
    with open(root / "sp_energies.csv", "w") as fh:
        fh.write("calc,method,basis,energy_Eh,cpu_time_s\n")
        for r in results:
            fh.write(f"{r['label']},{r['method']},{r['basis']},"
                     f"{r['energy']:.12f},{r['cpu_s']:.2f}\n")

     # --- plotting: all calculations in one series ---
    results.sort(key=lambda r: (r["method"], r["basis"]))
    labels = [f"{r['method']}/{r['basis']}" for r in results]
    x = list(range(len(results)))
    e_min = min(r["energy"] for r in results)
    rel_e = [(r["energy"] - e_min) * HARTREE_TO_KJMOL for r in results]
    cpu_s = [r["cpu_s"] for r in results]


    fig, (ax_e, ax_t) = plt.subplots(
        1, 2, figsize=(max(9, 1.4 * len(results)), 4.5)
    )

    ax_e.plot(x, rel_e, "--", color="tab:blue", alpha=0.6, zorder=1)
    ax_e.scatter(x, rel_e, color="tab:blue", s=40, zorder=2)
    ax_e.set_ylabel(r"$E - E_\mathrm{min}$ / kJ mol$^{-1}$")
    ax_e.set_title(f"Relative energy (E_min = {e_min:.6f} Eh)")

    ax_t.plot(x, cpu_s, "--", color="tab:orange", alpha=0.6, zorder=1)
    ax_t.scatter(x, cpu_s, color="tab:orange", s=40, zorder=2)
    ax_t.set_ylabel("CPU time / s")
    ax_t.set_yscale("log")
    ax_t.set_title("CPU time")

    for ax in (ax_e, ax_t):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(root / "sp_energies.png", dpi=200)
    plt.show()



if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")

 