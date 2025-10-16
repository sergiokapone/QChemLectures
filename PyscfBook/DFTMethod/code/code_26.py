from pyscf import gto, scf, dft
import matplotlib.pyplot as plt


def compare_orbital_energies(symbol, spin, basis="cc-pvtz"):
    """
    Порівняння орбітальних енергій HF vs DFT
    """

    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

    methods = {"HF": None, "LDA": "lda", "PBE": "pbe", "B3LYP": "b3lyp", "PBE0": "pbe0"}

    orbital_energies = {}

    for method, xc in methods.items():
        if method == "HF":
            mf = scf.UHF(mol) if spin > 0 else scf.RHF(mol)
        else:
            mf = dft.UKS(mol) if spin > 0 else dft.RKS(mol)
            mf.xc = xc

        mf.verbose = 0
        mf.kernel()

        if spin == 0:
            orbital_energies[method] = mf.mo_energy * 27.211386
        else:
            orbital_energies[method] = mf.mo_energy[0] * 27.211386

    # Графік
    fig, ax = plt.subplots(figsize=(12, 8))

    n_orb = min(10, len(orbital_energies["HF"]))
    x = np.arange(n_orb)
    width = 0.16

    colors = ["red", "blue", "green", "orange", "purple"]

    for i, (method, color) in enumerate(zip(methods.keys(), colors)):
        offset = (i - 2) * width
        energies = orbital_energies[method][:n_orb]
        ax.bar(x + offset, energies, width, label=method, color=color, alpha=0.8)

    # Лінія HOMO
    if spin == 0:
        n_occ = mol.nelectron // 2
    else:
        n_occ = mol.nelec[0]

    ax.axvline(
        x=n_occ - 0.5, color="black", linestyle="--", linewidth=2, label="HOMO/LUMO"
    )

    ax.set_xlabel("Орбіталь", fontsize=12)
    ax.set_ylabel("Енергія (eV)", fontsize=12)
    ax.set_title(f"Орбітальні енергії {symbol}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"MO{i + 1}" for i in x])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{symbol}_orbital_energies_comparison.pdf")
    plt.show()

    # Таблиця
    print(f"\nОрбітальні енергії {symbol} (eV)")
    print("=" * 80)
    print(f"{'MO':5s} {'HF':12s} {'LDA':12s} {'PBE':12s} {'B3LYP':12s} {'PBE0':12s}")
    print("-" * 80)

    for i in range(n_orb):
        row_str = f"{i + 1:5d}"
        for method in methods.keys():
            e = orbital_energies[method][i]
            row_str += f" {e:12.4f}"

        if i == n_occ - 1:
            row_str += "  ← HOMO"
        elif i == n_occ:
            row_str += "  ← LUMO"

        print(row_str)

    print("=" * 80)


# Приклади
compare_orbital_energies("C", spin=2)
compare_orbital_energies("Ne", spin=0)
compare_orbital_energies("O", spin=2)
