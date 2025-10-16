from pyscf import gto, scf, dft
import numpy as np
import matplotlib.pyplot as plt


def comprehensive_comparison(atoms_list, basis="cc-pvtz"):
    """
    Детальне порівняння HF та DFT для списку атомів
    """

    methods = {
        "HF": None,
        "LDA": "lda",
        "PBE": "pbe",
        "BLYP": "blyp",
        "B3LYP": "b3lyp",
        "PBE0": "pbe0",
    }

    results = {method: [] for method in methods}
    atom_symbols = []

    print(f"\nПорівняння методів (базис: {basis})")
    print("=" * 90)
    print(
        f"{'Атом':6s} {'HF':15s} {'LDA':15s} {'PBE':15s} "
        f"{'BLYP':15s} {'B3LYP':15s} {'PBE0':15s}"
    )
    print("-" * 90)

    for symbol, spin in atoms_list:
        mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

        energies_row = []

        # HF
        if spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)
        mf.verbose = 0
        mf.conv_tol = 1e-10
        e_hf = mf.kernel()
        results["HF"].append(e_hf)
        energies_row.append(e_hf)

        # DFT функціонали
        for method in ["LDA", "PBE", "BLYP", "B3LYP", "PBE0"]:
            if spin == 0:
                mf = dft.RKS(mol)
            else:
                mf = dft.UKS(mol)

            mf.xc = methods[method]
            mf.verbose = 0
            mf.conv_tol = 1e-10

            try:
                energy = mf.kernel()
                results[method].append(energy)
                energies_row.append(energy)
            except:
                results[method].append(np.nan)
                energies_row.append(np.nan)

        atom_symbols.append(symbol)

        # Виведення рядка
        row_str = f"{symbol:6s}"
        for e in energies_row:
            if not np.isnan(e):
                row_str += f" {e:15.8f}"
            else:
                row_str += f" {'N/A':15s}"
        print(row_str)

    print("=" * 90)

    # Аналіз різниць
    print("\nРізниці відносно HF (mHa):")
    print("=" * 90)
    print(
        f"{'Атом':6s} {'LDA-HF':12s} {'PBE-HF':12s} "
        f"{'BLYP-HF':12s} {'B3LYP-HF':12s} {'PBE0-HF':12s}"
    )
    print("-" * 90)

    for i, symbol in enumerate(atom_symbols):
        row_str = f"{symbol:6s}"
        e_hf = results["HF"][i]

        for method in ["LDA", "PBE", "BLYP", "B3LYP", "PBE0"]:
            e_dft = results[method][i]
            if not np.isnan(e_dft):
                diff = (e_dft - e_hf) * 1000
                row_str += f" {diff:12.4f}"
            else:
                row_str += f" {'N/A':12s}"
        print(row_str)

    print("=" * 90)

    # Графік
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Абсолютні енергії
    x = np.arange(len(atom_symbols))
    width = 0.14

    for i, method in enumerate(methods.keys()):
        offset = (i - 2.5) * width
        energies = results[method]
        ax1.bar(x + offset, energies, width, label=method)

    ax1.set_xlabel("Атом", fontsize=12)
    ax1.set_ylabel("Енергія (Ha)", fontsize=12)
    ax1.set_title("Абсолютні енергії", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(atom_symbols)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Різниці відносно HF
    for i, method in enumerate(["LDA", "PBE", "BLYP", "B3LYP", "PBE0"]):
        offset = (i - 2) * width
        diffs = [
            (results[method][j] - results["HF"][j]) * 1000
            for j in range(len(atom_symbols))
        ]
        ax2.bar(x + offset, diffs, width, label=method)

    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax2.set_xlabel("Атом", fontsize=12)
    ax2.set_ylabel("ΔE відносно HF (mHa)", fontsize=12)
    ax2.set_title("Різниці енергій", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(atom_symbols)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hf_vs_dft_comparison.pdf")
    plt.show()

    return results


# Атоми другого періоду
atoms_2nd = [
    ("Li", 1),
    ("Be", 0),
    ("B", 1),
    ("C", 2),
    ("N", 3),
    ("O", 2),
    ("F", 1),
    ("Ne", 0),
]

results = comprehensive_comparison(atoms_2nd, basis="cc-pvtz")
