from pyscf import gto, scf, dft
import numpy as np
import matplotlib.pyplot as plt


def compare_ionization_energies(atoms_data, basis="aug-cc-pvtz"):
    """
    Порівняння розрахункових та експериментальних IE
    """

    methods = ["HF", "LDA", "PBE", "B3LYP", "PBE0"]

    print(f"\nЕнергії іонізації (eV), базис: {basis}")
    print("=" * 90)
    print(
        f"{'Атом':6s} {'Експ.':10s} {'HF':10s} {'LDA':10s} "
        f"{'PBE':10s} {'B3LYP':10s} {'PBE0':10s}"
    )
    print("-" * 90)

    results = {method: [] for method in methods}
    experimental = []
    atom_symbols = []

    for symbol, spin_n, spin_c, ie_exp in atoms_data:
        atom_symbols.append(symbol)
        experimental.append(ie_exp)

        # Нейтральний атом
        mol_n = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin_n, verbose=0)

        # Катіон
        mol_c = gto.M(
            atom=f"{symbol} 0 0 0", basis=basis, charge=1, spin=spin_c, verbose=0
        )

        ie_values = [ie_exp]

        for method in methods:
            # Нейтральний
            if method == "HF":
                mf_n = scf.UHF(mol_n) if spin_n > 0 else scf.RHF(mol_n)
                mf_c = scf.UHF(mol_c) if spin_c > 0 else scf.RHF(mol_c)
            else:
                mf_n = dft.UKS(mol_n) if spin_n > 0 else dft.RKS(mol_n)
                mf_c = dft.UKS(mol_c) if spin_c > 0 else dft.RKS(mol_c)

                xc_dict = {"LDA": "lda", "PBE": "pbe", "B3LYP": "b3lyp", "PBE0": "pbe0"}
                mf_n.xc = xc_dict[method]
                mf_c.xc = xc_dict[method]

            mf_n.verbose = 0
            mf_c.verbose = 0

            e_n = mf_n.kernel()
            e_c = mf_c.kernel()

            ie = (e_c - e_n) * 27.211386  # eV
            results[method].append(ie)
            ie_values.append(ie)

        # Виведення
        row_str = f"{symbol:6s}"
        for ie in ie_values:
            row_str += f" {ie:10.4f}"
        print(row_str)

    print("=" * 90)

    # Статистика похибок
    print("\nСередні абсолютні похибки (MAE, eV):")
    print("-" * 50)

    for method in methods:
        errors = [
            abs(results[method][i] - experimental[i]) for i in range(len(experimental))
        ]
        mae = np.mean(errors)
        print(f"{method:10s}: {mae:8.4f} eV")

    # Графік
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Порівняння IE
    x = np.arange(len(atom_symbols))
    width = 0.14

    colors = ["black", "red", "blue", "green", "orange", "purple"]

    for i, (method, color) in enumerate(zip(["Експ."] + methods, colors)):
        offset = (i - 2.5) * width
        if method == "Експ.":
            values = experimental
        else:
            values = results[method]

        ax1.bar(x + offset, values, width, label=method, color=color, alpha=0.8)

    ax1.set_xlabel("Атом", fontsize=12)
    ax1.set_ylabel("Енергія іонізації (eV)", fontsize=12)
    ax1.set_title("Порівняння енергій іонізації", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(atom_symbols)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Похибки
    for method, color in zip(methods, colors[1:]):
        errors = [
            results[method][i] - experimental[i] for i in range(len(experimental))
        ]
        ax2.plot(
            atom_symbols,
            errors,
            "o-",
            label=method,
            color=color,
            linewidth=2,
            markersize=8,
        )

    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("Атом", fontsize=12)
    ax2.set_ylabel("Похибка (eV)", fontsize=12)
    ax2.set_title("Похибки відносно експерименту", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ionization_energies_comparison.pdf")
    plt.show()


# Дані: (символ, спін нейтрального, спін катіона, IE експ.)
atoms_ie = [
    ("Li", 1, 0, 5.39),
    ("Be", 0, 1, 9.32),
    ("B", 1, 0, 8.30),
    ("C", 2, 1, 11.26),
    ("N", 3, 2, 14.53),
    ("O", 2, 3, 13.62),
    ("F", 1, 2, 17.42),
    ("Ne", 0, 1, 21.56),
]

compare_ionization_energies(atoms_ie)
