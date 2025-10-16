import matplotlib.pyplot as plt
import numpy as np

from pyscf import gto, scf, dft


def systematic_ie_calculation(basis="aug-cc-pvtz"):
    """
    Розрахунок IE для атомів другого періоду
    """

    # Дані: (символ, спін нейтрального, спін катіона, IE експ.)
    atoms_data = [
        ("Li", 1, 0, 5.39),
        ("Be", 0, 1, 9.32),
        ("B", 1, 0, 8.30),
        ("C", 2, 1, 11.26),
        ("N", 3, 2, 14.53),
        ("O", 2, 3, 13.62),
        ("F", 1, 2, 17.42),
        ("Ne", 0, 1, 21.56),
    ]

    methods = {"HF": None, "PBE": "pbe", "B3LYP": "b3lyp", "PBE0": "pbe0"}

    results = {method: {"calc": [], "exp": [], "errors": []} for method in methods}
    symbols = []

    print(f"\nСистематичний розрахунок IE (базис: {basis})")
    print("=" * 90)
    print(
        f"{'Атом':6s} {'IE(exp)':10s} {'HF':10s} {'PBE':10s} {'B3LYP':10s} {'PBE0':10s}"
    )
    print("-" * 90)

    for symbol, spin_n, spin_c, ie_exp in atoms_data:
        symbols.append(symbol)
        row_data = [ie_exp]

        # Нейтральний атом
        mol_n = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin_n, verbose=0)

        # Катіон
        mol_c = gto.M(
            atom=f"{symbol} 0 0 0", basis=basis, charge=1, spin=spin_c, verbose=0
        )

        for method, xc in methods.items():
            # Нейтральний
            if method == "HF":
                mf_n = scf.UHF(mol_n) if spin_n > 0 else scf.RHF(mol_n)
                mf_c = scf.UHF(mol_c) if spin_c > 0 else scf.RHF(mol_c)
            else:
                mf_n = dft.UKS(mol_n) if spin_n > 0 else dft.RKS(mol_n)
                mf_c = dft.UKS(mol_c) if spin_c > 0 else dft.RKS(mol_c)
                mf_n.xc = xc
                mf_c.xc = xc

            mf_n.verbose = 0
            mf_c.verbose = 0
            mf_n.conv_tol = 1e-10
            mf_c.conv_tol = 1e-10

            e_n = mf_n.kernel()
            e_c = mf_c.kernel()

            ie_calc = (e_c - e_n) * 27.211386

            results[method]["calc"].append(ie_calc)
            results[method]["exp"].append(ie_exp)
            results[method]["errors"].append(ie_calc - ie_exp)

            row_data.append(ie_calc)

        # Виведення рядка
        print(
            f"{symbol:6s} {ie_exp:10.4f} {row_data[1]:10.4f} "
            f"{row_data[2]:10.4f} {row_data[3]:10.4f} "
            f"{row_data[4]:10.4f}"
        )

    print("=" * 90)

    # Статистика
    print("\nСтатистика похибок (eV):")
    print(f"{'Метод':10s} {'MAE':10s} {'MSE':10s} {'Max|Err|':10s}")
    print("-" * 50)

    for method in methods:
        errors = np.array(results[method]["errors"])
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors)
        max_err = np.max(np.abs(errors))

        print(f"{method:10s} {mae:10.4f} {mse:10.4f} {max_err:10.4f}")

    # Графіки
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # IE значення
    x = np.arange(len(symbols))
    width = 0.18

    colors = ["black", "blue", "orange", "green", "red"]
    labels = ["Експ."] + list(methods.keys())

    for i, (label, color) in enumerate(zip(labels, colors)):
        if label == "Експ.":
            values = [atoms_data[j][3] for j in range(len(symbols))]
        else:
            values = results[label]["calc"]

        offset = (i - 2) * width
        ax1.bar(x + offset, values, width, label=label, color=color, alpha=0.8)

    ax1.set_xlabel("Атом", fontsize=12)
    ax1.set_ylabel("IE (eV)", fontsize=12)
    ax1.set_title("Енергії іонізації", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(symbols)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Похибки
    for method, color in zip(methods.keys(), colors[1:]):
        errors = results[method]["errors"]
        ax2.plot(
            symbols, errors, "o-", label=method, color=color, linewidth=2, markersize=8
        )

    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("Атом", fontsize=12)
    ax2.set_ylabel("Похибка (eV)", fontsize=12)
    ax2.set_title("Похибки відносно експерименту", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ionization_energies_systematic.pdf")
    plt.show()

    return results


results_ie = systematic_ie_calculation()
