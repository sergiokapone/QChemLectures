from pyscf import gto, scf, dft
import numpy as np
import matplotlib.pyplot as plt


def compare_functionals(symbol, spin, basis="cc-pvtz"):
    """
    Систематичне порівняння функціоналів
    """

    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

    # Список методів для порівняння
    methods = [
        ("HF", "scf"),
        ("LDA", "lda"),
        ("PBE", "pbe"),
        ("BLYP", "blyp"),
        ("TPSS", "tpss"),
        ("B3LYP", "b3lyp"),
        ("PBE0", "pbe0"),
        ("CAM-B3LYP", "camb3lyp"),
    ]

    energies = []
    labels = []

    print(f"\nПорівняння методів для {symbol} (базис: {basis})")
    print("=" * 70)
    print(f"{'Метод':15s} {'Енергія, Ha':15s} {'Відносно HF, mHa':20s}")
    print("-" * 70)

    for name, method in methods:
        try:
            if method == "scf":
                if spin == 0:
                    mf = scf.RHF(mol)
                else:
                    mf = scf.UHF(mol)
            else:
                if spin == 0:
                    mf = dft.RKS(mol)
                else:
                    mf = dft.UKS(mol)
                mf.xc = method

            mf.verbose = 0
            mf.conv_tol = 1e-10
            energy = mf.kernel()

            if mf.converged:
                energies.append(energy)
                labels.append(name)

                if name == "HF":
                    e_ref = energy
                    rel = 0.0
                else:
                    rel = (energy - e_ref) * 1000

                print(f"{name:15s} {energy:15.8f} {rel:20.4f}")
            else:
                print(f"{name:15s} не конвергувало")
        except Exception as e:
            print(f"{name:15s} помилка: {str(e)[:30]}")

    print("=" * 70)

    # Графік
    if len(energies) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(labels))
        colors = [
            "red"
            if l == "HF"
            else "blue"
            if l in ["LDA", "PBE", "BLYP", "TPSS"]
            else "green"
            for l in labels
        ]

        bars = ax.bar(x, energies, color=colors, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Енергія (Ha)", fontsize=12)
        ax.set_title(f"Порівняння методів для атома {symbol}", fontsize=14)
        ax.grid(True, alpha=0.3, axis="y")

        # Легенда
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="red", alpha=0.7, label="HF"),
            Patch(facecolor="blue", alpha=0.7, label="Pure DFT"),
            Patch(facecolor="green", alpha=0.7, label="Hybrid DFT"),
        ]
        ax.legend(handles=legend_elements, loc="best")

        plt.tight_layout()
        plt.savefig(f"{symbol}_functionals_comparison.pdf")
        plt.show()

    return energies, labels


# Тестування на різних атомах
energies_c, labels_c = compare_functionals("C", spin=2)
energies_ne, labels_ne = compare_functionals("Ne", spin=0)
energies_fe, labels_fe = compare_functionals("Fe", spin=4, basis="def2-svp")
