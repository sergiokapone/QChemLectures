from pyscf import gto, scf
import matplotlib.pyplot as plt


def plot_orbital_diagram(symbol, spin, basis="cc-pvdz"):
    """Побудова діаграми орбітальних енергій"""

    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)

    mf.verbose = 0
    mf.kernel()

    fig, ax = plt.subplots(figsize=(8, 10))

    if spin == 0:
        # RHF: одна колонка орбіталей
        energies = mf.mo_energy * 27.211386  # eV
        n_occ = mol.nelec[0]

        for i, e in enumerate(energies[: n_occ + 5]):
            color = "blue" if i < n_occ else "red"
            label = mol.ao_labels()[i] if i < len(mol.ao_labels()) else ""
            ax.hlines(e, 0.2, 0.8, colors=color, linewidth=2)
            ax.text(0.85, e, f"{label}", fontsize=10, va="center")

            # Стрілки для електронів
            if i < n_occ:
                ax.arrow(
                    0.35,
                    e + 0.3,
                    0,
                    -0.25,
                    head_width=0.05,
                    head_length=0.1,
                    fc="black",
                    ec="black",
                )
                ax.arrow(
                    0.65,
                    e - 0.3,
                    0,
                    0.25,
                    head_width=0.05,
                    head_length=0.1,
                    fc="black",
                    ec="black",
                )
    else:
        # UHF: дві колонки (альфа та бета)
        e_alpha = mf.mo_energy[0] * 27.211386
        e_beta = mf.mo_energy[1] * 27.211386
        n_alpha, n_beta = mol.nelec

        # Альфа орбіталі
        for i, e in enumerate(e_alpha[: n_alpha + 3]):
            color = "blue" if i < n_alpha else "red"
            ax.hlines(e, 0.1, 0.4, colors=color, linewidth=2)
            if i < n_alpha:
                ax.arrow(
                    0.25,
                    e + 0.3,
                    0,
                    -0.25,
                    head_width=0.04,
                    head_length=0.1,
                    fc="black",
                    ec="black",
                )

        # Бета орбіталі
        for i, e in enumerate(e_beta[: n_beta + 3]):
            color = "blue" if i < n_beta else "red"
            ax.hlines(e, 0.6, 0.9, colors=color, linewidth=2)
            if i < n_beta:
                ax.arrow(
                    0.75,
                    e - 0.3,
                    0,
                    0.25,
                    head_width=0.04,
                    head_length=0.1,
                    fc="black",
                    ec="black",
                )

        ax.text(0.25, min(e_alpha[:5]) - 2, "α", fontsize=14, ha="center")
        ax.text(0.75, min(e_beta[:5]) - 2, "β", fontsize=14, ha="center")

    ax.set_xlim(0, 1)
    ax.set_ylabel("Енергія (eV)", fontsize=12)
    ax.set_title(f"Діаграма орбіталей {symbol}", fontsize=14)
    ax.set_xticks([])
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{symbol}_orbital_diagram.pdf")
    plt.show()


# Приклади
plot_orbital_diagram("C", spin=2)  # Вуглець
plot_orbital_diagram("Ne", spin=0)  # Неон
