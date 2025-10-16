from pyscf import gto, dft
import matplotlib.pyplot as plt


def compare_spin_states(symbol, spin_list, functional="pbe0", basis="def2-tzvp"):
    """
    Порівняння енергій різних спінових станів
    """

    print(f"\nПорівняння спінових станів {symbol}")
    print(f"Функціонал: {functional.upper()}, базис: {basis}")
    print("=" * 70)
    print(f"{'2S':5s} {'Mult':5s} {'Енергія, Ha':15s} {'Відносна, kcal/mol':20s}")
    print("-" * 70)

    energies = []
    spins = []

    for spin in spin_list:
        mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

        if spin == 0:
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)

        mf.xc = functional
        mf.conv_tol = 1e-10
        mf.max_cycle = 150

        try:
            energy = mf.kernel()

            if mf.converged:
                energies.append(energy)
                spins.append(spin)

                mult = spin + 1

                if len(energies) == 1:
                    e_ref = energy
                    rel = 0.0
                else:
                    rel = (energy - e_ref) * 627.509  # kcal/mol

                marker = " ← найнижча" if energy == min(energies) else ""
                print(f"{spin:5d} {mult:5d} {energy:15.8f} {rel:20.4f}{marker}")
        except:
            print(f"{spin:5d} {spin + 1:5d} не конвергувало")

    print("=" * 70)

    # Графік
    if len(energies) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Відносні енергії в kcal/mol
        e_min = min(energies)
        rel_energies = [(e - e_min) * 627.509 for e in energies]

        ax.plot(spins, rel_energies, "o-", markersize=10, linewidth=2)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        ax.set_xlabel("2S", fontsize=12)
        ax.set_ylabel("Відносна енергія (kcal/mol)", fontsize=12)
        ax.set_title(f"Спінові стани {symbol} ({functional.upper()})", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Підписи мультиплетностей
        for s, e in zip(spins, rel_energies):
            ax.text(s, e + 1, f"M={s + 1}", ha="center", fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{symbol}_spin_states_{functional}.pdf")
        plt.show()

    return energies, spins


# Приклад: Карбон (різні спінові стани)
# Основний стан C: ³P (триплет, 2S=2)
# Збуджені: ¹D (синглет, 2S=0), ¹S (синглет, 2S=0)
energies_c, spins_c = compare_spin_states("C", [0, 2, 4], functional="pbe0")

# Залізо (різні спінові стани)
# Fe може бути у низькоспіновому, середньо- та високоспіновому станах
energies_fe, spins_fe = compare_spin_states(
    "Fe", [0, 2, 4, 6], functional="tpss", basis="def2-svp"
)
