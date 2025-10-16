from pyscf import gto, scf, mp
import numpy as np
import matplotlib.pyplot as plt


def hf_vs_mp2_comparison(atoms_list, basis="cc-pvtz"):
    """
    Систематичне порівняння HF та MP2 для списку атомів
    """

    results = {"atoms": [], "e_hf": [], "e_mp2": [], "e_corr": [], "corr_percent": []}

    print(f"\nПорівняння HF та MP2 (базис: {basis})")
    print("=" * 80)
    print(
        f"{'Атом':6s} {'Спін':4s} {'E(HF), Ha':15s} {'E(MP2), Ha':15s} "
        f"{'E_corr, mHa':12s} {'% від HF':10s}"
    )
    print("-" * 80)

    for symbol, spin in atoms_list:
        mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

        # HF
        if spin == 0:
            mf = scf.RHF(mol)
            mp_method = mp.MP2
        else:
            mf = scf.UHF(mol)
            mp_method = mp.UMP2

        mf.verbose = 0
        mf.conv_tol = 1e-10
        e_hf = mf.kernel()

        # MP2
        mymp2 = mp_method(mf)
        mymp2.verbose = 0
        e_mp2_corr, _ = mymp2.kernel()
        e_mp2 = e_hf + e_mp2_corr

        # Зберігаємо результати
        results["atoms"].append(symbol)
        results["e_hf"].append(e_hf)
        results["e_mp2"].append(e_mp2)
        results["e_corr"].append(e_mp2_corr)

        corr_percent = abs(e_mp2_corr / e_hf) * 100
        results["corr_percent"].append(corr_percent)

        print(
            f"{symbol:6s} {spin:4d} {e_hf:15.8f} {e_mp2:15.8f} "
            f"{e_mp2_corr * 1000:12.4f} {corr_percent:10.4f}"
        )

    print("=" * 80)

    # Статистика
    avg_corr = np.mean(results["corr_percent"])
    print(f"\nСередній % кореляції: {avg_corr:.4f}%")

    # Графіки
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(results["atoms"]))

    # Абсолютні енергії
    ax1.plot(
        x, results["e_hf"], "o-", label="HF", linewidth=2, markersize=8, color="blue"
    )
    ax1.plot(
        x, results["e_mp2"], "s-", label="MP2", linewidth=2, markersize=8, color="red"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(results["atoms"])
    ax1.set_xlabel("Атом", fontsize=12)
    ax1.set_ylabel("Енергія (Ha)", fontsize=12)
    ax1.set_title("HF vs MP2 енергії", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Кореляційна енергія
    ax2.bar(x, np.array(results["e_corr"]) * 1000, color="green", alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(results["atoms"])
    ax2.set_xlabel("Атом", fontsize=12)
    ax2.set_ylabel("Кореляційна енергія (mHa)", fontsize=12)
    ax2.set_title("MP2 кореляційна енергія", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("hf_vs_mp2_comparison.pdf")
    plt.show()

    return results


# Атоми другого періоду
atoms_2nd = [
    ("He", 0),
    ("Li", 1),
    ("Be", 0),
    ("B", 1),
    ("C", 2),
    ("N", 3),
    ("O", 2),
    ("F", 1),
    ("Ne", 0),
]

results = hf_vs_mp2_comparison(atoms_2nd)
