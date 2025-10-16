from pyscf import gto, scf, mp, cc, fci
import matplotlib.pyplot as plt


def method_hierarchy_comparison(symbol, spin, basis="cc-pvtz"):
    """
    Порівняння HF → MP2 → CCSD → CCSD(T) → FCI
    """

    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

    # Перевірка розміру (FCI експоненційно зростає)
    if mol.nelectron > 10:
        print(f"Атом {symbol} занадто великий для FCI")
        return

    print(f"\nІєрархія методів для {symbol} (базис: {basis})")
    print("=" * 80)

    methods = {}

    # HF
    print("HF розрахунок...")
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)

    mf.verbose = 0
    mf.conv_tol = 1e-11
    e_hf = mf.kernel()
    methods["HF"] = e_hf

    # MP2
    print("MP2 розрахунок...")
    if spin == 0:
        mymp2 = mp.MP2(mf)
    else:
        mymp2 = mp.UMP2(mf)

    mymp2.verbose = 0
    e_mp2_corr, _ = mymp2.kernel()
    methods["MP2"] = e_hf + e_mp2_corr

    # CCSD
    print("CCSD розрахунок...")
    if spin == 0:
        mycc = cc.CCSD(mf)
    else:
        mycc = cc.UCCSD(mf)

    mycc.verbose = 0
    mycc.conv_tol = 1e-10
    e_ccsd_corr, _, _ = mycc.kernel()
    methods["CCSD"] = e_hf + e_ccsd_corr

    # CCSD(T)
    print("CCSD(T) розрахунок...")
    e_t = mycc.ccsd_t()
    methods["CCSD(T)"] = methods["CCSD"] + e_t

    # FCI (точний у даному базисі)
    print("FCI розрахунок...")
    if spin == 0:
        myfci = fci.FCI(mf)
    else:
        myfci = fci.FCI(mf)

    myfci.verbose = 0
    e_fci = myfci.kernel()[0]
    methods["FCI"] = e_fci

    # Результати
    print("\n" + "=" * 80)
    print(
        f"{'Метод':12s} {'Енергія, Ha':18s} {'Відносно HF, mHa':20s} "
        f"{'% FCI кореляції':18s}"
    )
    print("-" * 80)

    e_corr_fci = e_fci - e_hf

    for method in ["HF", "MP2", "CCSD", "CCSD(T)", "FCI"]:
        e = methods[method]
        rel = (e - e_hf) * 1000

        if method == "HF":
            percent = 0.0
        else:
            percent = abs((e - e_hf) / e_corr_fci) * 100

        print(f"{method:12s} {e:18.10f} {rel:20.6f} {percent:18.2f}")

    print("=" * 80)

    # Графік
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Абсолютні енергії
    method_names = list(methods.keys())
    energies = [methods[m] for m in method_names]
    colors = ["blue", "orange", "green", "red", "purple"]

    x = np.arange(len(method_names))
    bars = ax1.bar(x, energies, color=colors, alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_names)
    ax1.set_ylabel("Енергія (Ha)", fontsize=12)
    ax1.set_title(f"Повні енергії {symbol}", fontsize=14)
    ax1.grid(True, alpha=0.3, axis="y")

    # Значення на стовпчиках
    for bar, e in zip(bars, energies):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{e:.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Кореляційна енергія (відносно HF)
    corr_energies = [(methods[m] - e_hf) * 1000 for m in method_names[1:]]

    ax2.bar(range(len(corr_energies)), corr_energies, color=colors[1:], alpha=0.7)
    ax2.set_xticks(range(len(corr_energies)))
    ax2.set_xticklabels(method_names[1:])
    ax2.set_ylabel("Кореляційна енергія (mHa)", fontsize=12)
    ax2.set_title(f"Кореляційні енергії {symbol}", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{symbol}_method_hierarchy.pdf")
    plt.show()

    return methods


# Приклади (тільки для малих атомів через FCI)
methods_he = method_hierarchy_comparison("He", spin=0, basis="cc-pvdz")
methods_be = method_hierarchy_comparison("Be", spin=0, basis="cc-pvdz")
methods_c = method_hierarchy_comparison("C", spin=2, basis="cc-pvdz")
