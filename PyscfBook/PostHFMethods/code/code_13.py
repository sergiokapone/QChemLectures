from pyscf import gto, scf, mp, cc, fci
import numpy as np
import matplotlib.pyplot as plt


def helium_correlation_study():
    """
    Детальне дослідження кореляції у атомі He
    """

    # Різні базисні набори
    basis_sets = ["cc-pvdz", "cc-pvtz", "cc-pvqz", "cc-pv5z"]

    results = {"HF": [], "MP2": [], "CCSD": [], "CCSD(T)": [], "FCI": []}

    print("Збіжність до базисної межі для He")
    print("=" * 80)
    print(
        f"{'Базис':12s} {'HF':15s} {'MP2':15s} {'CCSD':15s} {'CCSD(T)':15s} {'FCI':15s}"
    )
    print("-" * 80)

    for basis in basis_sets:
        mol = gto.M(atom="He 0 0 0", basis=basis, verbose=0)

        # HF
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.conv_tol = 1e-12
        e_hf = mf.kernel()
        results["HF"].append(e_hf)

        # MP2
        mymp2 = mp.MP2(mf)
        mymp2.verbose = 0
        e_mp2_corr, _ = mymp2.kernel()
        e_mp2 = e_hf + e_mp2_corr
        results["MP2"].append(e_mp2)

        # CCSD
        mycc = cc.CCSD(mf)
        mycc.verbose = 0
        e_ccsd_corr, _, _ = mycc.kernel()
        e_ccsd = e_hf + e_ccsd_corr
        results["CCSD"].append(e_ccsd)

        # CCSD(T)
        e_t = mycc.ccsd_t()
        e_ccsdt = e_ccsd + e_t
        results["CCSD(T)"].append(e_ccsdt)

        # FCI (точний у даному базисі)
        myfci = fci.FCI(mf)
        e_fci = myfci.kernel()[0]
        results["FCI"].append(e_fci)

        print(
            f"{basis:12s} {e_hf:15.10f} {e_mp2:15.10f} "
            f"{e_ccsd:15.10f} {e_ccsdt:15.10f} {e_fci:15.10f}"
        )

    print("=" * 80)

    # Експериментальне значення
    e_exp = -2.90372  # Ha
    print(f"\nЕкспериментальна енергія He: {e_exp:.10f} Ha")

    # Похибки для найбільшого базису
    print(f"\nПохибки (cc-pv5z):")
    for method in results.keys():
        error = (results[method][-1] - e_exp) * 1000  # mHa
        print(f"{method:10s}: {error:8.4f} mHa")

    # Графік збіжності
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Абсолютні енергії
    x = np.arange(len(basis_sets))
    for method, energies in results.items():
        ax1.plot(x, energies, "o-", label=method, linewidth=2, markersize=8)

    ax1.axhline(y=e_exp, color="red", linestyle="--", linewidth=2, label="Експеримент")
    ax1.set_xticks(x)
    ax1.set_xticklabels(basis_sets, rotation=45)
    ax1.set_xlabel("Базисний набір", fontsize=12)
    ax1.set_ylabel("Енергія (Ha)", fontsize=12)
    ax1.set_title("Збіжність енергії He", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Кореляційна енергія
    for method in ["MP2", "CCSD", "CCSD(T)", "FCI"]:
        corr_energies = [
            results[method][i] - results["HF"][i] for i in range(len(basis_sets))
        ]
        ax2.plot(x, corr_energies, "o-", label=method, linewidth=2, markersize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(basis_sets, rotation=45)
    ax2.set_xlabel("Базисний набір", fontsize=12)
    ax2.set_ylabel("Кореляційна енергія (Ha)", fontsize=12)
    ax2.set_title("Кореляційна енергія He", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("he_correlation_convergence.pdf")
    plt.show()


helium_correlation_study()
