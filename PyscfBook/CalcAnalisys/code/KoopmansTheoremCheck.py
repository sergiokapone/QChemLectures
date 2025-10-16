import matplotlib.pyplot as plt

from pyscf import gto, scf


def test_koopmans_theorem(atoms_list, basis="cc-pvtz"):
    """
    Перевірка теореми Купманса
    """

    print(f"\nПеревірка теореми Купманса (базис: {basis})")
    print("=" * 80)
    print(
        f"{'Атом':6s} {'IE(ΔSCF)':12s} {'-εHOMO':12s} {'Різниця':12s} {'% похибки':12s}"
    )
    print("-" * 80)

    results = {"atoms": [], "ie_scf": [], "ie_koop": [], "diff": []}

    for symbol, spin_n, spin_c in atoms_list:
        # Нейтральний атом
        mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin_n, verbose=0)

        if spin_n == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)

        mf.verbose = 0
        mf.conv_tol = 1e-11
        e_neutral = mf.kernel()

        # HOMO енергія
        if spin_n == 0:
            n_occ = mol.nelectron // 2
            eps_homo = mf.mo_energy[n_occ - 1]
        else:
            # Для UHF беремо найвищу заповнену
            n_alpha = mol.nelec[0]
            eps_homo = mf.mo_energy[0][n_alpha - 1]

        ie_koopmans = -eps_homo * 27.211386  # eV

        # ΔSCF
        mol_cat = gto.M(
            atom=f"{symbol} 0 0 0", basis=basis, charge=1, spin=spin_c, verbose=0
        )

        if spin_c == 0:
            mf_cat = scf.RHF(mol_cat)
        else:
            mf_cat = scf.UHF(mol_cat)

        mf_cat.verbose = 0
        mf_cat.conv_tol = 1e-11
        e_cation = mf_cat.kernel()

        ie_scf = (e_cation - e_neutral) * 27.211386

        # Порівняння
        diff = ie_koopmans - ie_scf
        percent = abs(diff / ie_scf) * 100

        results["atoms"].append(symbol)
        results["ie_scf"].append(ie_scf)
        results["ie_koop"].append(ie_koopmans)
        results["diff"].append(diff)

        print(
            f"{symbol:6s} {ie_scf:12.4f} {ie_koopmans:12.4f} "
            f"{diff:12.4f} {percent:12.2f}"
        )

    print("=" * 80)

    # Середня похибка
    mean_diff = np.mean(np.abs(results["diff"]))
    print(f"\nСередня абсолютна різниця: {mean_diff:.4f} eV")

    # Графік
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(results["atoms"]))
    width = 0.35

    ax.bar(
        x - width / 2,
        results["ie_scf"],
        width,
        label="IE (ΔSCF)",
        color="blue",
        alpha=0.7,
    )
    ax.bar(
        x + width / 2,
        results["ie_koop"],
        width,
        label="IE (-εHOMO)",
        color="red",
        alpha=0.7,
    )

    ax.set_xlabel("Атом", fontsize=12)
    ax.set_ylabel("Енергія іонізації (eV)", fontsize=12)
    ax.set_title("Теорема Купманса: ΔSCF vs -εHOMO", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(results["atoms"])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("koopmans_theorem_test.pdf")
    plt.show()

    return results


# Атоми для тестування
atoms_test = [
    ("He", 0, 1),
    ("Li", 1, 0),
    ("Be", 0, 1),
    ("C", 2, 1),
    ("N", 3, 2),
    ("O", 2, 3),
    ("F", 1, 2),
    ("Ne", 0, 1),
]

results_koop = test_koopmans_theorem(atoms_test)
