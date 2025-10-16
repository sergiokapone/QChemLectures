import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf


def test_koopmans_theorem(atoms_list, basis="cc-pvtz"):
    """
    Перевірка теореми Купманса для серії атомів.
    Порівнює енергії іонізації з ΔSCF та за наближенням -ε(HOMO).
    """

    print(f"\nПеревірка теореми Купманса (базис: {basis})")
    print("=" * 90)
    print(f"{'Атом':6s} {'IE(ΔSCF)':>12s} {'-ε(HOMO)':>12s} {'Різниця':>12s} {'% похибки':>12s}")
    print("-" * 90)

    results = {"atoms": [], "ie_scf": [], "ie_koop": [], "diff": []}

    for symbol, spin_n, spin_c in atoms_list:
        # --- Нейтральний атом ---
        mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin_n, verbose=0)
        mf = scf.UHF(mol) if spin_n != 0 else scf.RHF(mol)
        mf.conv_tol = 1e-10
        e_neutral = mf.kernel()

        # --- HOMO-енергія ---
        if spin_n == 0:
            n_occ = mol.nelectron // 2
            eps_homo = mf.mo_energy[n_occ - 1]
        else:
            n_alpha, n_beta = mol.nelec
            eps_homo_a = mf.mo_energy[0][n_alpha - 1]
            eps_homo_b = mf.mo_energy[1][n_beta - 1]
            eps_homo = max(eps_homo_a, eps_homo_b)

        ie_koopmans = -eps_homo * 27.211386  # eV

        # --- Катіон ---
        mol_c = gto.M(atom=f"{symbol} 0 0 0", basis=basis, charge=1, spin=spin_c, verbose=0)
        mf_c = scf.UHF(mol_c) if spin_c != 0 else scf.RHF(mol_c)
        mf_c.conv_tol = 1e-10
        e_cation = mf_c.kernel()

        # --- Різниця повних енергій (ΔSCF) ---
        ie_scf = (e_cation - e_neutral) * 27.211386  # eV

        # --- Порівняння ---
        diff = ie_koopmans - ie_scf
        percent = abs(diff / ie_scf) * 100

        results["atoms"].append(symbol)
        results["ie_scf"].append(ie_scf)
        results["ie_koop"].append(ie_koopmans)
        results["diff"].append(diff)

        print(f"{symbol:6s} {ie_scf:12.4f} {ie_koopmans:12.4f} {diff:12.4f} {percent:12.2f}")

    print("=" * 90)
    mean_diff = np.mean(np.abs(results["diff"]))
    print(f"\nСередня абсолютна різниця: {mean_diff:.4f} eV")

    # --- Візуалізація ---
    x = np.arange(len(results["atoms"]))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width/2, results["ie_scf"], width, label="IE (ΔSCF)", color="C0", alpha=0.8)
    ax.bar(x + width/2, results["ie_koop"], width, label="IE (-εHOMO)", color="C3", alpha=0.7)
    ax.set_xlabel("Атом", fontsize=12)
    ax.set_ylabel("Енергія іонізації (eV)", fontsize=12)
    ax.set_title("Перевірка теореми Купманса", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(results["atoms"])
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

    return results


# --- Тестові атоми ---
atoms_test = [
    ("He", 0, 1),
    ("Li", 1, 0),
    ("Be", 0, 1),
    ("B", 1, 0),
    ("C", 2, 1),
    ("N", 3, 2),
    ("O", 2, 3),
    ("F", 1, 2),
    ("Ne", 0, 1),
]

results_koop = test_koopmans_theorem(atoms_test)

