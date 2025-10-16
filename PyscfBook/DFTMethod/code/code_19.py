from pyscf import gto, dft, lo
import numpy as np


def analyze_d_orbitals(symbol, spin, functional="pbe", basis="def2-tzvp"):
    """
    Аналіз заселеності d-орбіталей
    """

    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

    mf = dft.UKS(mol)
    mf.xc = functional
    mf.verbose = 0
    energy = mf.kernel()

    if not mf.converged:
        print(f"Не конвергувало для {symbol}")
        return

    print(f"\nАналіз d-орбіталей {symbol} ({functional.upper()})")
    print("=" * 70)

    # Заселеності Малікена
    pop, chg = mf.mulliken_pop()

    # Пошук d-орбіталей
    ao_labels = mol.ao_labels(fmt=False)

    d_orbitals_alpha = []
    d_orbitals_beta = []

    for i, label in enumerate(ao_labels):
        atom_id, atom_symbol, shell_type, *rest = label
        if shell_type.startswith("3d"):
            # Альфа заселеність
            dm_alpha = mf.make_rdm1()[0]
            s = mol.intor("int1e_ovlp")
            pop_alpha = (dm_alpha @ s)[i, i]

            # Бета заселеність
            dm_beta = mf.make_rdm1()[1]
            pop_beta = (dm_beta @ s)[i, i]

            d_orbitals_alpha.append((shell_type, pop_alpha))
            d_orbitals_beta.append((shell_type, pop_beta))

    if d_orbitals_alpha:
        print("\nЗаселеності d-орбіталей:")
        print(f"{'Орбіталь':12s} {'α':10s} {'β':10s} {'Сума':10s} {'Спін':10s}")
        print("-" * 70)

        for (orb_a, pop_a), (orb_b, pop_b) in zip(d_orbitals_alpha, d_orbitals_beta):
            total = pop_a + pop_b
            spin_dens = pop_a - pop_b
            print(
                f"{orb_a:12s} {pop_a:10.4f} {pop_b:10.4f} "
                f"{total:10.4f} {spin_dens:10.4f}"
            )

        # Загальна заселеність d-оболонки
        total_d_alpha = sum(p for _, p in d_orbitals_alpha)
        total_d_beta = sum(p for _, p in d_orbitals_beta)

        print("-" * 70)
        print(
            f"{'Разом':12s} {total_d_alpha:10.4f} "
            f"{total_d_beta:10.4f} "
            f"{total_d_alpha + total_d_beta:10.4f} "
            f"{total_d_alpha - total_d_beta:10.4f}"
        )

    # Орбітальні енергії
    print("\nОрбітальні енергії (eV):")
    print(f"{'MO':5s} {'α-енергія':12s} {'β-енергія':12s} {'Заповнення':15s}")
    print("-" * 70)

    n_alpha, n_beta = mol.nelec

    for i in range(min(10, len(mf.mo_energy[0]))):
        e_alpha = mf.mo_energy[0][i] * 27.211386
        e_beta = mf.mo_energy[1][i] * 27.211386

        occ_a = "occ" if i < n_alpha else "virt"
        occ_b = "occ" if i < n_beta else "virt"

        print(f"{i + 1:5d} {e_alpha:12.4f} {e_beta:12.4f} α:{occ_a:5s} β:{occ_b:5s}")


# Приклади
analyze_d_orbitals("Sc", spin=1)
analyze_d_orbitals("Ti", spin=2)
analyze_d_orbitals("V", spin=3)
analyze_d_orbitals("Cr", spin=6)
analyze_d_orbitals("Mn", spin=5)
analyze_d_orbitals("Fe", spin=4)
analyze_d_orbitals("Co", spin=3)
analyze_d_orbitals("Ni", spin=2)
analyze_d_orbitals("Cu", spin=1)
analyze_d_orbitals("Zn", spin=0)
