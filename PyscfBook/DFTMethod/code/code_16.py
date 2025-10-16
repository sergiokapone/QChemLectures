from pyscf import gto, dft
import numpy as np


def dft_second_period(functional="pbe", basis="def2-tzvp"):
    """
    Розрахунок всіх атомів другого періоду
    """

    atoms_data = [
        ("Li", 1, "²S"),
        ("Be", 0, "¹S"),
        ("B", 1, "²P"),
        ("C", 2, "³P"),
        ("N", 3, "⁴S"),
        ("O", 2, "³P"),
        ("F", 1, "²P"),
        ("Ne", 0, "¹S"),
    ]

    print(f"\nАтоми 2-го періоду ({functional.upper()}/{basis})")
    print("=" * 80)
    print(
        f"{'Атом':4s} {'Терм':6s} {'2S':3s} {'Енергія, Ha':15s} "
        f"{'Енергія, eV':12s} {'<S²>':8s}"
    )
    print("-" * 80)

    energies = {}

    for symbol, spin, term in atoms_data:
        mol = gto.M(
            atom=f"{symbol} 0 0 0", basis=basis, spin=spin, symmetry=True, verbose=0
        )

        if spin == 0:
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)

        mf.xc = functional
        mf.conv_tol = 1e-10
        energy = mf.kernel()

        energies[symbol] = energy
        e_ev = energy * 27.211386

        if spin == 0:
            s2 = 0.0
        else:
            s2_result = mf.spin_square()
            s2 = s2_result[0]

        print(f"{symbol:4s} {term:6s} {spin:3d} {energy:15.8f} {e_ev:12.2f} {s2:8.4f}")

    print("=" * 80)

    return energies


# Розрахунки з різними функціоналами
energies_pbe = dft_second_period("pbe")
energies_b3lyp = dft_second_period("b3lyp")
energies_pbe0 = dft_second_period("pbe0")
