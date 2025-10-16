from pyscf import gto, dft
import numpy as np


def test_grid_quality(symbol, spin, functional="pbe", basis="cc-pvtz"):
    """
    Тестування впливу якості сітки на результати
    """

    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

    # Різні рівні сіток
    grids = [
        (1, "грубо"),
        (2, "середньо"),
        (3, "добре (за замовчуванням)"),
        (4, "дуже добре"),
        (5, "ультра добре"),
    ]

    print(f"\nВплив якості сітки на енергію {symbol}")
    print(f"Функціонал: {functional.upper()}, базис: {basis}")
    print("=" * 70)
    print(f"{'Рівень':8s} {'Опис':30s} {'Енергія, Ha':15s} {'ΔE, μHa':10s}")
    print("-" * 70)

    energies = []

    for level, description in grids:
        if spin == 0:
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)

        mf.xc = functional
        mf.grids.level = level
        mf.verbose = 0
        mf.conv_tol = 1e-11

        energy = mf.kernel()
        energies.append(energy)

        if len(energies) == 1:
            e_ref = energy
            delta = 0.0
        else:
            delta = (energy - e_ref) * 1e6  # microHartree

        print(f"{level:8d} {description:30s} {energy:15.8f} {delta:10.2f}")

    print("=" * 70)

    # Оцінка збіжності
    if len(energies) >= 3:
        conv = abs(energies[-1] - energies[-2]) * 1e6
        print(f"\nЗбіжність (рівні 4→5): {conv:.4f} μHa")
        if conv < 1.0:
            print("Сітка рівня 4 достатня для точних розрахунків")
        else:
            print("Для високої точності використовуйте рівень 5")


# Тестування
test_grid_quality("C", spin=2)
test_grid_quality("Fe", spin=4, basis="def2-svp")
test_grid_quality("Kr", spin=0)
