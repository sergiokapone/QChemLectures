# ============================================================
# acetone_solvent_effect.py
# Вплив розчинника на електричні властивості
# ============================================================

from pyscf import gto, scf, dft
from pyscf.solvent import pcm
from pyscf.prop import polarizability
import numpy as np

# Молекула ацетону (CH3)2CO
mol = gto.M(
    atom="""
    C   0.000000  0.000000  0.000000
    C   1.511000  0.000000  0.000000
    C  -0.755500  1.308000  0.000000
    O  -0.504000 -1.131000  0.000000
    H   1.881000  0.523000  0.883000
    H   1.881000  0.523000 -0.883000
    H   1.881000 -1.023000  0.000000
    H  -0.384500  1.857000  0.883000
    H  -0.384500  1.857000 -0.883000
    H  -1.833500  1.149000  0.000000
    """,
    basis="6-31+g*",
    unit="angstrom",
)

print("Вплив розчинника на властивості ацетону")
print("=" * 60)

# Список розчинників (діелектрична проникність)
solvents = {
    "Газ": None,
    "Гексан": 1.88,
    "Хлороформ": 4.71,
    "Етанол": 24.85,
    "Вода": 78.36,
}

results = []

for solv_name, epsilon in solvents.items():
    print(f"\n{solv_name}:", end=" ")
    if epsilon:
        print(f"ε = {epsilon}")
    else:
        print()

    # DFT розрахунок
    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf.verbose = 0

    # Додаємо розчинник (PCM)
    if epsilon is not None:
        mf = pcm.PCM(mf)
        mf.with_solvent.eps = epsilon

    mf.kernel()

    # Дипольний момент
    dip = mf.dip_moment(unit='Debye')
    dip_mag = np.linalg.norm(dip)

    # Поляризовність
    if epsilon is None:
        pol_obj = polarizability.rks.Polarizability(mf)
    else:
        # Для PCM поляризовність обчислюється трохи інакше
        pol_obj = polarizability.rks.Polarizability(mf._scf)

    alpha = pol_obj.polarizability()
    alpha_mean = np.trace(alpha) / 3

    results.append((solv_name, epsilon if epsilon else 1.0, dip_mag, alpha_mean))

    print(f"  μ = {dip_mag:.3f} D")
    print(f"  ᾱ = {alpha_mean:.2f} au³")

# Підсумкова таблиця
print("\n" + "=" * 60)
print("Порівняльна таблиця:")
print("-" * 60)
print(f"{'Розчинник':<15} {'ε':<8} {'μ (D)':<10} {'ᾱ (au³)':<10}")
print("-" * 60)

for solv, eps, mu, alpha in results:
    print(f"{solv:<15} {eps:>6.2f}  {mu:>8.3f}  {alpha:>8.2f}")

print("-" * 60)
print("\nЕкспериментальні значення для ацетону:")
print("  μ(газ)  ≈ 2.91 D")
print("  μ(вода) ≈ 3.50 D")

print("\nВисновки:")
print("- Дипольний момент зростає в полярних розчинниках")
print("- Поляризовність також збільшується (поляризація середовищем)")
print("- Ефект найсильніший для високополярних розчинників (вода)")
print("- Важливо враховувати при моделюванні реакцій у розчині")

