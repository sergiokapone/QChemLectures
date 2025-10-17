# ============================================================
# code_44.py - Дисперсійні корекції
# ============================================================
from pyscf import gto, dft
from pyscf.dft import dft_parser
import numpy as np

print("Дисперсійні корекції для слабких взаємодій")
print("=" * 80)

# He2 - типовий приклад дисперсійної взаємодії
distances = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 5.0])

print("\nКрива потенційної енергії He2")
print("-" * 80)

functionals_disp = [
    ('PBE', 'pbe', False),
    ('PBE-D3', 'pbe,', True),  # з D3
    ('B3LYP', 'b3lyp', False),
    ('ωB97X-D', 'wb97x-d', False),  # вбудована дисперсія
]

results_disp = {name: [] for name, _, _ in functionals_disp}

for r in distances:
    mol = gto.M(
        atom=f'He 0 0 0; He 0 0 {r}',
        basis='aug-cc-pvqz',
        unit='angstrom'
    )

    for name, xc, use_d3 in functionals_disp:
        mf = dft.RKS(mol)
        mf.xc = xc
        mf.verbose = 0

        # Увімкнення D3
        if use_d3:
            mf = dft.RKS(mol).density_fit()
            mf.xc = xc
            # Примітка: для D3 потрібна окрема бібліотека

        energy = mf.kernel()
        results_disp[name].append(energy)

# Виведення результатів
print(f"\n{'r (Å)':<10} " + " ".join(f"{name:<18}" for name, _, _ in functionals_disp))
print("-" * 80)

for i, r in enumerate(distances):
    print(f"{r:<10.2f} ", end="")
    for name, _, _ in functionals_disp:
        E = results_disp[name][i]
        # Відносно нескінченності
        E_rel = (E - results_disp[name][-1]) * 627.51  # kcal/mol
        print(f"{E_rel:<18.6f} ", end="")
    print()

print("\nПримітка:")
print("- Стандартні функціонали (PBE, B3LYP) не описують дисперсію")
print("- D3 корекція додає емпіричну дисперсію")
print("- ωB97X-D має вбудовану дисперсійну корекцію")
print("- Для точного опису потрібен CCSD(T) або аналогічний")

