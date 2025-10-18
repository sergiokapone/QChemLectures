"""
Автоматична оптимізація геометрії H2+ за допомогою градієнтів
"""

import numpy as np
from pyscf import gto, scf
from pyscf.geomopt.geometric_solver import optimize

print("Оптимізація геометрії H2+ методом UHF")
print("=" * 60)

# Початкова геометрія (не обов'язково оптимальна)
mol = gto.Mole()
mol.atom = """
    H  0.0  0.0  0.0
    H  0.0  0.0  1.5
"""
mol.basis = "cc-pvtz"
mol.charge = 1
mol.spin = 1
mol.unit = "Bohr"
mol.build()

print("Початкова геометрія: R = 1.50 bohr")

# UHF метод
mf = scf.UHF(mol)

# Запуск оптимізації
print("\nОптимізація (це може зайняти кілька хвилин)...")
mol_eq = optimize(mf, maxsteps=50)

# Результати
coords = mol_eq.atom_coords()
R_optimized = np.linalg.norm(coords[1] - coords[0])
E_optimized = mf.e_tot

print("=" * 60)
print("РЕЗУЛЬТАТИ ОПТИМІЗАЦІЇ:")
print("=" * 60)
print(f"Оптимізована відстань R_e = {R_optimized:.6f} bohr")
print(f"                          = {R_optimized * 0.529177:.6f} Å")
print(f"Енергія в мінімумі E_e = {E_optimized:.8f} Ha")

# Енергія дисоціації
E_H = -0.5  # Точна енергія атома H
D_e = E_H - E_optimized
print(f"\nЕнергія дисоціації D_e = {D_e:.6f} Ha")
print(f"                        = {D_e * 27.2114:.4f} eV")
print(f"                        = {D_e * 627.509:.2f} kcal/mol")

# Порівняння з експериментом
D_e_exp = 2.79  # eV
error = abs(D_e * 27.2114 - D_e_exp) / D_e_exp * 100
print(f"\nЕкспериментальне D_e = {D_e_exp:.2f} eV")
print(f"Відносна похибка = {error:.2f}%")

# Збереження оптимізованої геометрії
with open("h2plus_optimized.xyz", "w") as f:
    f.write("2\n")
    f.write(f"H2+ optimized, E = {E_optimized:.8f} Ha\n")
    for i, coord in enumerate(coords):
        f.write(
            f"H  {coord[0] * 0.529177:.6f}  {coord[1] * 0.529177:.6f}  {coord[2] * 0.529177:.6f}\n"
        )

print("\nОптимізовану геометрію збережено: h2plus_optimized.xyz")
