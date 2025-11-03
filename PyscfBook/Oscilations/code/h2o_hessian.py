# ============================================================
# h2o_hessian.py
# Обчислення гесіану чисельним та аналітичним способами
# Їх порівняння
# ============================================================

import time

import numpy as np
from pyscf import gto, scf
from pyscf.hessian import rhf as rhf_hess

# Створення молекули води
mol = gto.M(
    atom="""
    O  0.0000  0.0000  0.1173
    H  0.0000  0.7572 -0.4692
    H  0.0000 -0.7572 -0.4692
    """,
    basis="6-31g",
    unit="angstrom",
)

print("=" * 60)
print("Обчислення Гесіану для молекули води (H2O)")
print("=" * 60)
print(f"\nБазис: {mol.basis}")
print(f"Кількість атомів: {mol.natm}")
print(f"Розмір Гесіану: {3 * mol.natm} x {3 * mol.natm}")

# Виконання SCF розрахунку
print("\n--- Розрахунок SCF ---")
mf = scf.RHF(mol)
mf.kernel()
print(f"Енергія SCF: {mf.e_tot:.10f} Ha")

# =====================================================
# 1. ЧИСЕЛЬНИЙ ГЕСІАН (числове диференціювання градієнта)
# =====================================================
print("\n" + "=" * 60)
print("1. ЧИСЕЛЬНИЙ ГЕСІАН")
print("=" * 60)
print("Обчислюємо через скінченні різниці градієнтів...")

start_time = time.time()

# Крок для числового диференціювання
step = 0.001  # в атомних одиницях (Bohr)

natm = mol.natm
hess_numerical = np.zeros((natm, natm, 3, 3))

# Обчислюємо градієнт для кожного зміщення
for atom_i in range(natm):
    for coord_i in range(3):
        # Позитивне зміщення
        coords_pos = mol.atom_coords().copy()
        coords_pos[atom_i, coord_i] += step

        mol_pos = gto.M(
            atom=[[mol.atom_symbol(i), coords_pos[i]] for i in range(natm)],
            basis=mol.basis,
            unit="Bohr",
        )
        mf_pos = scf.RHF(mol_pos)
        mf_pos.verbose = 0
        mf_pos.kernel()
        grad_pos = mf_pos.Gradients().kernel()

        # Негативне зміщення
        coords_neg = mol.atom_coords().copy()
        coords_neg[atom_i, coord_i] -= step

        mol_neg = gto.M(
            atom=[[mol.atom_symbol(i), coords_neg[i]] for i in range(natm)],
            basis=mol.basis,
            unit="Bohr",
        )
        mf_neg = scf.RHF(mol_neg)
        mf_neg.verbose = 0
        mf_neg.kernel()
        grad_neg = mf_neg.Gradients().kernel()

        # Числова похідна: d²E/dxi dxj = (dE/dxj|xi+h - dE/dxj|xi-h) / 2h
        hess_numerical[:, atom_i, :, coord_i] = (grad_pos - grad_neg) / (2 * step)

numerical_time = time.time() - start_time

print(f"\nЧас обчислення: {numerical_time:.2f} с")
print(f"Форма Гесіану: {hess_numerical.shape}")

# Перетворюємо в 2D для виводу
hess_num_2d = hess_numerical.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)
print("\nЧисельний Гесіан (перші 6x6 елементів):")
print(hess_num_2d[:6, :6])

# =====================================================
# 2. АНАЛІТИЧНИЙ ГЕСІАН (через аналітичні похідні)
# =====================================================
print("\n" + "=" * 60)
print("2. АНАЛІТИЧНИЙ ГЕСІАН")
print("=" * 60)

start_time = time.time()

# Аналітичний Гесіан через pyscf.hessian.rhf
hess_analytical = rhf_hess.Hessian(mf).kernel()

analytical_time = time.time() - start_time

print(f"\nЧас обчислення: {analytical_time:.2f} с")
print(f"Форма Гесіану: {hess_analytical.shape}")

# Перетворюємо в 2D для виводу
hess_anal_2d = hess_analytical.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)
print("\nАналітичний Гесіан (перші 6x6 елементів):")
print(hess_anal_2d[:6, :6])

# =====================================================
# ПОРІВНЯННЯ РЕЗУЛЬТАТІВ
# =====================================================
print("\n" + "=" * 60)
print("ПОРІВНЯННЯ РЕЗУЛЬТАТІВ")
print("=" * 60)

difference = hess_num_2d - hess_anal_2d
max_diff = np.max(np.abs(difference))
mean_diff = np.mean(np.abs(difference))

print(f"\nМаксимальна різниця: {max_diff:.2e}")
print(f"Середня абсолютна різниця: {mean_diff:.2e}")
print(f"\nПрискорення (аналітичний швидший у): {numerical_time / analytical_time:.2f}x")

print("\nМатриця різниць (перші 6x6 елементів):")
print(difference[:6, :6])

# =====================================================
# ВЛАСНІ ЗНАЧЕННЯ (частоти коливань)
# =====================================================
print("\n" + "=" * 60)
print("ВЛАСНІ ЗНАЧЕННЯ ГЕСІАНУ (з аналітичного)")
print("=" * 60)

# Перетворюємо Гесіан з форми (natm, natm, 3, 3) в (3*natm, 3*natm)
hess_2d = hess_analytical.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)

# Масово-зважений Гесіан для частот
mass = np.array([mol.atom_mass_list()[i] for i in range(natm)])
mass_matrix = np.repeat(mass, 3)
mass_weighted_hess = hess_2d / np.sqrt(mass_matrix[:, None] * mass_matrix[None, :])

eigenvalues, eigenvectors = np.linalg.eigh(mass_weighted_hess)

# Конвертація в частоти (cm^-1)
# 1 Hartree/(amu*Bohr^2) * (a0/Bohr)^2 = omega^2
# omega = sqrt(eigenvalue) * conversion_factor
conversion = 5140.48  # приблизний коефіцієнт для Ha/(amu*angstrom^2) -> cm^-1

frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * conversion

print("\nЧастоти коливань (см^-1):")
print("(перші 6 значень мають бути близькі до нуля - трансляції та обертання)")
for i, freq in enumerate(frequencies):
    mode_type = "трансляція/обертання" if i < 6 else "коливання"
    print(f"  Мода {i + 1}: {freq:10.2f} см^-1  ({mode_type})")

print("\n" + "=" * 60)
print("ПРИМІТКИ:")
print("=" * 60)
print("1. Чисельний метод: диференціювання градієнтів методом центральних різниць")
print("2. Аналітичний метод: pyscf.hessian.rhf.Hessian - точні другі похідні")
print("3. Аналітичний метод набагато швидший та точніший!")
print("=" * 60)
