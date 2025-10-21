"""
================================================================================
HF_procedure.py
================================================================================

ОПИС ПРОГРАМИ:
--------------
Детальна демонстрація методу Restricted Hartree-Fock (RHF) для атома гелію
у базисі Double Zeta (cc-pVDZ).

МЕТА:
-----
Навчальна програма, яка показує всі етапи RHF розрахунку "під капотом":
- Структуру базисних функцій (гаусові примітиви з експонентами і коефіцієнтами)
- Обчислення одно- та двоелектронних інтегралів
- Побудову початкових орбіталей
- Ітераційний SCF (Self-Consistent Field) процес
- Формування матриці Фока на кожному кроці
- Збіжність до самоузгодженого рішення
"""

import numpy as np
from pyscf import gto, scf
import sys

# Налаштування для повного виведення масивів
np.set_printoptions(precision=6, suppress=True, linewidth=120)

print("="*80)
print("РОЗРАХУНОК АТОМА ГЕЛІЮ МЕТОДОМ RHF У БАЗИСІ DOUBLE ZETA")
print("="*80)

# Створення молекули гелію
mol = gto.M(
    atom = 'He 0 0 0',
    basis = 'cc-pvdz',  # Double Zeta базис
    unit = 'Angstrom',
    symmetry = False,
    verbose = 0
)

print("\n" + "="*80)
print("1. ІНФОРМАЦІЯ ПРО МОЛЕКУЛУ ТА БАЗИС")
print("="*80)
print(f"Атом: {mol.atom}")
print(f"Заряд ядра: {mol.atom_charge(0)}")
print(f"Кількість електронів: {mol.nelectron}")
print(f"Базис: cc-pVDZ (Double Zeta)")
print(f"Кількість базисних функцій: {mol.nao}")

# Виведення детальної інформації про базис
print("\n" + "-"*80)
print("БАЗИСНІ ФУНКЦІЇ (Гаусові примітиви)")
print("-"*80)

# Правильний спосіб отримання базисної інформації
basis_set = mol._basis['He']
l_names = {0: 's', 1: 'p', 2: 'd', 3: 'f'}

for i, shell in enumerate(basis_set):
    angular_momentum = shell[0]
    print(f"\nОболонка {i+1}: {l_names.get(angular_momentum, angular_momentum)}-тип")

    # shell[1] містить список контракцій
    # Кожна контракція - це [експонента, коефіцієнт1, коефіцієнт2, ...]
    contractions = shell[1:]

    print(f"{'Експонента (α)':>20} {'Коефіцієнти':>20}")
    print("-"*42)

    for contraction in contractions:
        if isinstance(contraction, (list, tuple)):
            exponent = contraction[0]
            coeffs = contraction[1:]
            coeff_str = ", ".join([f"{c:.10f}" for c in coeffs])
            print(f"{exponent:>20.10f} {coeff_str}")

# Альтернативний спосіб - через PySCF функції
print("\n" + "-"*80)
print("ДЕТАЛІ БАЗИСУ (через PySCF API)")
print("-"*80)
for ib in range(mol.nbas):
    ia = mol.bas_atom(ib)
    l = mol.bas_angular(ib)
    nprim = mol.bas_nprim(ib)
    nctr = mol.bas_nctr(ib)
    print(f"\nБазисна функція {ib}: атом {ia}, l={l} ({l_names.get(l, l)}), "
          f"примітивів={nprim}, контракцій={nctr}")

    exps = mol.bas_exp(ib)
    coefs = mol.bas_ctr_coeff(ib)

    print(f"{'Експонента':>15} {'Коефіцієнт':>15}")
    print("-"*32)
    for j in range(nprim):
        for k in range(nctr):
            print(f"{exps[j]:>15.8f} {coefs[j,k]:>15.8f}")

# Математичний вигляд базисних функцій
print("\n" + "="*80)
print("МАТЕМАТИЧНИЙ ВИГЛЯД БАЗИСНИХ ФУНКЦІЙ")
print("="*80)
print("\nКожна базисна функція χ - це контрактована гаусіана:")
print("χ(r) = Σ d_i × g_i(r)")
print("де g_i(r) = N_i × r^l × exp(-α_i × r²)")
print("N_i - нормувальний множник, α_i - експонента, d_i - коефіцієнт контракції")

basis_functions = []
ao_labels = mol.ao_labels()

for ib in range(mol.nbas):
    l = mol.bas_angular(ib)
    nprim = mol.bas_nprim(ib)
    nctr = mol.bas_nctr(ib)
    exps = mol.bas_exp(ib)
    coefs = mol.bas_ctr_coeff(ib)

    for ictr in range(nctr):
        idx = len(basis_functions)
        print(f"\nχ_{idx+1} ({ao_labels[idx]}):")
        print("  χ(r) = ", end="")

        terms = []
        for iprim in range(nprim):
            coef = coefs[iprim, ictr]
            exp = exps[iprim]
            if abs(coef) > 1e-10:
                sign = "+" if coef >= 0 and len(terms) > 0 else ""
                terms.append(f"{sign}{coef:.6f}×exp(-{exp:.4f}r²)")

        print(" ".join(terms))
        basis_functions.append((idx, ao_labels[idx], exps, coefs[:, ictr]))

# Розрахунок інтегралів
print("\n" + "="*80)
print("2. ОДНОЕЛЕКТРОННІ ТА ДВОЕЛЕКТРОННІ ІНТЕГРАЛИ")
print("="*80)

# Матриця перекривання
S = mol.intor('int1e_ovlp')
print("\nМатриця перекривання S:")
print(S)

# Кінетична енергія
T = mol.intor('int1e_kin')
print("\nМатриця кінетичної енергії T:")
print(T)

# Потенціал ядра
V = mol.intor('int1e_nuc')
print("\nМатриця ядерного притягання V:")
print(V)

# Остовний гамільтоніан
H_core = T + V
print("\nОстовний гамільтоніан H_core = T + V:")
print(H_core)

# Двоелектронні інтеграли
eri = mol.intor('int2e')
print(f"\nДвоелектронні інтеграли (ERI) розмірність: {eri.shape}")
print(f"Приклади ERI[0,0,0,0] = {eri[0,0,0,0]:.6f}")
print(f"            ERI[0,0,1,1] = {eri[0,0,1,1]:.6f}")

# Енергія ядерного відштовхування
E_nuc = mol.energy_nuc()
print(f"\nЕнергія ядерного відштовхування: {E_nuc:.10f} Hartree")

print("\n" + "="*80)
print("3. ПОЧАТКОВІ НАБЛИЖЕННЯ")
print("="*80)

# Діагоналізація H_core для початкового наближення
S_minhalf = np.linalg.inv(np.linalg.cholesky(S))
print("\nS^(-1/2) матриця (для ортогоналізації):")
print(S_minhalf)

# Початкові МО коефіцієнти через діагоналізацію H_core
F_init = H_core
F_ortho = S_minhalf.T @ F_init @ S_minhalf
e_init, C_ortho_init = np.linalg.eigh(F_ortho)
C_init = S_minhalf @ C_ortho_init

print("\nПочаткові орбітальні енергії (з H_core):")
print(e_init)

print("\nПочаткові МО коефіцієнти C:")
print(C_init)

print("\n" + "-"*80)
print("ПОЧАТКОВІ МОЛЕКУЛЯРНІ ОРБІТАЛІ")
print("-"*80)
for i in range(mol.nao):
    print(f"\nφ_{i+1} (E = {e_init[i]:.6f} Hartree):")
    print("  φ(r) = ", end="")
    terms = []
    for j in range(mol.nao):
        coef = C_init[j, i]
        if abs(coef) > 1e-4:
            sign = "+" if coef >= 0 and len(terms) > 0 else ""
            terms.append(f"{sign}{coef:.6f}×χ_{j+1}")
    print(" ".join(terms) if terms else "0")

# Початкова матриця густини
n_occ = mol.nelectron // 2  # Кількість зайнятих орбіталей
P_init = 2 * C_init[:, :n_occ] @ C_init[:, :n_occ].T

print(f"\nКількість зайнятих орбіталей: {n_occ}")
print("\nПочаткова матриця густини P:")
print(P_init)

print("\n" + "="*80)
print("4. SCF ЦИКЛ (SELF-CONSISTENT FIELD)")
print("="*80)

# Ініціалізація змінних для SCF
P = P_init.copy()
max_iter = 50
conv_tol = 1e-8
E_old = 0.0

print(f"\nКритерій збіжності: {conv_tol}")
print(f"Максимальна кількість ітерацій: {max_iter}")

for iteration in range(max_iter):
    print("\n" + "="*80)
    print(f"ІТЕРАЦІЯ {iteration + 1}")
    print("="*80)

    # Побудова матриці Фока
    # G[μ,ν] = Σ_λσ P[λ,σ] * [(μν|λσ) - 0.5*(μλ|νσ)]
    G = np.zeros_like(H_core)
    for mu in range(mol.nao):
        for nu in range(mol.nao):
            for lam in range(mol.nao):
                for sig in range(mol.nao):
                    G[mu, nu] += P[lam, sig] * (eri[mu, nu, lam, sig] - 0.5 * eri[mu, lam, nu, sig])

    print("\nМатриця густини P:")
    print(P)

    print("\nМатриця G (електронне відштовхування):")
    print(G)

    # Матриця Фока
    F = H_core + G
    print("\nМатриця Фока F = H_core + G:")
    print(F)

    # Розрахунок електронної енергії
    E_elec = 0.5 * np.sum(P * (H_core + F))
    E_total = E_elec + E_nuc

    print(f"\nЕлектронна енергія: {E_elec:.10f} Hartree")
    print(f"Повна енергія: {E_total:.10f} Hartree")
    print(f"Зміна енергії: {E_total - E_old:.10e} Hartree")

    # Перевірка збіжності
    dE = abs(E_total - E_old)
    if dE < conv_tol and iteration > 0:
        print(f"\n{'*'*80}")
        print(f"ЗБІЖНІСТЬ ДОСЯГНУТА НА ІТЕРАЦІЇ {iteration + 1}")
        print(f"{'*'*80}")
        converged = True
        break

    E_old = E_total

    # Діагоналізація матриці Фока
    F_ortho = S_minhalf.T @ F @ S_minhalf
    orbital_energies, C_ortho = np.linalg.eigh(F_ortho)
    C = S_minhalf @ C_ortho

    print("\nОрбітальні енергії:")
    for i, e in enumerate(orbital_energies):
        occ_status = "зайнята" if i < n_occ else "віртуальна"
        print(f"  Орбіталь {i+1}: {e:.10f} Hartree ({occ_status})")

    print("\nМО коефіцієнти C:")
    print(C)

    print("\n" + "-"*40)
    print("ПОТОЧНІ МОЛЕКУЛЯРНІ ОРБІТАЛІ:")
    print("-"*40)
    for i in range(min(3, mol.nao)):  # Показуємо перші 3 орбіталі
        occ_status = "ЗАЙНЯТА" if i < n_occ else "ВІРТУАЛЬНА"
        print(f"\nφ_{i+1} ({occ_status}, E = {orbital_energies[i]:.6f} Hartree):")
        print("  φ(r) = ", end="")
        terms = []
        for j in range(mol.nao):
            coef = C[j, i]
            if abs(coef) > 1e-4:
                sign = "+" if coef >= 0 and len(terms) > 0 else ""
                terms.append(f"{sign}{coef:.6f}×χ_{j+1}")
        print(" ".join(terms) if terms else "0")

    # Оновлення матриці густини
    P = 2 * C[:, :n_occ] @ C[:, :n_occ].T

print("\n" + "="*80)
print("5. ФІНАЛЬНІ РЕЗУЛЬТАТИ")
print("="*80)

print(f"\nКількість SCF ітерацій: {iteration + 1}")
print(f"\nФінальна повна енергія: {E_total:.10f} Hartree")
print(f"Електронна енергія: {E_elec:.10f} Hartree")
print(f"Ядерна енергія: {E_nuc:.10f} Hartree")

print("\nФінальні орбітальні енергії:")
for i, e in enumerate(orbital_energies):
    occ_status = "зайнята" if i < n_occ else "віртуальна"
    print(f"  Орбіталь {i+1}: {e:.10f} Hartree ({occ_status})")

print("\nФінальні МО коефіцієнти:")
print(C)

print("\n" + "="*80)
print("ФІНАЛЬНІ МОЛЕКУЛЯРНІ ОРБІТАЛІ")
print("="*80)
for i in range(mol.nao):
    occ_status = "ЗАЙНЯТА" if i < n_occ else "ВІРТУАЛЬНА"
    print(f"\nφ_{i+1} ({occ_status}, E = {orbital_energies[i]:.8f} Hartree):")
    print("  φ(r) = ", end="")
    terms = []
    for j in range(mol.nao):
        coef = C[j, i]
        if abs(coef) > 1e-6:
            sign = "+" if coef >= 0 and len(terms) > 0 else ""
            terms.append(f"{sign}{coef:.6f}×χ_{j+1}")
    print(" ".join(terms) if terms else "0")

print("\nФінальна матриця густини:")
print(P)

print("\n" + "="*80)
print("6. ПЕРЕВІРКА З ВБУДОВАНИМ SCF PYSCF")
print("="*80)

# Запускаємо стандартний RHF для порівняння
mf = scf.RHF(mol)
mf.verbose = 0
E_pyscf = mf.kernel()

print(f"\nЕнергія з нашого коду: {E_total:.10f} Hartree")
print(f"Енергія з PySCF RHF:   {E_pyscf:.10f} Hartree")
print(f"Різниця:               {abs(E_total - E_pyscf):.10e} Hartree")

print("\n" + "="*80)
print("РОЗРАХУНОК ЗАВЕРШЕНО")
print("="*80)
