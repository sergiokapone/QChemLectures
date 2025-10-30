"""
Квантовий хімічний розрахунок молекули метану (CH₄) з використанням PySCF.

Мета:
    Порівняти енергії та кореляційні ефекти для методів:
        - Hartree-Fock (HF)
        - CISD (Configuration Interaction Singles and Doubles)
        - FCI (Full Configuration Interaction — точний результат у заданому базисі)

Система:
    Молекула CH₄ у тетраедричній геометрії (Td симетрія).
    Координати оптимізовані для рівноважної геометрії при базисі STO-3G.

Базис:
    STO-3G — мінімальний базис, дозволяє виконати FCI для демонстрації.

Функціональність:
    - Виведення загальної інформації про молекулу (електрони, орбіталі, симетрія)
    - Розрахунок HF, CISD, FCI з замірами часу
    - Порівняння енергій та кореляційних внесків
    - Оцінка відсотка кореляції, врахованої CISD
    - Виведення розмірності FCI простору

Зауваження:
    - FCI є точним у межах базису, але масштабується як ~N⁶ (N — кількість орбіталей).
    - Для STO-3G: 9 орбіталей → ~4000 детермінантів → швидко.
    - Для більших базисів (наприклад, cc-pVDZ) FCI стає неможливим.
"""

import numpy as np
from pyscf import gto, scf, ci, fci
import time


# Метан CH4 (тетраедрична геометрія)
mol = gto.M(
    atom='''
    C     0.000000    0.000000    0.000000
    H     0.629118    0.629118    0.629118
    H    -0.629118   -0.629118    0.629118
    H    -0.629118    0.629118   -0.629118
    H     0.629118   -0.629118   -0.629118
    ''',
    basis='6-31g',
    symmetry=True
)

print(f"Електронів: {mol.nelectron}")
print(f"Орбіталей: {mol.nao_nr()}")
print(f"Точкова група: {mol.groupname}\n")
print(f"Базис: {mol.basis}\n")

# HF
print("="*60)
print("Hartree-Fock розрахунок...")
print("="*60)
start_hf = time.time()
mf = scf.RHF(mol).run(verbose=0)
end_hf = time.time()
print(f"E(HF) = {mf.e_tot:.8f} Hartree")
print(f"Час HF: {end_hf - start_hf:.3f} с\n")

# CISD
print("="*60)
print("CISD розрахунок...")
print("="*60)
mycisd = ci.CISD(mf)
mycisd.verbose = 0

start_cisd = time.time()
e_cisd_corr, civec_cisd = mycisd.kernel()
end_cisd = time.time()

e_cisd_tot = mf.e_tot + e_cisd_corr
print(f"E(CISD) = {e_cisd_tot:.8f} Hartree")
print(f"Correlation energy = {e_cisd_corr:.8f} Hartree")
print(f"Час CISD: {end_cisd - start_cisd:.3f} с\n")

# FCI
print("="*60)
print("FCI розрахунок (це може зайняти хвилину)...")
print("="*60)
myfci = fci.FCI(mf)

start_fci = time.time()
e_fci, civec_fci = myfci.kernel()
end_fci = time.time()

print(f"E(FCI) = {e_fci:.8f} Hartree")
print(f"Час FCI: {end_fci - start_fci:.3f} с\n")

# Порівняння
print("="*70)
print("ПІДСУМКОВІ РЕЗУЛЬТАТИ")
print("="*70)
print(f"{'Метод':<15} {'Енергія (Hartree)':<20} {'Відносно HF (mHa)':<20} {'Час (с)':<10}")
print("-"*80)
print(f"{'HF':<15} {mf.e_tot:>18.8f} {0.0:>18.3f} {end_hf - start_hf:>9.3f}")
print(f"{'CISD':<15} {e_cisd_tot:>18.8f} {e_cisd_corr*1000:>18.3f} {end_cisd - start_cisd:>9.3f}")
print(f"{'FCI ':<15} {e_fci:>18.8f} {(e_fci - mf.e_tot)*1000:>18.3f} {end_fci - start_fci:>9.3f}")
print(f"{'Experiment (0K)':<15} {-40.4323:>18.8f} {'—':>18} {'—':>9}")

print("\n" + "="*70)
print("АНАЛІЗ КОРЕЛЯЦІЙНИХ ЕФЕКТІВ")
print("="*70)
print(f"Кореляція врахована CISD:        {e_cisd_corr*1000:>10.3f} mHartree")
print(f"Повна кореляція (FCI):           {(e_fci - mf.e_tot)*1000:>10.3f} mHartree")
print(f"Втрачена кореляція (T+Q+...):    {(e_fci - e_cisd_tot)*1000:>10.3f} mHartree")

# Відсоток
if abs(e_cisd_corr) > 1e-10:
    cisd_percentage = abs(e_cisd_corr / (e_fci - mf.e_tot) * 100)
    missing_percentage = 100 - cisd_percentage
    print(f"\nCISD враховує:                   {cisd_percentage:>10.1f}% кореляції")
    print(f"Втрачено (трипли+квадрупли):     {missing_percentage:>10.1f}% кореляції")

# Розмірність FCI
norb = mol.nao_nr()
nelec = mol.nelectron
na = nelec // 2
nb = nelec // 2
n_dets = fci.cistring.num_strings(norb, na) * fci.cistring.num_strings(norb, nb)
print(f"\nРозмірність FCI простору:        {n_dets:>10d} детермінантів")
print(f"Час розрахунку FCI зростає як ~N⁶ для {norb} орбіталей")

