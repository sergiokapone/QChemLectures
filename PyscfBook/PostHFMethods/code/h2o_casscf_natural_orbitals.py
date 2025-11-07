"""
Файл: h2o_casscf_natural_orbitals.py
Розрахунок CASSCF з натуральними орбіталями з CISD для молекули води

Цей скрипт виконує розрахунок CASSCF з використанням натуральних орбіталей,
отриманих з CISD для молекули води (H2O). Робочий процес включає:
1. RHF розрахунок для початкових орбіталей
2. CISD розрахунок для кореляційних ефектів
3. Аналіз натуральних орбіталей з 1-RDM CISD
4. Вибір активного простору на основі заселеностей натуральних орбіталей
5. Розрахунок CASSCF в базисі натуральних орбіталей

Натуральні орбіталі допомагають у виборі фізично більш обґрунтованого
активного простору для розрахунку CASSCF.
"""

import numpy as np
from pyscf import gto, scf, ci, mcscf

# --- 1. Створюємо молекулу і виконуємо RHF ---
mol = gto.M(
    atom = '''
        O  0.0000  0.0000  0.0000
        H  0.7586  0.0000  0.5043
        H -0.7586  0.0000  0.5043
    ''',
    basis = 'cc-pvdz',
    spin = 0,
    verbose = 0
)
mf = scf.RHF(mol).run()

# --- 2. Корельований розрахунок CI (CISD) ---
myci = ci.CISD(mf).run()

# --- 3. Отримання 1-RDM і побудова натуральних орбіталей ---
dm1 = myci.make_rdm1()
occs, natorbs = np.linalg.eigh(dm1)   # ← правильний порядок!

# Сортуємо за спаданням заселеностей
idx = np.argsort(-occs)
occs = occs[idx]
natorbs = natorbs[:, idx]

print("Заселеності натуральних орбіталей (CISD):")
for i, n in enumerate(occs):
    print(f"  Орбіталь {i+1:2d}:  n = {n:.4f}")

# --- 4. Вибір активного простору ---
# Наприклад: орбіталі з 0.02 < n < 1.98
active = [i for i, n in enumerate(occs) if 0.02 < n < 1.98]
ncas = len(active)
nelecas = sum(occs[i] for i in active)

print(f"\nАктивний простір (CAS): {ncas} орбіталей, {nelecas:.1f} електронів")
print("Індекси активних орбіталей:", active)

# --- 5. Перехід до базису натуральних орбіталей ---
mf.mo_coeff = mf.mo_coeff @ natorbs

# --- 6. CASSCF у базисі натуральних орбіталей ---
mc = mcscf.CASSCF(mf, ncas, round(nelecas))
mc.kernel()

# --- 7. Аналіз заселеностей після CASSCF ---
print("\nНатуральні заселеності після CASSCF:")
dm1_cas = mc.make_rdm1()  # <-- тут уже все пораховано
occs_cas, _ = np.linalg.eigh(dm1_cas)
occs_cas = np.sort(occs_cas)[::-1]  # сортуємо за спаданням
for i, n in enumerate(occs_cas):
    print(f"  Орбіталь {i+1:2d}:  n = {n:.4f}")

