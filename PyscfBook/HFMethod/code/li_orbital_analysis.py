"""
li_orbital_analysis.py
Аналіз молекулярних орбіталей атома літію методом ROHF
======================================================
Скрипт виконує розрахунок атома літію (Li) методом ROHF
з базисним набором STO-3G та виводить детальну інформацію
про заселеність молекулярних орбіталей.
"""
from pyscf import gto, scf
import numpy as np

# Створюємо молекулу літію
mol = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1)

# Виконуємо ROHF розрахунок
mf = scf.ROHF(mol).run()

# Отримуємо заселеності орбіталей
mo_occ = mf.mo_occ  # Для ROHF це один масив

# Отримуємо енергії орбіталей
mo_energy = mf.mo_energy

# Підраховуємо різні типи орбіталей
n_doubly_occ = int(np.sum(mo_occ == 2))  # Подвійно заповнені (закриті)
n_singly_occ = int(np.sum(mo_occ == 1))  # Одинично заповнені (відкриті)
n_virtual = int(np.sum(mo_occ == 0))     # Віртуальні

# Знаходимо HOMO та LUMO
homo_idx = np.where(mo_occ > 0)[0][-1] if np.any(mo_occ > 0) else None  # остання заповнена
lumo_idx = np.where(mo_occ == 0)[0][0] if np.any(mo_occ == 0) else None  # перша віртуальна

# Виведення в таблицю
print("\n" + "=" * 90)
print("ВІЗУАЛІЗАЦІЯ ЗАСЕЛЕНОСТІ ОРБІТАЛЕЙ (ROHF)")
print("=" * 90)
print(f"\n{'Орбіталь':<10} {'Спіни':<10} {'Заселеність':<15} {'Енергія (Ha)':<18} {'Тип':<20} {'Позначка':<10}")
print("-" * 90)

# Всі орбіталі
for i in range(len(mo_occ)):
    # Визначаємо позначення спінів
    if mo_occ[i] == 2:
        spin_mark = "↑↓"
        orbital_type = "закрита оболонка"
    elif mo_occ[i] == 1:
        spin_mark = "↑"
        orbital_type = "відкрита оболонка"
    else:
        spin_mark = "-"
        orbital_type = "віртуальна"

    # Позначка HOMO/LUMO
    label = ""
    if homo_idx is not None and i == homo_idx:
        label = "← HOMO"
    elif lumo_idx is not None and i == lumo_idx:
        label = "← LUMO"

    print(f"  {i+1:<8} {spin_mark:^10} {mo_occ[i]:^15.1f} {mo_energy[i]:>15.6f}   {orbital_type:<20} {label:<10}")

print("=" * 90)
print(f"\nСтатистика:")
print(f"  Подвійно заповнені (закриті): {n_doubly_occ}")
print(f"  Одинично заповнені (відкриті): {n_singly_occ}")
print(f"  Віртуальні: {n_virtual}")
print(f"  Всього електронів: {n_doubly_occ * 2 + n_singly_occ}")
print(f"  Спінова мультиплетність: {mol.spin + 1} (дублет)")

# Інформація про HOMO-LUMO
if homo_idx is not None and lumo_idx is not None:
    homo_lumo_gap = mo_energy[lumo_idx] - mo_energy[homo_idx]
    print(f"\n  HOMO (орбіталь {homo_idx + 1}): {mo_energy[homo_idx]:.6f} Ha")
    print(f"  LUMO (орбіталь {lumo_idx + 1}): {mo_energy[lumo_idx]:.6f} Ha")
    print(f"  HOMO-LUMO gap: {homo_lumo_gap:.6f} Ha ({homo_lumo_gap * 27.2114:.4f} eV)")

print("=" * 90)
