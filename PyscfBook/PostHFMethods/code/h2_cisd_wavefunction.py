"""
h2_cisd_wavefunction.py

Аналіз CISD хвильової функції для молекули H2
==============================================

Цей скрипт виконує CISD (Configuration Interaction Singles and Doubles) розрахунок
для молекули водню H2 та показує повну хвильову функцію у вигляді лінійної комбінації
детермінантів Слейтера.

Що робить програма:
-------------------
1. Будує молекулу H2 з міжатомною відстанню 0.74 Å у базисі STO-3G
2. Виконує Hartree-Fock розрахунок (референсна конфігурація)
3. Виконує CISD розрахунок (враховує одиничні та подвійні збудження)
4. Виводить хвильову функцію у явному вигляді:
   |Ψ⟩ = c₀|HF⟩ + Σ cᵢᵃ|i→a⟩ + Σ cᵢⱼᵃᵇ|ij→ab⟩

Нотація детермінантів:
----------------------
- |110⟩ означає: орбіталі 0,1 зайняті, орбіталь 2 вільна
- |101⟩ означає: орбіталі 0,2 зайняті, орбіталь 1 вільна
- Детермінанти показують електронну конфігурацію у просторових орбіталях

Вихідні дані:
-------------
- Енергія HF та CISD
- Коефіцієнти при кожному детермінанті
- Тип збудження (reference, single, double)
- Норма хвильової функції
- Внесок кореляційних ефектів
"""

from pyscf import gto, scf, ci
import numpy as np

# Молекула H2
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')

# Hartree-Fock
mf = scf.RHF(mol).run()
e_hf = mf.e_tot
print(f"E(HF) = {e_hf:.6f} a.u.\n")

# CISD
mycisd = ci.CISD(mf)
e_corr_cisd, civec_cisd = mycisd.kernel()
e_tot_cisd = e_hf + e_corr_cisd
print(f"E(CISD) total = {e_tot_cisd:.6f} a.u.\n")

# Параметри системи
nocc = mol.nelectron // 2
nvir = mol.nao_nr() - nocc
norb = mol.nao_nr()

print("="*70)
print("CISD ХВИЛЬОВА ФУНКЦІЯ")
print("="*70)

# Збираємо всі детермінанти з коефіцієнтами
determinants = []

# Референсний детермінант
c0 = civec_cisd[0]
det_ref = f"|{'1'*nocc}{'0'*nvir}⟩"  # зайняті перші nocc орбіталей
determinants.append((c0, det_ref, "HF reference"))

# Одиничні збудження
singles_start = 1
singles_end = 1 + nocc * nvir
singles = civec_cisd[singles_start:singles_end].reshape(nocc, nvir)

for i in range(nocc):
    for a in range(nvir):
        coef = singles[i, a]
        if abs(coef) > 1e-8:
            # Створюємо детермінант з збудженням i→a
            occ_list = list(range(nocc))
            occ_list[occ_list.index(i)] = nocc + a
            occ_list.sort()

            det_str_list = ['0'] * norb
            for orb in occ_list:
                det_str_list[orb] = '1'
            det_str = '|' + ''.join(det_str_list) + '⟩'

            determinants.append((coef, det_str, f"i={i}→a={nocc+a}"))

# Подвійні збудження
doubles_start = singles_end
doubles = civec_cisd[doubles_start:].reshape(nocc, nocc, nvir, nvir)

for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                coef = doubles[i, j, a, b]
                if abs(coef) > 1e-8:
                    # Створюємо детермінант з подвійним збудженням
                    occ_list = list(range(nocc))
                    if i in occ_list:
                        occ_list.remove(i)
                    if j in occ_list:
                        occ_list.remove(j)
                    occ_list.extend([nocc + a, nocc + b])
                    occ_list.sort()

                    det_str_list = ['0'] * norb
                    for orb in occ_list:
                        det_str_list[orb] = '1'
                    det_str = '|' + ''.join(det_str_list) + '⟩'

                    determinants.append((coef, det_str, f"ij={i},{j}→ab={nocc+a},{nocc+b}"))

# Виводимо хвильову функцію
print("\n|Ψ_CISD⟩ = \n")
for idx, (coef, det, label) in enumerate(determinants):
    sign = '+' if coef >= 0 and idx > 0 else ''
    print(f"  {sign}{coef:9.6f} × {det:20s}  # {label}")

# Перевірка нормування
norm = sum(c**2 for c, _, _ in determinants)
print(f"\n{'='*70}")
print(f"Норма: ∑c² = {norm:.6f} (має бути 1.0)")
print(f"Кількість детермінантів: {len(determinants)}")

# Внесок референсної конфігурації
ref_contrib = c0**2 * 100
print(f"Внесок HF детермінанта: {ref_contrib:.2f}%")
print(f"Кореляційний внесок: {100-ref_contrib:.2f}%")

