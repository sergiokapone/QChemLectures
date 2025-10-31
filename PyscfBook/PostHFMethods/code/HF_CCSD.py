# Демонстрація CC: HF -> CCSD -> амплітуди -> наближені коефіцієнти детермінантів
# Запустіть у середовищі з встановленим PySCF.
import numpy as np
from pyscf import gto, scf, mp, cc
from itertools import product

# ---------------------------
# Параметри обчислення
# ---------------------------
mol_spec = 'H 0 0 0; F 0 0 0.917'   # геометрія (Å)
basis = 'cc-pVDZ'                   # базис (змінити за потреби)
max_print = 10                       # скільки найсильніших амплітуд вивести

print("="*70)
print("ДЕМОНСТРАЦІЯ МЕТОДУ ЗВ'ЯЗАНИХ КЛАСТЕРІВ (CCSD)")
print("="*70)
print(f"\nМолекула: {mol_spec}")
print(f"Базис:    {basis}\n")

# ---------------------------
# Побудова молекули і HF
# ---------------------------
mol = gto.M(atom=mol_spec, basis=basis, verbose=0)
mf = scf.RHF(mol).run(verbose=0)

# ---------------------------
# MP2, CCSD, CCSD(T)
# ---------------------------
ccsd = cc.CCSD(mf)
e_ccsd_corr, t1, t2 = ccsd.kernel()

# print(eccsd, t1, t2)
et = ccsd.ccsd_t()

# ---------------------------
# 1. ВИВЕДЕННЯ ЕНЕРГІЙ
# ---------------------------
print("-"*70)
print("1. ЕНЕРГІЇ ТА КОРЕЛЯЦІЙНІ ВНЕСКИ")
print("-"*70)

e_hf = mf.e_tot
e_ccsd = ccsd.e_tot
e_ccsd_t = ccsd.e_tot + et

print(f"{'Метод':<10} {'Повна енергія (Ha)':>25} {'Кореляційний внесок (Ha)':>30}")
print("-"*70)
print(f"{'HF':<10} {e_hf:25.10f} {'---':>30}")
print(f"{'CCSD':<10} {e_ccsd:25.10f} {e_ccsd_corr:30.8f}")
print(f"{'CCSD(T)':<10} {e_ccsd_t:25.10f} {(e_ccsd_corr + et):30.8f}")
print("-"*70)


# ---------------------------
# Розміри
# ---------------------------
nocc = mf.mo_occ.nonzero()[0].size
nmo = mf.mo_coeff.shape[1]
nvir = nmo - nocc

# ---------------------------
# Діагностика T1
# ---------------------------
t1_norm = np.linalg.norm(t1)
max_t1 = np.max(np.abs(t1))

print("\n" + "="*70)
print("ДІАГНОСТИКА АМПЛІТУД")
print("="*70)
print(f"\n‖T₁‖ = {t1_norm:.6f}")
print(f"max|T₁| = {max_t1:.6f}", end="")
if max_t1 > 0.02:
    print("  ⚠️  УВАГА: можлива мультиреферентність!")
else:
    print("  ✓ Одноконфігураційна система")

# ---------------------------
# 2. РОЗКЛАД ХВИЛЬОВОЇ ФУНКЦІЇ
# ---------------------------
print("\n" + "="*70)
print("2. РОЗКЛАД ХВИЛЬОВОЇ ФУНКЦІЇ ЧЕРЕЗ ЗБУДЖЕНІ ДЕТЕРМІНАНТИ")
print("="*70)

# Знаходимо найбільші амплітуди T1 та T2
t1_flat = []
for i, a in product(range(nocc), range(nvir)):
    val = t1[i, a]
    if abs(val) > 1e-10:
        t1_flat.append(((i, a+nocc), abs(val), val))
t1_flat.sort(key=lambda x: x[1], reverse=True)

t2_flat = []
for i, j, a, b in product(range(nocc), range(nocc), range(nvir), range(nvir)):
    val = t2[i, j, a, b]
    if abs(val) > 1e-10:
        t2_flat.append(((i, j, a+nocc, b+nocc), abs(val), val))
t2_flat.sort(key=lambda x: x[1], reverse=True)

# Приблизні коефіцієнти подвійних збуджень
# c_{ij}^{ab} ≈ t2_{ij}^{ab} + 0.5*(t1_i^a * t1_j^b - t1_i^b * t1_j^a)
coeff_doubles = {}

for i in range(nocc):
    for j in range(i+1, nocc):
        for a in range(nvir):
            for b in range(a+1, nvir):
                A, B = a + nocc, b + nocc

                # Внесок від T2
                val_t2 = t2[i, j, a, b]

                # Внесок від T1² (антисиметризований)
                prod = 0.5 * (t1[i, a] * t1[j, b] - t1[i, b] * t1[j, a])

                c_total = val_t2 + prod
                if abs(c_total) > 1e-10:
                    coeff_doubles[(i, j, A, B)] = c_total

# Сортуємо всі коефіцієнти
all_coeffs = []
all_coeffs.append(('|Φ₀⟩', 1.0, 1.0))  # Основний детермінант (відносна нормалізація)

for (i, a), mag, val in t1_flat[:max_print]:
    all_coeffs.append((f'|Φ_{i}^{a}⟩', mag, val))

for (i, j, A, B), val in coeff_doubles.items():
    all_coeffs.append((f'|Φ_{i}{j}^{A}{B}⟩', abs(val), val))

# Сортуємо за величиною
all_coeffs.sort(key=lambda x: x[1], reverse=True)

# Виводимо розклад
print("\n|Ψ_CCSD⟩ = exp(T̂₁ + T̂₂)|Φ₀⟩ ≈ ")
print()

# Виводимо перші max_print + 1 компонентів
for i, (det, mag, val) in enumerate(all_coeffs[:max_print + 1]):
    sign = '+' if val >= 0 else '-'

    if i == 0:
        print(f"       {val:+.6f} {det}")
    else:
        print(f"     {sign} {abs(val):.6f} {det}")

if len(all_coeffs) > max_print + 1:
    print(f"     + ... (ще {len(all_coeffs) - max_print - 1} детермінантів)")

# Статистика
n_singles = len([x for x in all_coeffs if 'Φ_' in x[0] and '^' in x[0] and x[0].count('Φ_') == 1])
n_doubles = len([x for x in all_coeffs if x[0].count('^') == 1 and len(x[0].split('^')[0].split('_')) > 2])

print(f"\nСтатистика детермінантів (|c| > 1e-10):")
print(f"  Основний детермінант:   1")
print(f"  Одиничні збудження:     {n_singles}")
print(f"  Подвійні збудження:     {n_doubles}")
print(f"  Всього:                 {1 + n_singles + n_doubles}")

# ---------------------------
# 3. ТОП АМПЛІТУД
# ---------------------------
print("\n" + "="*70)
print("3. НАЙБІЛЬШІ КЛАСТЕРНІ АМПЛІТУДИ")
print("="*70)

print(f"\nТоп-{max_print} амплітуд T₁ (одиничні збудження i→a):")
print("-"*70)
print(f"{'Індекси':<15} {'Амплітуда':<20} {'|Величина|':<15}")
print("-"*70)
for (i, a), mag, val in t1_flat[:max_print]:
    print(f"t₁[{i}→{a}]{'':<7} {val:+.6e}{'':<8} {mag:.2e}")

print(f"\nТоп-{max_print} амплітуд T₂ (подвійні збудження ij→ab):")
print("-"*70)
print(f"{'Індекси':<15} {'Амплітуда':<20} {'|Величина|':<15}")
print("-"*70)
for (i, j, a, b), mag, val in t2_flat[:max_print]:
    print(f"t₂[{i}{j}→{a}{b}]{'':<5} {val:+.6e}{'':<8} {mag:.2e}")

# Збереження результатів
results = {
    'e_hf': e_hf, 'e_ccsd': e_ccsd, 'e_ccsd_t': e_ccsd_t,
    't1': t1, 't2': t2, 'c_doubles': coeff_doubles
}

