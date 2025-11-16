# ============================================================
# h2o_atomic_charges_comparison.py
# Порівняння різних схем атомних зарядів
# ============================================================

from pyscf import gto, scf
from pyscf.lo import iao
import numpy as np

mol = gto.M(
    atom="""
    O  0.0000  0.0000  0.1173
    H  0.0000  0.7572 -0.4692
    H  0.0000 -0.7572 -0.4692
    """,
    basis="6-31g*",
    unit="angstrom",
)

print("Порівняння схем атомних зарядів для H2O")
print("=" * 60)

# SCF розрахунок
mf = scf.RHF(mol)
mf.verbose = 0
mf.kernel()

# Отримуємо матрицю густини
dm = mf.make_rdm1()

# ===== 1. Заряди Малікена (Mulliken) =====
print("\n1. Аналіз популяцій Малікена:")
mulliken = mf.mulliken_pop()
charges_mulliken = mulliken[1]

print("   Атом   Заряд")
print("   " + "-" * 20)
for ia in range(mol.natm):
    symbol = mol.atom_symbol(ia)
    charge = charges_mulliken[ia]
    print(f"   {symbol:<4}  {charge:>8.4f}")

# ===== 2. Заряди Льовдіна (Löwdin) =====
print("\n2. Аналіз популяцій Льовдіна:")

# Матриця перекриття
s = mf.get_ovlp()
s_sqrt = np.linalg.matrix_power(s, 0.5)

# Льовдін: P_Löwdin = S^(1/2) D S^(1/2)
dm_lowdin = s_sqrt @ dm @ s_sqrt

# Розподіл по атомах
from pyscf import lo
charges_lowdin = np.zeros(mol.natm)

# Підраховуємо електрони на кожному атомі
for ia in range(mol.natm):
    ao_indices = np.where(mol.aoslice_by_atom()[:,0] == ia)[0]
    if len(ao_indices) == 0:
        ao_start = mol.aoslice_by_atom()[ia,2]
        ao_end = mol.aoslice_by_atom()[ia,3]
        charges_lowdin[ia] = mol.atom_charge(ia) - np.trace(dm_lowdin[ao_start:ao_end, ao_start:ao_end])
    else:
        ao_start = mol.aoslice_by_atom()[ia,2]
        ao_end = mol.aoslice_by_atom()[ia,3]
        charges_lowdin[ia] = mol.atom_charge(ia) - np.trace(dm_lowdin[ao_start:ao_end, ao_start:ao_end])

print("   Атом   Заряд")
print("   " + "-" * 20)
for ia in range(mol.natm):
    symbol = mol.atom_symbol(ia)
    charge = charges_lowdin[ia]
    print(f"   {symbol:<4}  {charge:>8.4f}")

# ===== 3. ESP-заряди (CHELPG-подібний метод) =====
print("\n3. ESP-заряди (спрощений CHELPG):")
print("   (Для повного CHELPG потрібен додатковий модуль)")

# Простий варіант: заряди з дипольного моменту
# Більш точний ESP вимагає окремої бібліотеки

dip = mf.dip_moment()
# Наближена оцінка зарядів з дипольного моменту
# μ = Σ q_i r_i

print("   Для точних ESP-зарядів використовуйте:")
print("   - RESP (psi4, Gaussian)")
print("   - CHELPG (Gaussian, Multiwfn)")
print("   - Bader (QTAIM) (Multiwfn, critic2)")

# ===== Порівняльна таблиця =====
print("\n" + "=" * 60)
print("Порівняльна таблиця:")
print("-" * 60)
print(f"{'Атом':<6} {'Mulliken':<12} {'Löwdin':<12}")
print("-" * 60)

for ia in range(mol.natm):
    symbol = mol.atom_symbol(ia)
    print(f"{symbol:<6} {charges_mulliken[ia]:>10.4f}  {charges_lowdin[ia]:>10.4f}")

print("-" * 60)
print(f"{'Сума':<6} {np.sum(charges_mulliken):>10.4f}  {np.sum(charges_lowdin):>10.4f}")

print("\nРекомендації:")
print("- Mulliken: швидкий, але залежить від базису")
print("- Löwdin: стабільніший за Mulliken")
print("- ESP (RESP/CHELPG): найкраще для силових полів")
print("- Bader (AIM): найбільш фізично обґрунтований")

