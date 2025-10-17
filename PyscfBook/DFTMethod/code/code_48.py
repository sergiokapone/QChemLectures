# ============================================================
# code_48.py - TD-DFT для збуджених станів
# ============================================================
from pyscf import gto, dft
from pyscf.tdscf import TDDFT

print("TD-DFT розрахунок збуджених станів N2")
print("=" * 80)

mol = gto.M(
    atom='N 0 0 0; N 0 0 1.098',
    basis='aug-cc-pvtz',
    unit='angstrom',
    symmetry=True
)

# Основний стан
mf = dft.RKS(mol)
mf.xc = 'pbe0'
mf.kernel()

print(f"Енергія основного стану: {mf.e_tot:.8f} Ha")
print(f"Точкова група: {mol.groupname}")

# TD-DFT
print("\n" + "=" * 80)
print("Вертикальні збудження (TD-PBE0/aug-cc-pVTZ):")
print("=" * 80)

td = TDDFT(mf)
td.nstates = 10  # Перші 10 станів

# Розрахунок
excitations = td.kernel()[0]  # Енергії збуджень в Ha

print(f"\n{'#':<5} {'E (eV)':<12} {'λ (nm)':<12} {'f':<12} {'Симетрія':<15}")
print("-" * 80)

for i, ex_energy in enumerate(excitations):
    ex_eV = ex_energy * 27.211
    wavelength = 1240 / ex_eV  # nm

    # Сила осцилятора (потребує додаткового розрахунку)
    # Для спрощення показуємо тільки енергію

    print(f"{i+1:<5} {ex_eV:<12.4f} {wavelength:<12.1f} {'---':<12} {'?':<15}")

print("\n" + "=" * 80)
print("Експериментальні значення для N2:")
print("  a¹Πg: 8.59 eV (144 nm) - заборонений")
print("  b¹Πu: 12.85 eV (96 nm)")
print("  c¹Πu: 12.93 eV (96 nm)")

print("\nПримітка:")
print("- TD-DFT добре для валентних збуджень")
print("- Рідбергові стани потребують дифузних функцій")
print("- Для точних результатів: EOM-CCSD або CASPT2")

