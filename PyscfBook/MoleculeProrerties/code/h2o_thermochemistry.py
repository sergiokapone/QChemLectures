# ============================================================
# h2o_thermochemistry.py
# Термохімічні поправки (ZPE, ентальпія, ентропія)
# ============================================================

from pyscf import gto, scf
from pyscf.hessian import thermo

mol = gto.M(
    atom="""
    O  0.0000  0.0000  0.1173
    H  0.0000  0.7572 -0.4692
    H  0.0000 -0.7572 -0.4692
    """,
    basis="6-31g",
    unit="angstrom",
)

print("Термохімічний аналіз H2O")
print("=" * 60)

# SCF розрахунок
mf = scf.RHF(mol)
e_elec = mf.kernel()

# Гесіан та частоти
hess = mf.Hessian()
h = hess.kernel()

# Термохімічні функції при T=298.15 K, p=1 atm
results = thermo.thermo(mf, h, 298.15, 101325)

print("\nЕлектронна енергія:")
print(f"E(elec) = {e_elec:.6f} Ha")
print(f"        = {e_elec * 627.509:.2f} ккал/моль")

print("\nПоправки при 298.15 K:")
print(f"Нульова коливальна енергія (ZPE): {results['ZPE']:.6f} Ha")
print(f"Термічна поправка до енергії:     {results['E_thermal']:.6f} Ha")
print(f"Термічна поправка до ентальпії:   {results['H_thermal']:.6f} Ha")

print("\nПовна енергія Гіббса:")
print(f"G(298K) = E(elec) + ZPE + H_thermal - T*S")
print(f"        = {results['G_total']:.6f} Ha")

print("\nЕнтропія:")
print(f"S = {results['S']:.3f} кал/(моль·K)")

print("\nРозклад ZPE по модах:")
for i, freq in enumerate(results['freqs']):
    if freq > 100:
        zpe_mode = 0.5 * freq * 1.4388  # см⁻¹ -> ккал/моль
        print(f"  Мода {i+1}: {freq:.1f} см⁻¹ → ZPE = {zpe_mode:.2f} ккал/моль")


# ============================================================
# h2co_tddft.py
# Розрахунок УФ-спектру формальдегіду методом TDDFT
# ============================================================

from pyscf import gto, scf, dft, tddft

# Молекула формальдегіду H2CO
mol = gto.M(
    atom="""
    C  0.0000  0.0000  0.0000
    O  0.0000  0.0000  1.2050
    H  0.0000  0.9428 -0.5876
    H  0.0000 -0.9428 -0.5876
    """,
    basis="aug-cc-pvdz",
    unit="angstrom",
)

print("TDDFT розрахунок для H2CO (формальдегід)")
print("=" * 60)

# Основний стан: DFT з функціоналом CAM-B3LYP
mf = dft.RKS(mol)
mf.xc = "cam-b3lyp"  # Range-separated функціонал
mf.kernel()

print(f"\nЕнергія основного стану: {mf.e_tot:.6f} Ha")

# TDDFT для збуджених станів
# Розраховуємо перші 5 синглетних збуджень
td = tddft.TDDFT(mf)
td.nstates = 5
td.kernel()

print("\nЗбуджені стани (синглети):")
print("-" * 80)
print(f"{'Стан':<8} {'ΔE (eV)':<12} {'λ (нм)':<12} {'f':<12} {'Характер'}")
print("-" * 80)

# Конвертуємо Hartree -> eV -> nm
au2ev = 27.2114
for i, e in enumerate(td.e):
    energy_ev = e * au2ev
    wavelength = 1240 / energy_ev  # eV -> nm
    osc_str = td.oscillator_strength()[i]

    # Простий аналіз характеру
    if i == 0:
        char = "n→π*"
    elif i == 1:
        char = "π→π*"
    else:
        char = "Рідберг/змішаний"

    print(f"S_{i+1:<6} {energy_ev:>10.3f}  {wavelength:>10.1f}  "
          f"{osc_str:>10.4f}  {char}")

print("\nПримітка:")
print("- Сила осцилятора f показує інтенсивність переходу")
print("- n→π* перехід слабкий (заборонений за симетрією)")
print("- π→π* перехід сильний (дозволений)")

