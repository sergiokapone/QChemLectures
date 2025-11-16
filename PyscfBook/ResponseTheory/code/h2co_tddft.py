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


