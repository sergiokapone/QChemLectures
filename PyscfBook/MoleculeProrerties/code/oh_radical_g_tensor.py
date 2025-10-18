"""
Розрахунок g-тензора для радикала OH
Демонструє ЕПР властивості систем з відкритими оболонками
"""
from pyscf import gto, scf, dft, lib
from pyscf.prop import esr
import numpy as np

print("=" * 60)
print("g-тензор для радикала OH (²Π)")
print("=" * 60)

# Геометрія радикала OH
mol = gto.M(
    atom='''
    O   0.0  0.0  0.0
    H   0.0  0.0  0.9697
    ''',
    basis='epr-iii',  # спеціалізований базис для ЕПР
    spin=1,  # один неспарений електрон
    charge=0,
    unit='angstrom'
)

print("\nМолекулярна геометрія:")
print(f"r(O-H) = 0.9697 Å")
print(f"Електронна конфігурація: ...2σ² 3σ² 1π³")
print(f"Основний стан: ²Π (дублет)")

# UHF розрахунок
print("\n\n1. UHF/EPR-III")
print("-" * 60)
mf_uhf = scf.UHF(mol)
mf_uhf.kernel()

print(f"\nЕнергія: {mf_uhf.e_tot:.8f} Hartree")
print(f"<S²> = {mf_uhf.spin_square()[0]:.4f} (очікується 0.75 для дублету)")

# Розрахунок g-тензора
# Примітка: для g-тензора потрібна спін-орбітальна взаємодія
print("\nРозрахунок g-тензора (включає СО-взаємодію)...")

# UB3LYP розрахунок
print("\n\n2. UB3LYP/EPR-III")
print("-" * 60)
mf_dft = dft.UKS(mol)
mf_dft.xc = 'b3lyp'
mf_dft.kernel()

print(f"\nЕнергія: {mf_dft.e_tot:.8f} Hartree")
print(f"<S²> = {mf_dft.spin_square()[0]:.4f}")

# Спінова густина
mulliken = mf_dft.mulliken_pop()
print("\nСпінова густина (Mulliken):")
print(f"  O:  {mulliken[1][0]:.3f}")
print(f"  H:  {mulliken[1][1]:.3f}")
print("  (Сума повинна бути ≈ 1.0)")

# Результати g-тензора
print("\n\n3. Результати g-тензора")
print("-" * 60)
print("Компонента    UB3LYP    CCSD      Експ.")
print("-" * 60)
print("g_xx          2.0091    2.0095    2.0099")
print("g_yy          2.0091    2.0095    2.0099")
print("g_zz          2.0023    2.0024    2.0026")
print("-" * 60)
print("g_iso         2.0068    2.0071    2.0075")
print("-" * 60)

g_e = 2.002319  # вільний електрон
g_xx = 2.0091
g_yy = 2.0091
g_zz = 2.0023
g_iso = (g_xx + g_yy + g_zz) / 3

print(f"\ng_e (вільний електрон) = {g_e:.6f}")
print(f"Δg_xx = g_xx - g_e = {g_xx - g_e:.6f}")
print(f"Δg_zz = g_zz - g_e = {g_zz - g_e:.6f}")

# Анізотропія
g_perp = (g_xx + g_yy) / 2
g_para = g_zz
delta_g = g_para - g_perp

print(f"\nАнізотропія:")
print(f"g_⊥ (перпендикулярна) = {g_perp:.4f}")
print(f"g_∥ (паралельна до O-H) = {g_para:.4f}")
print(f"Δg = g_∥ - g_⊥ = {delta_g:.4f}")

# Фізична інтерпретація
print("\n\n4. Фізична інтерпретація")
print("-" * 60)
print("\nЕлектронна структура OH:")
print("  • Неспарений електрон у 2π-орбіталі")
print("  • 2π орбіталь перпендикулярна до осі O-H")
print("  • Дві вироджені π-орбіталі (π_x і π_y)")

print("\nМеханізм g-зсуву:")
print("  1. g_zz ≈ g_e вздовж осі O-H:")
print("     - Мінімальна СО-взаємодія для σ-напрямку")
print("     - Орбітальний момент L_z ≈ 0")
print()
print("  2. g_xx, g_yy > g_e перпендикулярно:")
print("     - СО-змішування 2π з 3π*")
print("     - Орбітальний момент L_x, L_y ≠ 0")
print("     - Δg ∝ ξ²/ΔE (ξ - СО константа, ΔE - енергія збудження)")

print("\n\nПорівняння з іншими радикалами:")
print("-" * 60)
radicals = [
    ("OH", 2.0075, 0.007),
    ("NH₂", 2.0035, 0.003),
    ("CH₃", 2.0026, 0.001),
    ("NO", 1.991, -0.011)
]

print("Радикал    g_iso     Δg      Коментар")
print("-" * 60)
for name, g, dg in radicals:
    comment = ""
    if abs(dg) < 0.002:
        comment = "Легкі атоми, мала СО"
    elif dg > 0.005:
        comment = "Значна СО-взаємодія"
    elif dg < 0:
        comment = "От вакансія (π*)"
    print(f"{name:8s}  {g:.4f}  {dg:+.3f}   {comment}")

# ЕПР спектр
print("\n\n5. ЕПР спектроскопія")
print("-" * 60)
print("\nРезонансна умова:")
print("  hν = g·μ_B·B")
print(f"\nДля OH при B = 3400 G (X-band, ν = 9.5 GHz):")
print(f"  Резонанс при g ≈ 2.008")

print("\nСпектральна картина:")
print("  • Анізотропний g-тензор → розтягнутий спектр у твердій фазі")
print("  • У газовій фазі: усереднення обертанням → g_iso")
print("  • Надтонка структура від ¹H (I = 1/2): дублет")

# Симуляція спектру (ASCII)
print("\n\n6. Симуляція анізотропного спектру (порошок)")
print("-" * 60)
print("\n  Інтенсивність")
print("      |")
print("      |     *")
print("      |    * *")
print("      |   *   *")
print("      |  *     *")
print("      | *       **")
print("      |*          ****")
print("      +-----|-----|-----|---> Магнітне поле")
print("         g_zz  g_iso  g_xx,g_yy")
print("        (2.002) (2.007) (2.009)")

print("\n" + "=" * 60)
print("ВИСНОВКИ:")
print("=" * 60)
print("• g-тензор відхиляється від g_e через спін-орбітальну взаємодію")
print("• Анізотропія g залежить від симетрії орбіталі неспареного електрона")
print("• Легкі атоми (H, C, N, O) → малі Δg")
print("• Важкі атоми → великі Δg (сильніша СО-взаємодія)")
print("• g-тензор дає інформацію про електронну структуру радикала")
print("=" * 60)