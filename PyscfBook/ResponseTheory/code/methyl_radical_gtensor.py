# ============================================================
# methyl_radical_gtensor.py
# g-тензор для метильного радикала
# ============================================================

from pyscf import gto, scf
from pyscf.prop.gtensor import uhf as gtensor_uhf
import numpy as np

# Метильний радикал •CH3 (плоска структура, ²A₂″)
mol = gto.M(
    atom="""
    C   0.0000   0.0000   0.0000
    H   1.0790   0.0000   0.0000
    H  -0.5395   0.9343   0.0000
    H  -0.5395  -0.9343   0.0000
    """,
    basis="6-311g**",
    spin=1,  # Один непарений електрон
    symmetry=False,
)

print("g-тензор для метильного радикала •CH₃")
print("=" * 60)

# UHF розрахунок для радикала
print("\nUHF розрахунок...")
mf = scf.UHF(mol)
mf.verbose = 0
mf.kernel()

# Перевірка спінового забруднення
s2 = mf.spin_square()[0]
s2_expected = 0.75  # S=1/2: S(S+1) = 0.75
print(f"\nСпіновий стан:")
print(f"  <S²> = {s2:.6f} (очікується 0.750)")
print(f"  Забруднення: {s2 - s2_expected:.6f}")

# Обчислюємо g-тензор
print("\nОбчислення g-тензора...")
gtensor = gtensor_uhf.GTensor(mf)
g = gtensor.kernel()

print("\ng-тензор:")
print("       x          y          z")
for i, label in enumerate(['x', 'y', 'z']):
    print(f"{label}  ", end="")
    for j in range(3):
        print(f"{g[i,j]:>10.6f} ", end="")
    print()

# Головні значення (діагоналізація)
g_vals = np.linalg.eigvalsh(g)
g_vals = np.sort(g_vals)

print(f"\nГоловні значення:")
print(f"  g_xx = {g_vals[0]:.6f}")
print(f"  g_yy = {g_vals[1]:.6f}")
print(f"  g_zz = {g_vals[2]:.6f}")

# Ізотропний g-фактор
g_iso = np.trace(g) / 3
print(f"\nІзотропний g-фактор:")
print(f"  g_iso = {g_iso:.6f}")

# Зсув від вільного електрона
g_e = 2.002319
delta_g = g_iso - g_e
print(f"\nЗсув від g_e:")
print(f"  Δg = {delta_g:.6f}")
print(f"     = {delta_g * 1e6:.1f} ppm")

print("\n" + "=" * 60)
print("Порівняння з експериментом:")
print(f"  Експериментальний g_iso ≈ 2.00249")
print(f"  Розрахунок UHF/6-311G**: {g_iso:.5f}")

print("\nПримітка:")
print("- Малий зсув g для легких атомів (слабка SOC)")
print("- Для важких атомів (S, P, метали) зсуви набагато більші")
print("- Анізотропія g видна в замороженому ЕПР спектрі")

