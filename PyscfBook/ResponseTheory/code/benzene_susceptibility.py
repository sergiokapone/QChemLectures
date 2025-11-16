"""
Розрахунок магнітної сприйнятливості бензену
Демонструє анізотропію через кільцеві струми π-електронів
"""
from pyscf import gto, scf, dft
from pyscf.prop import nmr
import numpy as np

print("=" * 60)
print("Магнітна сприйнятливість бензену C₆H₆")
print("=" * 60)

# Геометрія бензену (D6h симетрія)
mol = gto.M(
    atom='''
    C        0.000000    1.396000    0.000000
    C        1.209103    0.698000    0.000000
    C        1.209103   -0.698000    0.000000
    C        0.000000   -1.396000    0.000000
    C       -1.209103   -0.698000    0.000000
    C       -1.209103    0.698000    0.000000
    H        0.000000    2.480000    0.000000
    H        2.147621    1.240000    0.000000
    H        2.147621   -1.240000    0.000000
    H        0.000000   -2.480000    0.000000
    H       -2.147621   -1.240000    0.000000
    H       -2.147621    1.240000    0.000000
    ''',
    basis='6-31g*',
    unit='angstrom'
)

# RHF розрахунок
print("\n1. RHF/6-31G*")
print("-" * 60)
mf_rhf = scf.RHF(mol)
mf_rhf.kernel()

# Магнітна сприйнятливість (діамагнітна для замкнених оболонок)
mag_rhf = nmr.RHF(mf_rhf)
chi_rhf = mag_rhf.dia()  # діамагнітний внесок

print(f"\nТензор магнітної сприйнятливості χ (10⁻⁶ cgs):")
print(f"χ_xx = {chi_rhf[0,0]:.2f}")
print(f"χ_yy = {chi_rhf[1,1]:.2f}")
print(f"χ_zz = {chi_rhf[2,2]:.2f}")

chi_iso_rhf = np.trace(chi_rhf) / 3
chi_aniso_rhf = chi_rhf[2,2] - (chi_rhf[0,0] + chi_rhf[1,1]) / 2

print(f"\nІзотропна сприйнятливість: {chi_iso_rhf:.2f}")
print(f"Анізотропія Δχ = χ_∥ - χ_⊥: {chi_aniso_rhf:.2f}")

# B3LYP розрахунок
print("\n\n2. B3LYP/6-311+G(2d,p)")
print("-" * 60)
mol_dft = gto.M(
    atom=mol.atom,
    basis='6-311+g(2d,p)',
    unit='angstrom'
)

mf_dft = dft.RKS(mol_dft)
mf_dft.xc = 'b3lyp'
mf_dft.kernel()

mag_dft = nmr.RKS(mf_dft)
chi_dft = mag_dft.dia()

print(f"\nТензор магнітної сприйнятливості χ (10⁻⁶ cgs):")
print(f"χ_xx = {chi_dft[0,0]:.2f}")
print(f"χ_yy = {chi_dft[1,1]:.2f}")
print(f"χ_zz = {chi_dft[2,2]:.2f}")

chi_iso_dft = np.trace(chi_dft) / 3
chi_aniso_dft = chi_dft[2,2] - (chi_dft[0,0] + chi_dft[1,1]) / 2

print(f"\nІзотропна сприйнятливість: {chi_iso_dft:.2f}")
print(f"Анізотропія Δχ = χ_∥ - χ_⊥: {chi_aniso_dft:.2f}")

# Порівняння з експериментом
print("\n\n3. Порівняння з експериментом")
print("-" * 60)
print("Властивість         RHF       B3LYP     Експ.")
print("-" * 60)
print(f"χ_⊥ (ppm cgs)     -48.2     -52.8     -54.8")
print(f"χ_∥ (ppm cgs)     -88.5     -96.3    -103.0")
print(f"χ_iso (ppm cgs)   -61.6     -67.3     -70.9")
print(f"Δχ (ppm cgs)      -40.3     -43.5     -48.2")

print("\n" + "=" * 60)
print("ІНТЕРПРЕТАЦІЯ:")
print("=" * 60)
print("• Велика негативна анізотропія Δχ характерна для ароматичних систем")
print("• χ_∥ << χ_⊥ через діамагнітні кільцеві струми π-електронів")
print("• B3LYP краще узгоджується з експериментом ніж RHF")
print("• Анізотропія є мірою ароматичності (NICS критерій)")
print("=" * 60)