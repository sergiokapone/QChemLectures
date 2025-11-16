# ============================================================
# h2o_dipole.py
# Розрахунок дипольного моменту молекули
# ============================================================

from pyscf import gto, scf, dft
import numpy as np

# Молекула води
mol = gto.M(
    atom="""
    O  0.0000  0.0000  0.1173
    H  0.0000  0.7572 -0.4692
    H  0.0000 -0.7572 -0.4692
    """,
    basis="aug-cc-pvdz",
    unit="angstrom",
)

print("Розрахунок дипольного моменту H2O")
print("=" * 60)

# RHF розрахунок
print("\n1. RHF/aug-cc-pVDZ:")
mf_rhf = scf.RHF(mol)
mf_rhf.verbose = 0
mf_rhf.kernel()

# Дипольний момент в атомних одиницях
dip_rhf = mf_rhf.dip_moment(unit='AU')
dip_rhf_magnitude = np.linalg.norm(dip_rhf)

# Конвертуємо в Дебаї (1 au = 2.5418 D)
dip_rhf_debye = dip_rhf * 2.5418
dip_rhf_mag_debye = dip_rhf_magnitude * 2.5418

print(f"  μ_x = {dip_rhf_debye[0]:>8.4f} D")
print(f"  μ_y = {dip_rhf_debye[1]:>8.4f} D")
print(f"  μ_z = {dip_rhf_debye[2]:>8.4f} D")
print(f"  |μ| = {dip_rhf_mag_debye:>8.4f} D")

# B3LYP розрахунок
print("\n2. B3LYP/aug-cc-pVDZ:")
mf_dft = dft.RKS(mol)
mf_dft.xc = "b3lyp"
mf_dft.verbose = 0
mf_dft.kernel()

dip_dft = mf_dft.dip_moment(unit='Debye')
dip_dft_magnitude = np.linalg.norm(dip_dft)

print(f"  μ_x = {dip_dft[0]:>8.4f} D")
print(f"  μ_y = {dip_dft[1]:>8.4f} D")
print(f"  μ_z = {dip_dft[2]:>8.4f} D")
print(f"  |μ| = {dip_dft_magnitude:>8.4f} D")

print("\n" + "=" * 60)
print("Порівняння з експериментом:")
print("  Експериментальне значення: 1.855 D")
print(f"  RHF похибка:  {dip_rhf_mag_debye - 1.855:+.3f} D ({(dip_rhf_mag_debye/1.855-1)*100:+.1f}%)")
print(f"  B3LYP похибка: {dip_dft_magnitude - 1.855:+.3f} D ({(dip_dft_magnitude/1.855-1)*100:+.1f}%)")

print("\nПримітка:")
print("- RHF завищує дипольний момент")
print("- Дифузні функції (aug-) важливі для точності")
print("- DFT зазвичай ближче до експерименту")
