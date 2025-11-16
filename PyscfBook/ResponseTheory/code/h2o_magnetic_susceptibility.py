# ============================================================
# h2o_magnetic_susceptibility.py
# Розрахунок магнітної сприйнятливості
# ============================================================

from pyscf import gto, scf
from pyscf.prop import magnetizability
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

print("Магнітна сприйнятливість H2O")
print("=" * 60)

# SCF розрахунок
mf = scf.RHF(mol)
mf.verbose = 0
mf.kernel()

# Магнітна сприйнятливість (діамагнітний внесок)
print("\nОбчислення магнітної сприйнятливості...")
mag = magnetizability.rhf.Magnetizability(mf)
chi = mag.magnetizability()

print("\nТензор магнітної сприйнятливості χ (ppm·cgs):")
print("       x          y          z")
for i, label in enumerate(['x', 'y', 'z']):
    print(f"{label}  ", end="")
    for j in range(3):
        print(f"{chi[i,j]:>10.4f} ", end="")
    print()

# Середнє значення (ізотропна сприйнятливість)
chi_iso = np.trace(chi) / 3
print(f"\nІзотропна сприйнятливість:")
print(f"  χ̄ = {chi_iso:.4f} ppm·cgs")
print(f"     = {chi_iso * 1.5958:.4f} × 10⁻⁶ cm³/mol")

# Анізотропія
chi_aniso = chi[2,2] - (chi[0,0] + chi[1,1]) / 2
print(f"\nАнізотропія:")
print(f"  Δχ = χ_∥ - χ_⊥ = {chi_aniso:.4f} ppm·cgs")

print("\nПорівняння з експериментом:")
print("  Експериментальне χ̄ ≈ -13.0 × 10⁻⁶ cm³/mol")
print(f"  Розрахунок:        {chi_iso * 1.5958:.1f} × 10⁻⁶ cm³/mol")

print("\nПримітка:")
print("- Негативне значення → діамагнетик")
print("- H2O відштовхується від магнітного поля")
print("- Для парамагнетиків (непарені електрони) χ > 0")

