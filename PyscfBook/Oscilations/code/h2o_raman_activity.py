# ============================================================
# h2o_raman_activity.py
# ============================================================
from pyscf import gto, scf
from pyscf.prop.polarizability import rhf as pol_rhf
import numpy as np

mol = gto.M(
    atom="""
    O  0.0000  0.0000  0.1173
    H  0.0000  0.7572 -0.4692
    H  0.0000 -0.7572 -0.4692
    """,
    basis="6-31g",
    unit="angstrom",
)

print("Розрахунок Раман-активності H2O")
print("=" * 60)

# SCF
mf = scf.RHF(mol)
mf.kernel()

# Поляризовність
polar = pol_rhf.Polarizability(mf)
alpha = polar.polarizability()

print("\nСтатична поляризовність (au³):")
print(f"αₓₓ = {alpha[0, 0]:.4f}")
print(f"αᵧᵧ = {alpha[1, 1]:.4f}")
print(f"αᵤᵤ = {alpha[2, 2]:.4f}")

alpha_mean = np.trace(alpha) / 3
print(f"\nСередня поляризовність: {alpha_mean:.4f} au³")
print(f"Середня поляризовність: {alpha_mean * 0.1482:.4f} Å³")

print("\nРаман-активні моди (якісно):")
print("- Симетричне розтягування OH: сильна")
print("- Деформація HOH: середня")
print("- Асиметричне розтягування OH: слабка")
