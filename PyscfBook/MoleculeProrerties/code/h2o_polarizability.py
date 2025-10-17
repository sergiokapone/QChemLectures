# ============================================================
# h2o_polarizability.py
# Розрахунок статичної поляризовності
# ============================================================

from pyscf import gto, scf, dft
from pyscf.prop import polarizability
import numpy as np

mol = gto.M(
    atom="""
    O  0.0000  0.0000  0.1173
    H  0.0000  0.7572 -0.4692
    H  0.0000 -0.7572 -0.4692
    """,
    basis="aug-cc-pvdz",
    unit="angstrom",
)

print("Розрахунок поляризовності H2O")
print("=" * 60)

# RHF розрахунок
print("\n1. RHF/aug-cc-pVDZ:")
mf_rhf = scf.RHF(mol)
mf_rhf.verbose = 0
mf_rhf.kernel()

# Статична поляризовність (ω=0)
alpha_rhf = polarizability.rhf.Polarizability(mf_rhf).polarizability()

print("\nТензор поляризовності α (au³):")
print(f"  αₓₓ = {alpha_rhf[0,0]:.4f}")
print(f"  αᵧᵧ = {alpha_rhf[1,1]:.4f}")
print(f"  αᵤᵤ = {alpha_rhf[2,2]:.4f}")

# Середня поляризовність
alpha_mean_rhf = np.trace(alpha_rhf) / 3
print(f"\nСередня поляризовність:")
print(f"  ᾱ = (αₓₓ + αᵧᵧ + αᵤᵤ)/3 = {alpha_mean_rhf:.4f} au³")

# Анізотропія
alpha_aniso_rhf = np.sqrt(0.5 * (
    (alpha_rhf[0,0] - alpha_rhf[1,1])**2 +
    (alpha_rhf[1,1] - alpha_rhf[2,2])**2 +
    (alpha_rhf[2,2] - alpha_rhf[0,0])**2
))
print(f"  Δα = {alpha_aniso_rhf:.4f} au³")

# B3LYP розрахунок
print("\n2. B3LYP/aug-cc-pVDZ:")
mf_dft = dft.RKS(mol)
mf_dft.xc = "b3lyp"
mf_dft.verbose = 0
mf_dft.kernel()

alpha_dft = polarizability.rks.Polarizability(mf_dft).polarizability()
alpha_mean_dft = np.trace(alpha_dft) / 3

print(f"  ᾱ = {alpha_mean_dft:.4f} au³")

print("\n" + "=" * 60)
print("Порівняння з експериментом (ᾱ):")
print(f"  RHF:        {alpha_mean_rhf:.2f} au³")
print(f"  B3LYP:      {alpha_mean_dft:.2f} au³")
print(f"  Експеримент: 9.56 au³")

print("\nКонвертація одиниць:")
print(f"  1 au³ = 0.1482 Ų")
print(f"  ᾱ(B3LYP) = {alpha_mean_dft * 0.1482:.3f} Ų")

