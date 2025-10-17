# ============================================================
# h2o_raman_activity.py
# Розрахунок Раман-активності (потребує похідних поляризовності)
# ============================================================

from pyscf import gto, scf, lib
from pyscf.hessian import thermo
from pyscf.prop import polarizability
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

# SCF розрахунок
mf = scf.RHF(mol)
mf.kernel()

# Обчислюємо статичну поляризовність
alpha = polarizability.rhf.Polarizability(mf).polarizability()

print("\nСтатична поляризовність (au³):")
print("αₓₓ = {:.4f}".format(alpha[0, 0]))
print("αᵧᵧ = {:.4f}".format(alpha[1, 1]))
print("αᵤᵤ = {:.4f}".format(alpha[2, 2]))

# Середня поляризовність
alpha_mean = np.trace(alpha) / 3
print(f"\nСередня поляризовність: {alpha_mean:.4f} au³")

# Для повного Раман-спектру потрібні похідні поляризовності
# по нормальних координатах (складніший розрахунок)
print("\nПримітка:")
print("Повний Раман-спектр потребує обчислення ∂α/∂Q")
print("Це вимагає чисельного диференціювання або CPHF")


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

