# ============================================================
# h2o_ir_spectrum.py
# Розрахунок ІЧ-спектру (частоти + інтенсивності)
# ============================================================

from pyscf import gto, scf
from pyscf.hessian import rhf as rhf_hess
from pyscf.prop import infrared
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

print("Розрахунок ІЧ-спектру H2O")
print("=" * 60)

# SCF розрахунок
mf = scf.RHF(mol)
mf.kernel()

# Обчислюємо похідні дипольного моменту
# та частоти одночасно
ir_data = infrared.rhf.Infrared(mf)
ir_data.kernel()

print("\nІЧ-активні моди:")
print("-" * 70)
print(f"{'Мода':<8} {'ν (см⁻¹)':<15} {'Інтенсивність (км/моль)':<25}")
print("-" * 70)

for i, (freq, intensity) in enumerate(zip(ir_data.freq, ir_data.ir_inten)):
    if freq > 100:  # Тільки справжні коливання
        print(f"{i+1:<8} {freq:>12.1f}    {intensity:>20.2f}")

print("\nПримітка:")
print("- Інтенсивність залежить від зміни дипольного моменту")
print("- Деформаційна мода (ножиці) найінтенсивніша для H2O")


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

