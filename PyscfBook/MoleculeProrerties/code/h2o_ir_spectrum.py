# ============================================================
# h2o_ir_spectrum.py
# Розрахунок ІЧ-спектру (частоти + інтенсивності)
# ============================================================
from pyscf import gto, scf
from pyscf.hessian import rhf as rhf_hess
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

# Обчислюємо Гессіан (матрицю других похідних)
hess = mf.Hessian().kernel()

# Аналіз нормальних мод
from pyscf.hessian import thermo
freq_info = thermo.harmonic_analysis(mol, hess)

print("\nІЧ-активні моди:")
print("-" * 70)
print(f"{'Мода':<8} {'ν (см⁻¹)':<15}")
print("-" * 70)

for i, freq in enumerate(freq_info['freq_wavenumber']):
    if freq > 100:  # Тільки справжні коливання
        print(f"{i+1:<8} {freq:>12.1f}")

print("\nПримітка:")
print("- Для інтенсивностей потрібні похідні дипольного моменту")
print("- Використовуйте numerical differentiation або analytical градієнти")

