# ============================================================
# h2o_frequencies.py
# Розрахунок частот коливань для H2O
# ============================================================

from pyscf import gto, scf
from pyscf.hessian import thermo
import numpy as np

# Молекула води (оптимізована геометрія)
mol = gto.M(
    atom="""
    O  0.0000  0.0000  0.1173
    H  0.0000  0.7572 -0.4692
    H  0.0000 -0.7572 -0.4692
    """,
    basis="6-31g",
    unit="angstrom",
)

print("Розрахунок коливальних частот H2O")
print("=" * 60)

# SCF розрахунок
mf = scf.RHF(mol)
mf.kernel()

# Обчислюємо гесіан
hess = mf.Hessian()
h = hess.kernel()

# Аналіз частот
freq_info = thermo.harmonic_analysis(mol, h)

print("\nКоливальні частоти:")
print("-" * 60)
print(f"{'№':<5} {'Частота (см⁻¹)':<20} {'Тип'}")
print("-" * 60)

# Виведемо тільки справжні коливання (без трансляцій/обертань)
modes = freq_info['freq_wavenumber']
for i, freq in enumerate(modes):
    if freq > 100:  # Фільтруємо дуже малі частоти
        mode_type = "коливання"
        print(f"{i+1:<5} {freq:>15.1f}     {mode_type}")

print("\nПорівняння з експериментом:")
print("ν₁ (симетр. валент.):  ~3657 см⁻¹")
print("ν₂ (деформаційна):     ~1595 см⁻¹")
print("ν₃ (антисим. валент.): ~3756 см⁻¹")
print("\nПримітка: RHF/6-31G завищує частоти на ~10-15%")

