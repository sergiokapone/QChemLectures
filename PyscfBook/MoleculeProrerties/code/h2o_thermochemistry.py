# ============================================================
# h2o_thermochemistry.py
# Термохімічні поправки (ZPE, ентальпія, ентропія)
# ============================================================

ffrom pyscf import gto, scf
from pyscf.hessian import thermo
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

print("Термохімічний аналіз H2O")
print("=" * 60)

# SCF розрахунок
mf = scf.RHF(mol)
e_elec = mf.kernel()

# Гессіан
hess = mf.Hessian()
h = hess.kernel()

# Обчислення вібраційних частот (у cm⁻¹)
freq_info = thermo.harmonic_analysis(mol, h)
freqs_cm = freq_info['freq_wavenumber']

# Термохімічні функції при T=298.15 K, p=1 atm
# thermo.thermo приймає freq у cm⁻¹ і автоматично конвертує
results = thermo.thermo(mf, freqs_cm, 298.15, 101325)

print("\nЕлектронна енергія:")
print(f"E(elec) = {e_elec:.6f} Ha")
print(f"        = {e_elec * 627.509:.2f} ккал/моль")

print("\nПоправки при 298.15 K:")
zpe_ha = results['ZPE'][0]
print(f"Нульова коливальна енергія (ZPE): {zpe_ha:.6f} Ha")
e_thermal_ha = results['E_tot'][0] - (e_elec + zpe_ha)  # Термічна поправка до енергії (без ZPE)
print(f"Термічна поправка до енергії:     {e_thermal_ha:.6f} Ha")
h_thermal_ha = results['H_tot'][0] - (e_elec + zpe_ha)  # Термічна поправка до ентальпії (приблизна)
print(f"Термічна поправка до ентальпії:   {h_thermal_ha:.6f} Ha")

print("\nПовна енергія Гіббса:")
print(f"G(298K) = E(elec) + ZPE + H_thermal - T*S")
g_ha = results['G_tot'][0]
print(f"        = {g_ha:.6f} Ha")

print("\nЕнтропія:")
s = results['S_tot'][0]
print(f"S = {s:.3f} кал/(моль·K)")

print("\nРозклад ZPE по модах (лише >100 cm⁻¹):")
conversion_factor = 0.0014295  # Правильний: 0.5 * h c N_A / (4.184 * 1000) в ккал/моль на cm⁻¹
for i, freq in enumerate(freqs_cm):
    if freq.imag == 0 and freq.real > 100:  # Ігноруємо уявні частоти та низькочастотні
        zpe_mode_kcal = 0.5 * freq.real * conversion_factor * 2  # Помилка в оригіналі виправлена; *2? Ні, це 0.5 * factor * freq
        # Правильно: zpe_mode = (1/2) * nu * factor, де factor = 0.002859 для повної енергії, тож 0.0014295 для ZPE
        zpe_mode_kcal = 0.0014295 * freq.real
        print(f"  Мода {i+1}: {freq.real:.1f} cm⁻¹ → ZPE = {zpe_mode_kcal:.2f} ккал/моль")
        
