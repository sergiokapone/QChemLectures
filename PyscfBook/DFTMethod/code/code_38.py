# ============================================================
# code_38.py - Спектроскопічні константи
# ============================================================
from pyscf import gto, dft
import numpy as np
from scipy.optimize import curve_fit
from scipy.misc import derivative

def morse_potential(r, D_e, r_e, a):
    """Потенціал Морзе"""
    return D_e * (1 - np.exp(-a * (r - r_e)))**2

def harmonic_frequency(r_values, energies, r_e):
    """Частота коливань з другої похідної"""
    # Чисельна друга похідна в точці r_e
    f = lambda r: np.interp(r, r_values, energies)
    k = derivative(f, r_e, n=2, dx=0.001)  # Силова константа

    # ω = sqrt(k/μ) де μ - зведена маса
    # Для H2: μ = m_H/2
    mass_H = 1.008 * 1822.888  # в а.о.
    mu = mass_H / 2
    omega = np.sqrt(k / mu)

    # Переведення в см⁻¹
    omega_cm = omega * 219474.63  # Hartree to cm⁻¹
    return omega_cm

print("Спектроскопічні константи H2 (PBE0/aug-cc-pVQZ)")
print("=" * 70)

# Детальне сканування навколо мінімуму
r_values = np.linspace(0.5, 2.5, 50)
energies = []

for r in r_values:
    mol = gto.M(
        atom=f'H 0 0 0; H 0 0 {r}',
        basis='aug-cc-pvqz',
        unit='angstrom'
    )

    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.verbose = 0
    energy = mf.kernel()
    energies.append(energy)

energies = np.array(energies)

# Знаходження мінімуму
min_idx = np.argmin(energies)
r_e = r_values[min_idx]
E_min = energies[min_idx]

# Енергія дисоціації
E_inf = energies[-1]
D_e = (E_inf - E_min) * 27.211  # eV

# Апроксимація потенціалом Морзе
try:
    # Початкові значення
    a_init = 1.0
    popt, _ = curve_fit(
        morse_potential,
        r_values,
        energies - E_inf,
        p0=[E_min - E_inf, r_e, a_init]
    )
    D_fit, r_fit, a_fit = popt

    print(f"\nПараметри потенціалу Морзе:")
    print(f"  D_e = {abs(D_fit)*27.211:.4f} eV")
    print(f"  r_e = {r_fit:.6f} Å")
    print(f"  a   = {a_fit:.4f} Å⁻¹")
except:
    print("\nНе вдалося апроксимувати потенціалом Морзе")

# Частота коливань
omega_e = harmonic_frequency(r_values, energies, r_e)

print(f"\nОбчислені спектроскопічні константи:")
print(f"  r_e    = {r_e:.6f} Å")
print(f"  D_e    = {D_e:.4f} eV")
print(f"  ω_e    = {omega_e:.1f} cm⁻¹")

print(f"\nЕкспериментальні значення:")
print(f"  r_e    = 0.741300 Å")
print(f"  D_e    = 4.7500 eV")
print(f"  ω_e    = 4401.2 cm⁻¹")

print(f"\nВідхилення:")
print(f"  Δr_e   = {(r_e - 0.7413)*100:.3f} %")
print(f"  ΔD_e   = {(D_e - 4.75)/4.75*100:.3f} %")
print(f"  Δω_e   = {(omega_e - 4401.2)/4401.2*100:.3f} %")

