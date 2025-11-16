# ============================================================
# franck_condon_N2.py
# Електронно-вібраційні переходи N₂: CASSCF потенціали + принцип Франка–Кондона
# Обчислення вимагає багато часу, тому краще його проводити в google colab
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, mcscf
from scipy.linalg import eigh_tridiagonal

# ------------------------------------------------------------
# Константи
# ------------------------------------------------------------
ANGSTROM_TO_BOHR = 1.8897259886
AMU_TO_AU = 1822.888486
M_H_AMU = 1.00784
MU_AU = (M_H_AMU / 2.0) * AMU_TO_AU  # Редуктована маса H₂ в атомних одиницях

# ------------------------------------------------------------
# 1. Розрахунок потенціальних кривих (PySCF)
# ------------------------------------------------------------
Rvals = np.linspace(0.5, 3.0, 40)  # Å
E_gs, E_es = [], []

nelec = 10
ncas = 10

print(f"Обчислення CASSCF({ncas}, {nelec}) потенціальних кривих...")

for R in Rvals:
    mol = gto.Mole()
    mol.atom = f"N 0 0 0; N 0 0 {R}"
    mol.basis = "cc-pvdz"
    mol.unit = "Angstrom"
    mol.spin = 0
    mol.build()

    mf = scf.RHF(mol).run(verbose=0)
    mc = mcscf.CASSCF(mf, ncas=10, nelecas=10)
    mc.state_average_([0.5, 0.5])
    mc.verbose = 0
    mc.kernel()

    E_gs.append(mc.e_states[0].real)
    E_es.append(mc.e_states[1].real)

E_gs = np.array(E_gs)
E_es = np.array(E_es)

# ------------------------------------------------------------
# 2. Квадратична апроксимація біля мінімумів
# ------------------------------------------------------------
R_bohr = Rvals * ANGSTROM_TO_BOHR

def fit_quadratic(Rb, E):
    """Повертає R_eq (bohr), E_min (Ha), k (Ha/bohr²)"""
    idx_min = np.argmin(E)
    left = max(0, idx_min - 3)
    right = min(len(Rb), idx_min + 4)
    p = np.polyfit(Rb[left:right], E[left:right], 2)
    a, b, c = p
    R_eq = -b / (2 * a)
    E_min = a * R_eq**2 + b * R_eq + c
    k = 2 * a
    return R_eq, E_min, k

R_eq_g_b, E_min_g, k_g = fit_quadratic(R_bohr, E_gs)
R_eq_e_b, E_min_e, k_e = fit_quadratic(R_bohr, E_es)

R_eq_g = R_eq_g_b / ANGSTROM_TO_BOHR
R_eq_e = R_eq_e_b / ANGSTROM_TO_BOHR

print(f"Основний стан: R_eq = {R_eq_g:.3f} Å, E_min = {E_min_g:.4f} Ha, k = {k_g:.4f} Ha/bohr²")
print(f"Збуджений стан: R_eq = {R_eq_e:.3f} Å, E_min = {E_min_e:.4f} Ha, k = {k_e:.4f} Ha/bohr²")

# ------------------------------------------------------------
# 3. Побудова гармонічної моделі для принципу Франка–Кондона
# ------------------------------------------------------------
# Сітка в Å для plot, але в bohr для обчислень
R_angs_plot = np.linspace(0.3, 2.3, 800)
R_bohr_plot = R_angs_plot * ANGSTROM_TO_BOHR

Vg = E_min_g + 0.5 * k_g * (R_bohr_plot - R_eq_g_b)**2
Ve = E_min_e + 0.5 * k_e * (R_bohr_plot - R_eq_e_b)**2

# --- Вібраційні рівні (чисельно) ---
dR = R_bohr_plot[1] - R_bohr_plot[0]  # в bohr

# Кінетична частина: T = -1/(2μ) d²/dR²
# Finite difference: off-diag = -1/(2 μ dR²), main kinetic = 1/(μ dR²)
offdiag = -1.0 / (2.0 * MU_AU * dR**2)
main_kin = 1.0 / (MU_AU * dR**2)

main_diag_g = main_kin + 0.5 * k_g * (R_bohr_plot - R_eq_g_b)**2
main_diag_e = main_kin + 0.5 * k_e * (R_bohr_plot - R_eq_e_b)**2

sub_diag = np.full(len(R_bohr_plot) - 1, offdiag)

evals_g, _ = eigh_tridiagonal(main_diag_g, sub_diag)
evals_e, _ = eigh_tridiagonal(main_diag_e, sub_diag)

E_vib_g = E_min_g + evals_g[:4]
E_vib_e = E_min_e + evals_e[:4]

# ------------------------------------------------------------
# 4. Побудова графіків
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
ax1, ax2 = axes

# --- (A) Потенціальні криві з PySCF ---
ax1.plot(Rvals, E_gs, 'b-', lw=2, label='Основний стан $X^1\\Sigma_g^+$')
ax1.plot(Rvals, E_es, 'r-', lw=2, label='Збуджений стан $B^1\\Sigma_u^+$')
ax1.set_xlabel('Відстань H–H, Å')
ax1.set_ylabel('Енергія, Hartree')
ax1.set_title(f'Потенціальні криві (CASSCF({ncas}, {nelec}))')
ax1.legend()
ax1.grid(True)

# --- (B) Принцип Франка–Кондона ---
ax2.plot(R_angs_plot, Vg, 'b', label='Основний стан')
ax2.plot(R_angs_plot, Ve, 'r', label='Збуджений стан')
ax2.hlines(E_vib_g, R_angs_plot.min(), R_angs_plot.max(), colors='b', linestyles='--')
ax2.hlines(E_vib_e, R_angs_plot.min(), R_angs_plot.max(), colors='r', linestyles='--')

# Вертикальні переходи
R0 = R_eq_g
E_start = np.interp(R0 * ANGSTROM_TO_BOHR, R_bohr_plot, Vg)
E_end = np.interp(R0 * ANGSTROM_TO_BOHR, R_bohr_plot, Ve)
ax2.plot([R0, R0], [E_start, E_end], 'k-', lw=2)
ax2.text(R0 + 0.02, (E_start + E_end) / 2, 'FC перехід', rotation=90, va='center')
ax2.set_ylim(-109,2, -108.7)
ax2.set_xlabel('R (Å)')
ax2.set_ylabel('E (Hartree)')
ax2.set_title('Ілюстрація принципу Франка–Кондона')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

