# ============================================================
# franck_condon_H2_full.py
# Електронно-вібраційні переходи H₂: CASSCF потенціали + принцип Франка–Кондона
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, mcscf
from scipy.constants import physical_constants
from scipy.linalg import eigh_tridiagonal

# ------------------------------------------------------------
# 1. Розрахунок потенціальних кривих (PySCF)
# ------------------------------------------------------------
Rvals = np.linspace(0.5, 3.0, 40)  # Å
E_gs, E_es = [], []

for R in Rvals:
    mol = gto.Mole()
    mol.atom = f"H 0 0 0; H 0 0 {R}"
    mol.basis = "6-31g"
    mol.unit = "Angstrom"
    mol.spin = 0
    mol.build()

    mf = scf.RHF(mol).run(verbose=0)
    mc = mcscf.CASSCF(mf, ncas=2, nelecas=2)
    mc.state_average_([0.5, 0.5])
    mc.kernel()

    E_gs.append(mc.e_states[0])
    E_es.append(mc.e_states[1])

E_gs = np.array(E_gs)
E_es = np.array(E_es)

# ------------------------------------------------------------
# 2. Побудова гармонічної моделі для принципу Франка–Кондона
# ------------------------------------------------------------
amu_to_au = 1822.888486
m_H = 1.00784 * amu_to_au / 2  # редукована маса H₂ / 2

R_eq_g = 0.75
R_eq_e = 1.30
k_g = 0.5
k_e = 0.35
E_min_g = -1.14
E_min_e = -0.73

R = np.linspace(0.3, 2.3, 800)
Vg = E_min_g + 0.5 * k_g * (R - R_eq_g)**2
Ve = E_min_e + 0.5 * k_e * (R - R_eq_e)**2

# --- Вібраційні рівні (чисельно) ---
dR = R[1] - R[0]
kin = np.ones(len(R)-1) * (-1/(2*m_H*dR**2))
Hmat_g = np.diag(1/(m_H*dR**2) + 0.5*k_g*(R - R_eq_g)**2) + np.diag(kin,1) + np.diag(kin,-1)
Hmat_e = np.diag(1/(m_H*dR**2) + 0.5*k_e*(R - R_eq_e)**2) + np.diag(kin,1) + np.diag(kin,-1)

evals_g, _ = eigh_tridiagonal(np.diag(Hmat_g), np.diag(Hmat_g,1))
evals_e, _ = eigh_tridiagonal(np.diag(Hmat_e), np.diag(Hmat_e,1))
E_vib_g = E_min_g + evals_g[:4]
E_vib_e = E_min_e + evals_e[:4]

# ------------------------------------------------------------
# 3. Побудова графіків
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
ax1, ax2 = axes

# --- (A) Потенціальні криві з PySCF ---
ax1.plot(Rvals, E_gs, 'b-', lw=2, label='Основний стан $X^1\\Sigma_g^+$')
ax1.plot(Rvals, E_es, 'r-', lw=2, label='Збуджений стан $B^1\\Sigma_u^+$')
ax1.set_xlabel('Відстань H–H, Å')
ax1.set_ylabel('Енергія, Hartree')
ax1.set_title('Потенціальні криві H₂ (CASSCF(2,2))')
ax1.legend()
ax1.grid(True)

# --- (B) Принцип Франка–Кондона ---
ax2.plot(R, Vg, 'b', label='Основний стан')
ax2.plot(R, Ve, 'r', label='Збуджений стан')
ax2.hlines(E_vib_g, R.min(), R.max(), colors='b', linestyles='--')
ax2.hlines(E_vib_e, R.min(), R.max(), colors='r', linestyles='--')

# Вертикальні переходи
R0 = R_eq_g
E_start = np.interp(R0, R, Vg)
E_end = np.interp(R0, R, Ve)
ax2.plot([R0, R0], [E_start, E_end], 'k-', lw=2)
ax2.text(R0+0.02, (E_start+E_end)/2, 'FC перехід', rotation=90, va='center')

ax2.set_xlabel('R (Å)')
ax2.set_ylabel('E (Hartree)')
ax2.set_title('Ілюстрація принципу Франка–Кондона')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

