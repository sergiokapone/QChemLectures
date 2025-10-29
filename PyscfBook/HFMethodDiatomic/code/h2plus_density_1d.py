# ============================================================
# h2plus_density_1d.py
# ============================================================
#
# 1D профіль електронної густини уздовж осі z
# для зв’язувальної (σ_g) і антизв’язувальної (σ_u*) МО іона H2+
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf

# --- 1. Молекула ---------------------------------------------
mol = gto.Mole()
mol.atom = '''
H 0 0 -0.37
H 0 0  0.37
'''
mol.basis = 'sto-3g'
mol.charge = 1
mol.spin = 1
mol.build()

# --- 2. SCF (UHF, бо один електрон) ---------------------------
mf = scf.UHF(mol)
mf.kernel()

# --- 3. Створюємо сітку вздовж осі z -------------------------
z = np.linspace(-3.0, 3.0, 400)  # у Борових радіусах
coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])

# --- 4. Обчислюємо значення МО -------------------------------
# α-спіновий набір орбіталей
mo_coeff = mf.mo_coeff[0]

# перша — σ_g, друга — σ_u*
ao_values = mol.eval_gto('GTOval_sph', coords)
psi_g = ao_values @ mo_coeff[:, 0]
psi_u = ao_values @ mo_coeff[:, 1]

# густина (|ψ|²)
rho_g = psi_g**2
rho_u = psi_u**2

# --- 5. Побудова графіка -------------------------------------
plt.figure(figsize=(7, 4))
plt.plot(z, rho_g, label=r'Зв’язувальна $\sigma_g$', lw=2)
plt.plot(z, rho_u, label=r'Антизв’язувальна $\sigma_u^*$', lw=2)
plt.axvline(-0.37, color='k', ls='--', lw=0.8)
plt.axvline( 0.37, color='k', ls='--', lw=0.8)
plt.text(-0.37, max(rho_g)*0.9, 'H$_A$', ha='center', va='top')
plt.text( 0.37, max(rho_g)*0.9, 'H$_B$', ha='center', va='top')

plt.xlabel('z')
plt.ylabel(r'$\rho(z) = |\psi(z)|^2$')
plt.title(r'Густина електронного заряду уздовж осі $z$ для $H_2^+$')
plt.legend()
plt.show()

