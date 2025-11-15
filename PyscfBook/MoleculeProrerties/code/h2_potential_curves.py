# ============================================================
# h2_potential_curves.py
# Розрахунок потенціальних кривих для H₂ молекули за допомогою CASSCF
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, mcscf

# --- 1. Параметри системи ---
Rvals = np.linspace(0.5, 3.0, 40)  # відстані H-H в ангстремах
E_gs, E_es = [], []

# --- 2. Основний стан (1σg²) і збуджений (1σg 1σu) ---
for R in Rvals:
    mol = gto.Mole()
    mol.atom = f"H 0 0 0; H 0 0 {R}"
    mol.basis = "6-31g"
    mol.unit = "Angstrom"
    mol.spin = 0
    mol.build()

    mf = scf.RHF(mol).run()

    # CASSCF(2,2): 2 електрони у 2 орбіталях (σg, σu)
    mc = mcscf.CASSCF(mf, ncas=2, nelecas=2)
    mc.state_average_([0.5, 0.5])  # усереднення по двох станах
    e_mc = mc.kernel()
    e_states = mc.e_states  # енергії станів (всього, включаючи HF)
    E_gs.append(e_states[0])
    E_es.append(e_states[1])

E_gs = np.array(E_gs)
E_es = np.array(E_es)

# --- 3. Побудова потенціальних кривих ---
plt.figure(figsize=(8,6))
plt.plot(Rvals, E_gs, 'b-', label='Основний стан $X^1\\Sigma_g^+$')
plt.plot(Rvals, E_es, 'r-', label='Збуджений стан $B^1\\Sigma_u^+$')
plt.xlabel('Відстань H–H, Å')
plt.ylabel('Енергія, Hartree')
plt.legend()
plt.title('Потенціальні криві H₂ (CASSCF(2,2))')
plt.grid(True)
plt.tight_layout()
plt.show()

