# ============================================================
# cisd_basic.py
# Демонстрація: CISD для He з таблицею порівняння
# ============================================================

from pyscf import gto, scf
from pyscf.ci import cisd
import numpy as np

# --- Задання атома He ---
mol = gto.Mole()
mol.atom = 'He 0.0 0.0 0.0'
mol.basis = 'cc-pVDZ'
mol.build()

# --- HF ---
mf = scf.RHF(mol)
E_HF = mf.kernel()
print(f'Енергія Hartree–Fock: {E_HF:.8f} Ha')

# --- CISD ---
myci = cisd.CISD(mf)
E_corr = myci.kernel()[0]
print(f'Кореляційна енергія CISD:        {E_corr :.8f} Ha')

# --- Таблиця порівняння ---
print('\n' + '='*50)
print('ПОРІВНЯННЯ З ЕКСПЕРИМЕНТОМ (full CI limit)')
print('='*50)
print(f"{'Метод':<15} {'E (Ha)':<15} {'ΔE від експ. (mHa)':<20}")
print('-'*50)
E_exp = -2.9037243770  # Експериментальне значення (basis set limit)
delta_HF = (E_HF - E_exp) * 1000
delta_CISD = (E_HF + E_corr - E_exp) * 1000
print(f"{'HF':<15} {E_HF:<15.8f} {delta_HF:<20.2f}")
print(f"{'CISD':<15} {E_HF + E_corr:<15.8f} {delta_CISD:<20.2f}")
print(f"{'Експеримент':<15} {E_exp:<15.8f} {'0.00':<20}")
print('='*50)
