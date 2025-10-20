# ============================================================
# mp2_example.py
# Демонстрація: розрахунок енергії MP2 для молекули води
# ============================================================

from pyscf import gto, scf, mp

# --- 1. Задання молекули ---
mol = gto.Mole()
mol.atom = '''
O  0.0000  0.0000  0.0000
H  0.0000  0.7570  0.5870
H  0.0000 -0.7570  0.5870
'''
mol.basis = 'cc-pVDZ'
mol.unit = 'Angstrom'
mol.build()

# --- 2. Розрахунок Хартрі–Фока ---
mf = scf.RHF(mol)
mf.verbose = 0
E_HF = mf.kernel()
print(f'Hartree–Fock energy = {E_HF:.10f} Hartree')

# --- 3. Увімкнення MP2 ---
mp2_calc = mp.MP2(mf)
mp2_calc.verbose = 0
mp2_result = mp2_calc.kernel()  # Повертає (E_corr, T2_amplitudes)
E_corr = mp2_result[0]  # Кореляційна енергія (float)
E_MP2 = E_HF + E_corr   # Загальна MP2 енергія

print(f'MP2 total energy      = {E_MP2:.10f} Hartree')
print(f'MP2 correlation energy = {E_corr:.10f} Hartree')

