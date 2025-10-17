# ============================================================
# code_34.py - Базовий DFT розрахунок молекули H2
# ============================================================
from pyscf import gto, dft
import numpy as np

# Створення молекули H2
mol = gto.M(
    atom='H 0 0 0; H 0 0 0.74',  # Відстань ~0.74 Å
    basis='cc-pvdz',
    unit='angstrom',
    symmetry=True
)

print("Молекула H2")
print("=" * 50)
print(f"Кількість електронів: {mol.nelectron}")
print(f"Спін: {mol.spin}")
print(f"Точкова група: {mol.groupname}")
print()

# DFT розрахунок з різними функціоналами
functionals = ['lda,vwn', 'pbe', 'b3lyp', 'pbe0']

for xc in functionals:
    mf = dft.RKS(mol)
    mf.xc = xc
    energy = mf.kernel()

    print(f"{xc:15s}: E = {energy:15.8f} Hartree")

print("\nПорівняння з HF:")
from pyscf import scf
mf_hf = scf.RHF(mol).run()
print(f"{'HF':15s}: E = {mf_hf.e_tot:15.8f} Hartree")

