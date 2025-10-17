# ============================================================
# code_50.py - Порівняння часу обчислень
# ============================================================
from pyscf import gto, dft, scf
import time
import numpy as np

print("Порівняння часу обчислень для CO")
print("=" * 80)

basis_sets_time = ['cc-pvdz', 'cc-pvtz', 'cc-pvqz', 'aug-cc-pvqz']
methods = [
    ('HF', None),
    ('LDA', 'lda,vwn'),
    ('PBE', 'pbe'),
    ('B3LYP', 'b3lyp'),
    ('PBE0', 'pbe0'),
    ('M06-2X', 'm06-2x'),
]

print(f"\n{'Метод':<12} {'Базис':<15} {'N_AO':<8} {'Час (с)':<12} "
      f"{'Енергія (Ha)':<18}")
print("-" * 80)

for basis in basis_sets_time:
    mol = gto.M(
        atom='C 0 0 0; O 0 0 1.128',
        basis=basis,
        unit='angstrom'
    )

    n_ao = mol.nao

    for method_name, xc in methods:
        t0 = time.time()

        if xc is None:  # HF
            mf = scf.RHF(mol)
        else:
            mf = dft.RKS(mol)
            mf.xc = xc

        mf.verbose = 0
        energy = mf.kernel()

        t1 = time.time()
        elapsed = t1 - t0

        print(f"{method_name:<12} {basis:<15} {n_ao:<8} {elapsed:<12.4f} {energy:<18.8f}")

    print("-" * 80)

print("\nВисновки:")
print("1. LDA, PBE найшвидші (без точного обміну)")
print("2. Гібриди (B3LYP, PBE0) повільніші через HF обмін")
print("3. M06-2X найповільніший (54% HF + meta-GGA)")
print("4. Час масштабується як O(N³) для чистих DFT")
print("5. Час масштабується як O(N⁴) для гібридів")

