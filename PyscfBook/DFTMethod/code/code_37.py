# ============================================================
# code_37.py - Порівняння функціоналів для N2
# ============================================================
from pyscf import gto, dft, scf
import numpy as np

# N2 молекула з експериментальною геометрією
mol = gto.M(
    atom='N 0 0 0; N 0 0 1.098',
    basis='aug-cc-pvtz',
    unit='angstrom',
    symmetry=True
)

print("Порівняння функціоналів для N2")
print("=" * 70)
print(f"Базис: {mol.basis}")
print(f"Геометрія: r(N-N) = 1.098 Å (експеримент)")
print("=" * 70)

functionals = [
    ('LDA', 'lda,vwn'),
    ('PBE', 'pbe'),
    ('BLYP', 'blyp'),
    ('B3LYP', 'b3lyp'),
    ('PBE0', 'pbe0'),
    ('M06-2X', 'm06-2x'),
    ('ωB97X-D', 'wb97x-d')
]

print(f"\n{'Функціонал':<15} {'Енергія (Ha)':<18} {'Час (с)':<10}")
print("-" * 70)

for name, xc in functionals:
    import time
    t0 = time.time()

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.verbose = 0
    energy = mf.kernel()

    t1 = time.time()
    print(f"{name:<15} {energy:<18.8f} {t1-t0:<10.3f}")

# Порівняння з HF
print("-" * 70)
t0 = time.time()
mf_hf = scf.RHF(mol)
mf_hf.verbose = 0
e_hf = mf_hf.kernel()
t1 = time.time()
print(f"{'HF':<15} {e_hf:<18.8f} {t1-t0:<10.3f}")

print("\n" + "=" * 70)
print("Примітка: N2 має потрійний зв'язок - складна система")
print("Кореляція критична для правильного опису зв'язку")

