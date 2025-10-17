# ============================================================
# code_41.py - Порівняння HF та DFT для молекул
# ============================================================
from pyscf import gto, scf, dft
import numpy as np

molecules = {
    'H2': ('H 0 0 0; H 0 0 0.74', 0),
    'LiH': ('Li 0 0 0; H 0 0 1.596', 0),
    'N2': ('N 0 0 0; N 0 0 1.098', 0),
    'CO': ('C 0 0 0; O 0 0 1.128', 0),
    'O2': ('O 0 0 0; O 0 0 1.208', 2),  # триплет!
}

print("Систематичне порівняння HF та DFT")
print("=" * 90)
print(f"{'Молекула':<10} {'HF':<16} {'LDA':<16} {'PBE':<16} "
      f"{'B3LYP':<16} {'PBE0':<16}")
print("-" * 90)

for name, (geom, spin) in molecules.items():
    mol = gto.M(
        atom=geom,
        basis='cc-pvtz',
        unit='angstrom',
        spin=spin
    )

    energies = []

    # HF
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)
    mf.verbose = 0
    energies.append(mf.kernel())

    # DFT функціонали
    for xc in ['lda,vwn', 'pbe', 'b3lyp', 'pbe0']:
        if spin == 0:
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)
        mf.xc = xc
        mf.verbose = 0
        energies.append(mf.kernel())

    print(f"{name:<10} " + " ".join(f"{e:<16.8f}" for e in energies))

print("\nСпостереження:")
print("1. HF систематично дає вищі (менш від'ємні) енергії")
print("2. LDA зазвичай переоцінює енергію зв'язку")
print("3. Гібриди (B3LYP, PBE0) дають найкращий баланс")
print("4. Для O2 (триплет) різниця між методами більша")

