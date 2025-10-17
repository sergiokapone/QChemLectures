# ============================================================
# code_40.py - Молекули з кратними зв'язками
# ============================================================
from pyscf import gto, dft
import numpy as np

print("Молекули з кратними зв'язками")
print("=" * 80)

molecules = [
    ('H2', 'H 0 0 0; H 0 0 0.74', 'одинарний'),
    ('N2', 'N 0 0 0; N 0 0 1.098', 'потрійний'),
    ('O2', 'O 0 0 0; O 0 0 1.208', 'подвійний (триплет!)'),
    ('F2', 'F 0 0 0; F 0 0 1.412', 'одинарний (слабкий)'),
]

print(f"\n{'Молекула':<10} {'Зв\'язок':<15} {'Спін':<8} {'E(PBE0) Ha':<18} "
      f"{'E(B3LYP) Ha':<18}")
print("-" * 80)

for name, geom, bond_type in molecules:
    # Визначення спіну
    if name == 'O2':
        spin = 2  # Триплет
    else:
        spin = 0  # Синглет

    mol = gto.M(
        atom=geom,
        basis='cc-pvtz',
        unit='angstrom',
        spin=spin
    )

    # PBE0
    if spin == 0:
        mf1 = dft.RKS(mol)
    else:
        mf1 = dft.UKS(mol)
    mf1.xc = 'pbe0'
    mf1.verbose = 0
    e1 = mf1.kernel()

    # B3LYP
    if spin == 0:
        mf2 = dft.RKS(mol)
    else:
        mf2 = dft.UKS(mol)
    mf2.xc = 'b3lyp'
    mf2.verbose = 0
    e2 = mf2.kernel()

    print(f"{name:<10} {bond_type:<15} {spin:<8} {e1:<18.8f} {e2:<18.8f}")

print("\nВажливо: O2 має основний стан ³Σg⁻ (триплет)")
print("Це унікальна властивість, пов'язана з молекулярними орбіталями π*")

