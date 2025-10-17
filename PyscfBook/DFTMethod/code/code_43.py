# ============================================================
# code_43.py - Відкритооболонкові молекули (радикали)
# ============================================================
from pyscf import gto, dft
import numpy as np

print("Відкритооболонкові молекули (радикали)")
print("=" * 80)

radicals = [
    ('OH', 'O 0 0 0; H 0 0 0.97', 1, '²Π'),
    ('NO', 'N 0 0 0; O 0 0 1.15', 1, '²Π'),
    ('O2', 'O 0 0 0; O 0 0 1.21', 2, '³Σg⁻'),
    ('CN', 'C 0 0 0; N 0 0 1.17', 1, '²Σ⁺'),
]

print(f"\n{'Молекула':<10} {'Стан':<10} {'Спін':<8} {'E(UKS) Ha':<18} "
      f"{'<S²>':<12} {'Забруднення':<15}")
print("-" * 80)

for name, geom, spin, state in radicals:
    mol = gto.M(
        atom=geom,
        basis='cc-pvtz',
        unit='angstrom',
        spin=spin
    )

    mf = dft.UKS(mol)
    mf.xc = 'pbe0'
    mf.verbose = 0
    energy = mf.kernel()

    # Спінове забруднення
    s_squared = mf.spin_square()[0]
    s_exact = spin * (spin + 2) / 4  # S(S+1) де S = spin/2
    contamination = s_squared - s_exact

    print(f"{name:<10} {state:<10} {spin:<8} {energy:<18.8f} "
          f"{s_squared:<12.6f} {contamination:<15.6f}")

print("\nПримітка:")
print("- UKS не має спінового забруднення (на відміну від UHF)")
print("- <S²> = S(S+1) = (M_s)(M_s + 2)/4 для чистого стану")
print("- Забруднення ≈ 0 для DFT")

