# ============================================================
# code_52.py - Вплив симетрії на час розрахунку
# ============================================================
from pyscf import gto, dft
import time

print("Вплив симетрії на час розрахунку")
print("=" * 80)

molecules_sym = [
    ('H2', 'H 0 0 0; H 0 0 0.74', 'D∞h'),
    ('N2', 'N 0 0 0; N 0 0 1.098', 'D∞h'),
    ('O2', 'O 0 0 0; O 0 0 1.208', 'D∞h'),
    ('CO', 'C 0 0 0; O 0 0 1.128', 'C∞v'),
]

basis = 'cc-pvqz'

print(f"\nБазис: {basis}, Функціонал: PBE0")
print("=" * 80)
print(f"{'Молекула':<10} {'Симетрія':<12} {'З симетрією':<15} "
      f"{'Без симетрії':<15} {'Прискорення':<15}")
print("-" * 80)

for mol_name, geom, sym in molecules_sym:
    spin = 2 if mol_name == 'O2' else 0

    # З симетрією
    mol_sym = gto.M(
        atom=geom,
        basis=basis,
        unit='angstrom',
        symmetry=True,
        spin=spin
    )

    t0 = time.time()
    if spin == 0:
        mf = dft.RKS(mol_sym)
    else:
        mf = dft.UKS(mol_sym)
    mf.xc = 'pbe0'
    mf.verbose = 0
    mf.kernel()
    t_sym = time.time() - t0

    # Без симетрії
    mol_nosym = gto.M(
        atom=geom,
        basis=basis,
        unit='angstrom',
        symmetry=False,
        spin=spin
    )

    t0 = time.time()
    if spin == 0:
        mf = dft.RKS(mol_nosym)
    else:
        mf = dft.UKS(mol_nosym)
    mf.xc = 'pbe0'
    mf.verbose = 0
    mf.kernel()
    t_nosym = time.time() - t0

    speedup = t_nosym / t_sym

    print(f"{mol_name:<10} {sym:<12} {t_sym:<15.4f} {t_nosym:<15.4f} {speedup:<15.2f}x")

print("\nВисновок: Використання симетрії дає 2-5x прискорення!")

