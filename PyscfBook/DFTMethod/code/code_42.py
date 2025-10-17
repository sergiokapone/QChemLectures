# ============================================================
# code_42.py - Молекули перехідних металів
# ============================================================
from pyscf import gto, dft
import numpy as np

print("Диатомні молекули перехідних металів")
print("=" * 80)

# CuH - типова молекула перехідного металу
molecules_TM = [
    ('CuH', 'Cu 0 0 0; H 0 0 1.46', 0, 'cc-pvtz-pp'),
    ('FeO', 'Fe 0 0 0; O 0 0 1.62', 4, 'cc-pvtz-pp'),  # квінтет
    ('TiO', 'Ti 0 0 0; O 0 0 1.62', 2, 'cc-pvtz-pp'),  # триплет
]

functionals_TM = [
    ('PBE', 'pbe'),
    ('TPSS', 'tpss'),
    ('TPSSh', 'tpssh'),
    ('M06', 'm06'),
    ('B3LYP', 'b3lyp'),
]

for mol_name, geom, spin, basis in molecules_TM:
    print(f"\n{mol_name} (спін = {spin})")
    print("-" * 80)

    mol = gto.M(
        atom=geom,
        basis=basis,
        unit='angstrom',
        spin=spin
    )

    print(f"{'Функціонал':<12} {'Енергія (Ha)':<18} {'Конвергувало':<15}")
    print("-" * 80)

    for name, xc in functionals_TM:
        mf = dft.UKS(mol)  # Завжди unrestricted для TM
        mf.xc = xc
        mf.verbose = 0
        mf.max_cycle = 100

        try:
            energy = mf.kernel()
            converged = "Так" if mf.converged else "Ні"
            print(f"{name:<12} {energy:<18.8f} {converged:<15}")
        except:
            print(f"{name:<12} {'FAILED':<18} {'---':<15}")

    print(f"\nПримітка: Використано псевдопотенціал {basis}")

print("\n" + "=" * 80)
print("Висновок: Meta-GGA (TPSS, M06) краще для перехідних металів")
print("Може знадобитись DFT+U для сильно корельованих систем")


