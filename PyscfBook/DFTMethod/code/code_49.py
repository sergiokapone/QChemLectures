# ============================================================
# code_49.py - Систематичне дослідження галогенів
# ============================================================
from pyscf import gto, dft
import numpy as np

print("Систематичне дослідження галогенів X2")
print("=" * 80)

halogens = {
    'F2': {'r': 1.412, 'D_exp': 1.66, 'basis': 'aug-cc-pvtz'},
    'Cl2': {'r': 1.988, 'D_exp': 2.51, 'basis': 'aug-cc-pvtz'},
    'Br2': {'r': 2.281, 'D_exp': 1.99, 'basis': 'aug-cc-pvtz-pp'},  # PP для Br
    'I2': {'r': 2.666, 'D_exp': 1.54, 'basis': 'aug-cc-pvtz-pp'},   # PP для I
}

functionals_hal = ['pbe', 'b3lyp', 'pbe0', 'wb97x-d']

print(f"\n{'Молекула':<10} {'Функціонал':<12} {'E (Ha)':<18} "
      f"{'D_e calc (eV)':<15} {'D_e exp (eV)':<15}")
print("-" * 80)

for mol_name, data in halogens.items():
    atom1 = mol_name[:1]
    r = data['r']
    basis = data['basis']
    D_exp = data['D_exp']

    for xc in functionals_hal:
        # Молекула
        mol_X2 = gto.M(
            atom=f'{atom1} 0 0 0; {atom1} 0 0 {r}',
            basis=basis,
            unit='angstrom'
        )

        mf = dft.RKS(mol_X2)
        mf.xc = xc
        mf.verbose = 0
        E_X2 = mf.kernel()

        # Атом
        mol_X = gto.M(
            atom=f'{atom1} 0 0 0',
            basis=basis
        )

        mf_atom = dft.RKS(mol_X)
        mf_atom.xc = xc
        mf_atom.verbose = 0
        E_X = mf_atom.kernel()

        # Енергія дисоціації
        D_calc = (2 * E_X - E_X2) * 27.211

        print(f"{mol_name:<10} {xc:<12} {E_X2:<18.8f} {D_calc:<15.4f} {D_exp:<15.4f}")

print("\n" + "=" * 80)
print("Спостереження:")
print("1. F2 - найскладніша (слабкий зв'язок, багато кореляції)")
print("2. Дисперсійні корекції важливі (wb97x-d)")
print("3. Для Br, I потрібні релятивістські псевдопотенціали")

