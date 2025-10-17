# ============================================================
# code_39.py - Гетероядерні молекули (CO, HF, LiH)
# ============================================================
from pyscf import gto, dft
import numpy as np

molecules = {
    'CO': {
        'atom': 'C 0 0 0; O 0 0 1.128',
        'exp_r': 1.128,
        'exp_D': 11.16,  # eV
        'description': 'Полярний потрійний зв\'язок'
    },
    'HF': {
        'atom': 'H 0 0 0; F 0 0 0.917',
        'exp_r': 0.917,
        'exp_D': 6.12,
        'description': 'Сильно полярний одинарний зв\'язок'
    },
    'LiH': {
        'atom': 'Li 0 0 0; H 0 0 1.596',
        'exp_r': 1.596,
        'exp_D': 2.43,
        'description': 'Іонний характер зв\'язку'
    }
}

print("Дослідження гетероядерних молекул")
print("=" * 80)

functionals = ['pbe', 'b3lyp', 'pbe0']

for mol_name, data in molecules.items():
    print(f"\n{mol_name}: {data['description']}")
    print("-" * 80)

    mol = gto.M(
        atom=data['atom'],
        basis='aug-cc-pvtz',
        unit='angstrom'
    )

    print(f"{'Функціонал':<12} {'Енергія (Ha)':<16} {'Дипольний момент (D)':<25}")
    print("-" * 80)

    for xc in functionals:
        mf = dft.RKS(mol)
        mf.xc = xc
        mf.verbose = 0
        energy = mf.kernel()

        # Дипольний момент
        dipole = mf.dip_moment(unit='Debye')
        dip_mag = np.linalg.norm(dipole)

        print(f"{xc:<12} {energy:<16.8f} {dip_mag:<8.4f}")

    print(f"\nЕксперимент: r_e = {data['exp_r']:.3f} Å, D_e = {data['exp_D']:.2f} eV")


