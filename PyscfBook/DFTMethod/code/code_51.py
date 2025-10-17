# ============================================================
# code_51.py - Тестовий набір для оцінки функціоналів
# ============================================================
from pyscf import gto, dft
import numpy as np

print("Тестовий набір G2 (вибірка диатомних молекул)")
print("=" * 80)

# Експериментальні дані (r_e в Å, D_e в eV)
g2_molecules = {
    'H2':  {'geom': 'H 0 0 0; H 0 0 0.741',   'r_exp': 0.741,  'D_exp': 4.75},
    'Li2': {'geom': 'Li 0 0 0; Li 0 0 2.673', 'r_exp': 2.673,  'D_exp': 1.05},
    'LiH': {'geom': 'Li 0 0 0; H 0 0 1.596',  'r_exp': 1.596,  'D_exp': 2.43},
    'N2':  {'geom': 'N 0 0 0; N 0 0 1.098',   'r_exp': 1.098,  'D_exp': 9.91},
    'O2':  {'geom': 'O 0 0 0; O 0 0 1.208',   'r_exp': 1.208,  'D_exp': 5.21, 'spin': 2},
    'F2':  {'geom': 'F 0 0 0; F 0 0 1.412',   'r_exp': 1.412,  'D_exp': 1.66},
    'CO':  {'geom': 'C 0 0 0; O 0 0 1.128',   'r_exp': 1.128,  'D_exp': 11.16},
    'NO':  {'geom': 'N 0 0 0; O 0 0 1.151',   'r_exp': 1.151,  'D_exp': 6.51, 'spin': 1},
}

functionals_test = ['pbe', 'b3lyp', 'pbe0', 'm06-2x']
basis = 'aug-cc-pvtz'

# Зберігаємо результати
results_test = {xc: {'D_errors': [], 'names': []} for xc in functionals_test}

print(f"\nБазис: {basis}")
print("=" * 80)

for mol_name, data in g2_molecules.items():
    spin = data.get('spin', 0)
    D_exp = data['D_exp']

    print(f"\n{mol_name}:")
    print("-" * 40)

    for xc in functionals_test:
        # Молекула
        mol = gto.M(
            atom=data['geom'],
            basis=basis,
            unit='angstrom',
            spin=spin
        )

        if spin == 0:
            mf_mol = dft.RKS(mol)
        else:
            mf_mol = dft.UKS(mol)
        mf_mol.xc = xc
        mf_mol.verbose = 0
        E_mol = mf_mol.kernel()

        # Атоми (розібрати геометрію)
        atoms = data['geom'].split(';')
        atom1 = atoms[0].strip().split()[0]
        atom2 = atoms[1].strip().split()[0]

        # Енергія атомів
        E_atoms = 0
        for atom_name in [atom1, atom2]:
            mol_atom = gto.M(atom=f'{atom_name} 0 0 0', basis=basis)

            # Визначення спіну атома
            atom_spins = {'H': 1, 'Li': 1, 'N': 3, 'O': 2, 'F': 1, 'C': 2}
            atom_spin = atom_spins.get(atom_name, 0)

            if atom_spin == 0:
                mf_atom = dft.RKS(mol_atom)
            else:
                mf_atom = dft.UKS(mol_atom)
            mf_atom.xc = xc
            mf_atom.verbose = 0
            E_atoms += mf_atom.kernel()

        # Енергія дисоціації
        D_calc = (E_atoms - E_mol) * 27.211
        error = D_calc - D_exp

        results_test[xc]['D_errors'].append(error)
        results_test[xc]['names'].append(mol_name)

        print(f"  {xc:<10}: D_e = {D_calc:6.3f} eV (Δ = {error:+6.3f} eV)")

# Статистика
print("\n" + "=" * 80)
print("Статистика помилок D_e (eV):")
print("=" * 80)
print(f"{'Функціонал':<12} {'MAE':<10} {'RMSE':<10} {'Max|error|':<12}")
print("-" * 80)

for xc in functionals_test:
    errors = np.array(results_test[xc]['D_errors'])
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    max_err = np.max(np.abs(errors))

    print(f"{xc:<12} {mae:<10.4f} {rmse:<10.4f} {max_err:<12.4f}")

print("\nВисновок: B3LYP та PBE0 зазвичай найкращі для термохімії")

