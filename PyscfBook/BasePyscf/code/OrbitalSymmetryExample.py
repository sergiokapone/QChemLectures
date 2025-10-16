from pyscf import gto, scf

mol = gto.M(
    atom='N 0 0 0',
    basis='cc-pvdz',
    spin=3,  # Основний стан ⁴S
    symmetry=True
)

mf = scf.UHF(mol)
mf.kernel()

# Орбітальна симетрія
if mol.symmetry:
    orbsym = scf.hf_symm.get_orbsym(mol, mf.mo_coeff[0])
    from pyscf.symm import label_orb_symm

    labels = label_orb_symm(mol, mol.irrep_name, mol.symm_orb,
                           mf.mo_coeff[0])

    print('\nСиметрія заповнених орбіталей (альфа):')
    for i in range(mol.nelec[0]):
        print(f'  MO {i+1}: {labels[i]}')
