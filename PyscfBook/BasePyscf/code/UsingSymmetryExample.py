from pyscf import gto, scf

# З симетрією (швидше, менше пам'яті)
mol_sym = gto.M(
    atom='Ar 0 0 0',
    basis='cc-pvdz',
    symmetry=True
)
mf_sym = scf.RHF(mol_sym)
e_sym = mf_sym.kernel()

# Без симетрії
mol_nosym = gto.M(
    atom='Ar 0 0 0',
    basis='cc-pvdz',
    symmetry=False
)
mf_nosym = scf.RHF(mol_nosym)
e_nosym = mf_nosym.kernel()

print(f'З симетрією:  {e_sym:.8f} Ha')
print(f'Без симетрії: {e_nosym:.8f} Ha')
print(f'Різниця: {abs(e_sym - e_nosym):.2e} Ha')
