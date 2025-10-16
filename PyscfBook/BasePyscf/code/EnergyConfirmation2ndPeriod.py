from pyscf import gto, scf

# Правильні спінові стани для другого періоду
atoms_2nd_period = [
    ('Li', 1), ('Be', 0), ('B', 1), ('C', 2),
    ('N', 3), ('O', 2), ('F', 1), ('Ne', 0)
]

print('Основні стани атомів другого періоду:\n')
for symbol, spin in atoms_2nd_period:
    mol = gto.M(
        atom=f'{symbol} 0 0 0',
        basis='6-31g',
        spin=spin
    )
    mf = scf.UHF(mol) if spin > 0 else scf.RHF(mol)
    mf.verbose = 0
    energy = mf.kernel()

    mult = spin + 1
    print(f'{symbol:2s}: 2S={spin}, M={mult}, E={energy:12.6f} Ha')
