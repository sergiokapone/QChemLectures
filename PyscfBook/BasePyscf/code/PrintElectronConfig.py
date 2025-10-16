from pyscf import gto

def print_electron_config(atom_symbol, charge=0, spin=0):
    mol = gto.M(
        atom=f'{atom_symbol} 0 0 0',
        basis='sto-3g',
        charge=charge,
        spin=spin
    )

    n_alpha, n_beta = mol.nelec
    print(f'{atom_symbol} (charge={charge}, 2S={spin}):')
    print(f'  Всього електронів: {mol.nelectron}')
    print(f'  α-електронів: {n_alpha}')
    print(f'  β-електронів: {n_beta}')
    print(f'  Неспарених: {n_alpha - n_beta}\n')

# Приклади
print_electron_config('Li', charge=0, spin=1)    # Li (2s¹)
print_electron_config('C',  charge=0, spin=2)    # C (³P)
print_electron_config('O',  charge=0, spin=2)    # O (³P)
print_electron_config('O',  charge=-1, spin=1)   # O⁻ (дублет)
