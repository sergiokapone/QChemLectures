from pyscf import gto

mol = gto.M(atom='C 0 0 0', basis='6-31g')

# Детальна інформація
print('Базисні функції по типу:')
for atom_idx in range(mol.natm):
    symbol = mol.atom_symbol(atom_idx)
    print(f'\nАтом {symbol}:')

    # Кількість функцій кожного типу
    basis_atom = mol._basis[symbol]
    for shell in basis_atom:
        l_quantum = shell[0]  # Азимутальне квантове число
        n_contractions = len(shell[1:])

        shell_type = ['s', 'p', 'd', 'f', 'g'][l_quantum]
        print(f'  {shell_type}-тип: {n_contractions} contracted')

# Діапазони базисних функцій для атома
ao_labels = mol.ao_labels()
print(f'\nМітки базисних функцій:')
for i, label in enumerate(ao_labels):
    print(f'{i}: {label}')
